"""PostDiffIO 使用的紧耦合误差状态 EKF。

状态包含位置、速度、姿态四元数、加速度计偏置和陀螺仪偏置。预测阶段使用
IMU 捷联惯导动力学，更新阶段使用 PostDiffIO 输出的速度均值和协方差。
该模块用于序列级后处理，不参与通用模型训练流程的自动求导。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# 四元数辅助函数。


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """将四元数转换为从机体系到世界系的旋转矩阵。"""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def small_angle_to_quat(omega_dt: np.ndarray) -> np.ndarray:
    """将小旋转向量转换为四元数。"""
    angle = np.linalg.norm(omega_dt)
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega_dt / angle
    half = angle / 2
    return np.array([np.cos(half), *(axis * np.sin(half))])


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


# EKF 状态定义。


class EKFState:
    """保存 EKF 状态；误差状态使用十五维表示。"""

    __slots__ = ("pos", "vel", "q", "b_a", "b_g", "P")

    def __init__(self, pos=None, vel=None, q=None, b_a=None, b_g=None, P=None):
        self.pos = np.zeros(3) if pos is None else pos.astype(np.float64)
        self.vel = np.zeros(3) if vel is None else vel.astype(np.float64)
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) if q is None else q.astype(np.float64)
        self.b_a = np.zeros(3) if b_a is None else b_a.astype(np.float64)
        self.b_g = np.zeros(3) if b_g is None else b_g.astype(np.float64)
        # 十五维误差协方差：位置、速度、姿态、加速度计偏置和陀螺仪偏置。
        self.P = np.eye(15) * 0.01 if P is None else P.astype(np.float64)


# EKF 动力学与更新。


GRAVITY_W = np.array([0.0, 0.0, -9.81])


def predict(state: EKFState, accel_b: np.ndarray, gyro_b: np.ndarray, dt: float,
            sigma_a: float = 0.1, sigma_g: float = 0.01,
            sigma_ba: float = 1e-4, sigma_bg: float = 1e-5) -> EKFState:
    """使用捷联惯导动力学执行单步 EKF 预测。

    参数：
        accel_b：机体系加速度计读数。
        gyro_b：机体系陀螺仪读数。
        dt：时间间隔。
    """
    # 去除传感器偏置。
    a = accel_b - state.b_a
    w = gyro_b - state.b_g

    R_wb = quat_to_rot(state.q)
    a_w = R_wb @ a + GRAVITY_W

    # 使用中点欧拉法更新状态。
    new_pos = state.pos + state.vel * dt + 0.5 * a_w * dt * dt
    new_vel = state.vel + a_w * dt
    dq = small_angle_to_quat(w * dt)
    new_q = quat_mul(state.q, dq)
    new_q = new_q / (np.linalg.norm(new_q) + 1e-12)

    # 在当前状态处线性化误差状态转移矩阵。
    F = np.eye(15)
    F[0:3, 3:6] = np.eye(3) * dt                     # dpos / dvel
    F[3:6, 6:9] = -skew(R_wb @ a) * dt               # dvel / dtheta
    F[3:6, 9:12] = -R_wb * dt                        # dvel / db_a
    F[6:9, 12:15] = -np.eye(3) * dt                  # dtheta / db_g

    # 使用对角过程噪声协方差。
    Q = np.zeros((15, 15))
    Q[3:6, 3:6] = np.eye(3) * (sigma_a ** 2) * dt    # vel noise
    Q[6:9, 6:9] = np.eye(3) * (sigma_g ** 2) * dt    # theta noise
    Q[9:12, 9:12] = np.eye(3) * (sigma_ba ** 2) * dt
    Q[12:15, 12:15] = np.eye(3) * (sigma_bg ** 2) * dt

    new_P = F @ state.P @ F.T + Q

    return EKFState(pos=new_pos, vel=new_vel, q=new_q,
                    b_a=state.b_a.copy(), b_g=state.b_g.copy(), P=new_P)


def update_velocity(state: EKFState, vel_meas: np.ndarray, vel_cov: np.ndarray) -> EKFState:
    """使用速度观测更新状态。

    参数：
        vel_meas：世界系三维速度观测。
        vel_cov：三维速度观测协方差。
    """
    # 观测矩阵直接选择速度状态。
    H = np.zeros((3, 15))
    H[:, 3:6] = np.eye(3)

    P = state.P
    S = H @ P @ H.T + vel_cov              # innovation cov
    K = P @ H.T @ np.linalg.inv(S)         # Kalman gain  (15 x 3)

    innovation = vel_meas - state.vel
    dx = K @ innovation                    # 15D error-state correction

    # 应用误差状态修正。
    new_pos = state.pos + dx[0:3]
    new_vel = state.vel + dx[3:6]
    dtheta = dx[6:9]
    dq_corr = small_angle_to_quat(dtheta)
    new_q = quat_mul(state.q, dq_corr)
    new_q = new_q / (np.linalg.norm(new_q) + 1e-12)
    new_b_a = state.b_a + dx[9:12]
    new_b_g = state.b_g + dx[12:15]

    # 使用 Joseph 形式更新协方差，提高数值稳定性。
    I = np.eye(15)
    new_P = (I - K @ H) @ P @ (I - K @ H).T + K @ vel_cov @ K.T

    return EKFState(pos=new_pos, vel=new_vel, q=new_q,
                    b_a=new_b_a, b_g=new_b_g, P=new_P)


# 序列级运行入口。


def run_ekf(
    imu: np.ndarray,
    ts: np.ndarray,
    vel_obs: np.ndarray,
    vel_obs_cov: np.ndarray,
    obs_idx: np.ndarray,
    init_state: Optional[EKFState] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """使用稀疏速度观测在完整 IMU 序列上运行 EKF。

    参数：
        imu：形状为 `[N, 6]` 的 IMU 序列。
        ts：形状为 `[N]` 的时间戳。
        vel_obs：形状为 `[M, 3]` 的世界系速度观测。
        vel_obs_cov：形状为 `[M, 3, 3]` 的观测协方差。
        obs_idx：每个速度观测对应的时间步索引。
        init_state：可选初始状态。

    返回：
        positions：形状为 `[N, 3]` 的位置序列。
        velocities：形状为 `[N, 3]` 的速度序列。
    """
    N = len(imu)
    state = init_state if init_state is not None else EKFState()
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))
    positions[0] = state.pos
    velocities[0] = state.vel

    obs_idx_set = set(obs_idx.tolist())
    obs_iter = iter(zip(obs_idx, vel_obs, vel_obs_cov))
    next_obs = next(obs_iter, None)

    for k in range(1, N):
        dt = float(ts[k] - ts[k - 1])
        if dt <= 0:
            dt = 1e-3
        accel = imu[k - 1, :3]
        gyro = imu[k - 1, 3:]
        state = predict(state, accel, gyro, dt)

        if k in obs_idx_set:
            while next_obs is not None and next_obs[0] < k:
                next_obs = next(obs_iter, None)
            if next_obs is not None and next_obs[0] == k:
                _, v_meas, v_cov = next_obs
                state = update_velocity(state, v_meas, v_cov)
                next_obs = next(obs_iter, None)

        positions[k] = state.pos
        velocities[k] = state.vel

    return positions, velocities


if __name__ == "__main__":
    # 简单检查：零 IMU 配合已知速度观测时，轨迹应跟随观测。
    np.random.seed(0)
    N = 200
    ts = np.arange(N) * 0.005  # 200 Hz
    imu = np.zeros((N, 6))
    imu[:, 2] = 9.81  # pure gravity, no motion in world frame

    obs_idx = np.array([50, 100, 150])
    vel_obs = np.array([[1.0, 0.0, 0.0]] * 3)
    vel_obs_cov = np.tile(np.eye(3) * 0.01, (3, 1, 1))

    pos, vel = run_ekf(imu, ts, vel_obs, vel_obs_cov, obs_idx)
    print(f"positions[0:5]:\n{pos[:5]}")
    print(f"positions[-1]:  {pos[-1]}")
    print(f"velocities[-1]: {vel[-1]}")
    print("EKF runs without errors.")
