
"""
G354 + 6轴 Attitude EKF 姿态解算测试版。

保留内容：
    1. G354 UART burst 配置
    2. G354 38字节 raw frame 读取
    3. 陀螺仪零偏校准
    4. 加速度 roll/pitch 初始化
    5. 姿态 EKF：名义状态 q + gyro residual bias，误差状态 [dtheta, dbg]

坐标约定：
    x：前方
    y：右手边
    z：向下，和真实重力方向相同

重要约定：
    G354 加速度原始输出是 specific force，比力。
    静止时 raw_acc ≈ -gravity_body。
    EKF 加速度观测使用真实重力方向：gravity_meas = -raw_acc / |raw_acc|。
"""

import argparse
import math
import time

import numpy as np
import serial


PORT = "/dev/ttyUSB0"
BAUDRATE = 460800
BYTEORDER = "big"

SF_GYRO = 0.016     
SF_ACC = 0.2      
SCALE_DIV = 65536.0

TARGET_HZ = 100.0
TARGET_DT = 1.0 / TARGET_HZ
DT_MIN = 0.001
DT_MAX = 0.05

GYRO_CALIB_SAMPLES = 500
ACC_INIT_SAMPLES = 200
WARMUP_SAMPLES = 300
ACC_NORM_MIN_G = 0.85
ACC_NORM_MAX_G = 1.15
CALIBRATE_GYRO_Z = True
ATT_EKF_ACC_NOISE_UNIT = 0.035
ATT_EKF_GYRO_NOISE_DPS_SQRT_HZ = 0.10
ATT_EKF_BIAS_RW_DPS_SQRT_S = 0.005
VERIFY_BURST_CONFIG = False
G354_FRAME_LEN = 38
G354_ADDR_WIN_CTRL = 0x7E
G354_ADDR_MODE_CTRL_HI = 0x03
G354_ADDR_UART_CTRL_LO = 0x08
G354_ADDR_BURST_CTRL1_LO = 0x0C
G354_ADDR_BURST_CTRL2_LO = 0x0E

G354_CMD_WINDOW0 = 0x00
G354_CMD_WINDOW1 = 0x01
G354_CMD_BEGIN_SAMPLING = 0x01
G354_CMD_END_SAMPLING = 0x02

G354_BURST_CTRL1_FIXED = 0xF007
G354_BURST_CTRL2_FIXED = 0x7000


DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def open_serial(port: str, baudrate: int) -> serial.Serial:
    return serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.2,
        write_timeout=0.2,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )


def write_g354_reg8(ser: serial.Serial, addr: int, value: int) -> None:
    cmd = bytes([
        0x80 | (addr & 0x7F),
        value & 0xFF,
        0x0D,
    ])
    ser.write(cmd)
    ser.flush()
    time.sleep(0.002)


def write_g354_reg16_parts(ser: serial.Serial, addr_lo: int, value: int) -> None:
    """G354 xxx_LO 写 D[7:0]，xxx_HI 写 D[15:8]。"""
    write_g354_reg8(ser, addr_lo, value & 0xFF)
    write_g354_reg8(ser, addr_lo + 1, (value >> 8) & 0xFF)


def read_exact(ser: serial.Serial, n: int, timeout: float = 0.2) -> bytes:
    old_timeout = ser.timeout
    ser.timeout = timeout
    data = bytearray()

    deadline = time.monotonic() + timeout
    while len(data) < n and time.monotonic() < deadline:
        chunk = ser.read(n - len(data))
        if chunk:
            data.extend(chunk)

    ser.timeout = old_timeout
    return bytes(data)


def read_g354_reg16(ser: serial.Serial, addr_even: int) -> int:
    ser.reset_input_buffer()

    cmd = bytes([
        addr_even & 0x7E,
        0x00,
        0x0D,
    ])
    ser.write(cmd)
    ser.flush()

    resp = read_exact(ser, 4, timeout=0.2)
    if len(resp) != 4:
        raise RuntimeError(f"read reg 0x{addr_even:02X} failed: got {len(resp)} bytes")

    if resp[0] != (addr_even & 0x7E) or resp[3] != 0x0D:
        raise RuntimeError(
            f"read reg 0x{addr_even:02X} bad resp: {resp.hex(' ')}"
        )

    return (resp[1] << 8) | resp[2]


def init_g354_sampling_mode(ser: serial.Serial, verify: bool = VERIFY_BURST_CONFIG) -> None:
    print(
        "Configuring G354 fixed UART burst format: "
        "BURST_CTRL1=0xF007, BURST_CTRL2=0x7000, frame_len=38"
    )

    # window0，停止采样
    write_g354_reg8(ser, G354_ADDR_WIN_CTRL, G354_CMD_WINDOW0)
    write_g354_reg8(ser, G354_ADDR_MODE_CTRL_HI, G354_CMD_END_SAMPLING)

    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # window1，配置 UART/burst
    write_g354_reg8(ser, G354_ADDR_WIN_CTRL, G354_CMD_WINDOW1)
    write_g354_reg8(ser, G354_ADDR_UART_CTRL_LO, 0x00)

    # 38字节 burst：FLAG + TEMP + GYRO + ACCEL + GPIO + COUNT + CHECKSUM
    write_g354_reg16_parts(ser, G354_ADDR_BURST_CTRL1_LO, G354_BURST_CTRL1_FIXED)
    write_g354_reg16_parts(ser, G354_ADDR_BURST_CTRL2_LO, G354_BURST_CTRL2_FIXED)

    if verify:
        burst1 = read_g354_reg16(ser, G354_ADDR_BURST_CTRL1_LO)
        burst2 = read_g354_reg16(ser, G354_ADDR_BURST_CTRL2_LO)
        print(f"readback BURST_CTRL1=0x{burst1:04X}, BURST_CTRL2=0x{burst2:04X}")

        if burst1 != G354_BURST_CTRL1_FIXED or burst2 != G354_BURST_CTRL2_FIXED:
            raise RuntimeError(
                f"G354 burst register mismatch: "
                f"BURST_CTRL1=0x{burst1:04X}, BURST_CTRL2=0x{burst2:04X}"
            )
    else:
        print("skip register readback, start sampling directly")

    # window0，开始采样
    write_g354_reg8(ser, G354_ADDR_WIN_CTRL, G354_CMD_WINDOW0)
    write_g354_reg8(ser, G354_ADDR_MODE_CTRL_HI, G354_CMD_BEGIN_SAMPLING)

    time.sleep(0.1)
    ser.reset_input_buffer()
    ser.reset_output_buffer()


def i32_from(data: bytes, start: int) -> int:
    return int.from_bytes(data[start:start + 4], byteorder=BYTEORDER, signed=True)


def read_g354_raw_frame(ser: serial.Serial):
    ser.write(bytes([0x80, 0x00, 0x0D]))
    ser.flush()
    data = read_exact(ser, G354_FRAME_LEN, timeout=0.2)

    if len(data) != G354_FRAME_LEN:
        raise RuntimeError(f"G354 frame length error: got {len(data)}, expected {G354_FRAME_LEN}")

    if data[0] != 0x80 or data[-1] != 0x0D:
        raise RuntimeError(
            f"G354 frame header/tail error: head=0x{data[0]:02X}, tail=0x{data[-1]:02X}"
        )

    gyro_x = i32_from(data, 7) * (SF_GYRO / SCALE_DIV)
    gyro_y = i32_from(data, 11) * (SF_GYRO / SCALE_DIV)
    gyro_z = i32_from(data, 15) * (SF_GYRO / SCALE_DIV)

    acc_x = i32_from(data, 19) * (SF_ACC / SCALE_DIV)
    acc_y = i32_from(data, 23) * (SF_ACC / SCALE_DIV)
    acc_z = i32_from(data, 27) * (SF_ACC / SCALE_DIV)

    gyro = np.array([gyro_x, gyro_y, gyro_z], dtype=float)
    accel = np.array([acc_x, acc_y, acc_z], dtype=float)
    return gyro, accel

class AttitudeEKF:

    ERR_N = 6

    def __init__(self,
                 acc_noise_unit: float = ATT_EKF_ACC_NOISE_UNIT,
                 gyro_noise_dps_sqrt_hz: float = ATT_EKF_GYRO_NOISE_DPS_SQRT_HZ,
                 gyro_bias_rw_dps_sqrt_s: float = ATT_EKF_BIAS_RW_DPS_SQRT_S,
                 acc_norm_min_g: float = ACC_NORM_MIN_G,
                 acc_norm_max_g: float = ACC_NORM_MAX_G):
        self.acc_noise_unit = float(acc_noise_unit)
        self.gyro_noise_rad_sqrt_hz = float(gyro_noise_dps_sqrt_hz) * DEG2RAD
        self.gyro_bias_rw_rad_sqrt_s = float(gyro_bias_rw_dps_sqrt_s) * DEG2RAD
        self.acc_norm_min_g = float(acc_norm_min_g)
        self.acc_norm_max_g = float(acc_norm_max_g)

        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.bg = np.zeros(3, dtype=float)
        self.P = np.zeros((self.ERR_N, self.ERR_N), dtype=float)

        self.last_acc_norm_g = 0.0
        self.last_acc_used = False

        self.reset_covariance()

    def reset_covariance(self) -> None:
        self.P.fill(0.0)
        angle_std = 5.0 * DEG2RAD
        bias_std = 0.20 * DEG2RAD
        self.P[0, 0] = angle_std * angle_std
        self.P[1, 1] = angle_std * angle_std
        self.P[2, 2] = angle_std * angle_std
        self.P[3, 3] = bias_std * bias_std
        self.P[4, 4] = bias_std * bias_std
        self.P[5, 5] = bias_std * bias_std

    @staticmethod
    def normalize_quaternion(q: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(q)
        if n <= 1e-18:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / n

    @staticmethod
    def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array([
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ], dtype=float)

    @classmethod
    def delta_quat_from_rot_vec(cls, rv: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rv))
        if angle < 1e-12:
            dq = np.array([1.0, 0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2]], dtype=float)
            return cls.normalize_quaternion(dq)

        half = 0.5 * angle
        s = math.sin(half) / angle
        dq = np.array([
            math.cos(half),
            rv[0] * s,
            rv[1] * s,
            rv[2] * s,
        ], dtype=float)
        return cls.normalize_quaternion(dq)

    @staticmethod
    def gravity_body_from_quaternion(q: np.ndarray) -> np.ndarray:
        # q=[1,0,0,0] 时 gravity_body=[0,0,+1]
        return np.array([
            2.0 * (q[1] * q[3] - q[0] * q[2]),
            2.0 * (q[0] * q[1] + q[2] * q[3]),
            q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3],
        ], dtype=float)

    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ], dtype=float)

    def init_from_accel(self, accel_mg: np.ndarray, yaw_deg: float = 0.0) -> bool:
        acc_g = np.asarray(accel_mg, dtype=float) / 1000.0
        n = np.linalg.norm(acc_g)
        if n <= 1e-12:
            return False

        gx, gy, gz = -acc_g / n
        roll = math.atan2(gy, gz)
        pitch = math.atan2(-gx, math.sqrt(gy * gy + gz * gz))
        yaw = float(yaw_deg) * DEG2RAD

        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        self.q = np.array([
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        ], dtype=float)
        self.q = self.normalize_quaternion(self.q)
        self.bg.fill(0.0)
        self.reset_covariance()
        return True

    def update_imu(self, gyro_dps: np.ndarray, accel_mg: np.ndarray, dt: float) -> bool:
        if dt <= 0.0:
            return False

        gyro_rad = np.asarray(gyro_dps, dtype=float) * DEG2RAD
        self.predict(gyro_rad, float(dt))

        acc_g = np.asarray(accel_mg, dtype=float) / 1000.0
        acc_norm = float(np.linalg.norm(acc_g))
        self.last_acc_norm_g = acc_norm

        use_acc = self.acc_norm_min_g <= acc_norm <= self.acc_norm_max_g and acc_norm > 1e-12
        self.last_acc_used = bool(use_acc)

        if use_acc:
            measured_gravity_body = -acc_g / acc_norm
            self.update_accel_gravity(measured_gravity_body)

        self.symmetrize_p()
        return True

    def predict(self, gyro_rad: np.ndarray, dt: float) -> None:
        if dt <= 0.0 or dt > 0.2:
            return

        omega = gyro_rad - self.bg
        rv = omega * dt
        dq = self.delta_quat_from_rot_vec(rv)
        self.q = self.quat_multiply(self.q, dq)
        self.q = self.normalize_quaternion(self.q)

        F = np.eye(self.ERR_N, dtype=float)
        W = self.skew(omega)
        F[0:3, 0:3] += -W * dt
        F[0:3, 3:6] = -np.eye(3) * dt

        self.P = F @ self.P @ F.T

        qg = self.gyro_noise_rad_sqrt_hz * self.gyro_noise_rad_sqrt_hz * dt
        qbg = self.gyro_bias_rw_rad_sqrt_s * self.gyro_bias_rw_rad_sqrt_s * dt

        self.P[0, 0] += qg
        self.P[1, 1] += qg
        self.P[2, 2] += qg
        self.P[3, 3] += qbg
        self.P[4, 4] += qbg
        self.P[5, 5] = 1e-14

    def update_accel_gravity(self, measured_gravity_body: np.ndarray) -> None:
        h = self.gravity_body_from_quaternion(self.q)

        H = np.zeros((3, self.ERR_N), dtype=float)
        eps = 1e-5
        for col in range(3):
            rv = np.zeros(3, dtype=float)
            rv[col] = eps
            dq = self.delta_quat_from_rot_vec(rv)
            q_pert = self.quat_multiply(self.q, dq)
            q_pert = self.normalize_quaternion(q_pert)
            hp = self.gravity_body_from_quaternion(q_pert)
            H[:, col] = (hp - h) / eps

        y = measured_gravity_body - h
        R = np.eye(3, dtype=float) * (self.acc_noise_unit * self.acc_noise_unit)
        S = H @ self.P @ H.T + R

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        dx = K @ y
        self.apply_error_state(dx)

        I = np.eye(self.ERR_N, dtype=float)
        self.P = (I - K @ H) @ self.P

    def apply_error_state(self, dx: np.ndarray) -> None:
        rv = dx[0:3]
        dq = self.delta_quat_from_rot_vec(rv)
        self.q = self.quat_multiply(self.q, dq)
        self.q = self.normalize_quaternion(self.q)
        self.bg[0] += dx[3]
        self.bg[1] += dx[4]
        self.bg[2] = 0.0

    def symmetrize_p(self) -> None:
        self.P = 0.5 * (self.P + self.P.T)
        for i in range(self.ERR_N):
            if self.P[i, i] < 1e-14:
                self.P[i, i] = 1e-14

    def get_quaternion(self) -> np.ndarray:
        return self.q.copy()

    def get_euler(self) -> np.ndarray:
        q0, q1, q2, q3 = self.q

        roll = math.atan2(
            2.0 * (q0 * q1 + q2 * q3),
            1.0 - 2.0 * (q1 * q1 + q2 * q2)
        )

        sin_pitch = 2.0 * (q0 * q2 - q3 * q1)
        sin_pitch = max(-1.0, min(1.0, sin_pitch))
        pitch = math.asin(sin_pitch)

        yaw = math.atan2(
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q2 * q2 + q3 * q3)
        )

        return np.array([roll * RAD2DEG, pitch * RAD2DEG, yaw * RAD2DEG], dtype=float)

    def get_gyro_bias_dps(self) -> np.ndarray:
        return self.bg * RAD2DEG

def calibrate_gyro_bias(ser: serial.Serial, samples: int = GYRO_CALIB_SAMPLES) -> np.ndarray:
    print(f"开始陀螺仪零偏校准：{samples} samples。请保持 IMU 完全静止。你不要动他口牙")

    gyro_list = []
    valid = 0

    for i in range(samples):
        try:
            gyro, accel = read_g354_raw_frame(ser)
        except RuntimeError as e:
            print(f"  calib bad frame: {e}")
            continue

        gyro_list.append(gyro)
        valid += 1

        if (i + 1) % 100 == 0:
            acc_norm = np.linalg.norm(accel) / 1000.0
            print(f"  calib {i + 1}/{samples}, gyro={gyro}, acc_norm={acc_norm:.3f}g")

        time.sleep(0.002)

    if valid <= 0:
        raise RuntimeError("gyro calib failed: no valid frame")

    gyro_arr = np.asarray(gyro_list, dtype=float)
    bias = gyro_arr.mean(axis=0)
    std = gyro_arr.std(axis=0)
    p2p = gyro_arr.max(axis=0) - gyro_arr.min(axis=0)

    print(
        "陀螺仪零偏校准完成，做完了才说道歉...\n"
        f"  valid = {valid}/{samples}\n"
        f"  bias[dps] = gx:{bias[0]:.6f}, gy:{bias[1]:.6f}, gz:{bias[2]:.6f}\n"
        f"  std [dps] = gx:{std[0]:.6f}, gy:{std[1]:.6f}, gz:{std[2]:.6f}\n"
        f"  p2p [dps] = gx:{p2p[0]:.6f}, gy:{p2p[1]:.6f}, gz:{p2p[2]:.6f}"
    )
    return bias


def average_accel_for_init(ser: serial.Serial, samples: int = ACC_INIT_SAMPLES) -> np.ndarray:
    print(f"开始加速度姿态初始化：{samples} samples。请保持 IMU 静止。你不要动他口牙")

    acc_sum = np.zeros(3, dtype=float)
    valid = 0

    for _ in range(samples):
        try:
            _, accel = read_g354_raw_frame(ser)
        except RuntimeError as e:
            print(f"  init bad frame: {e}")
            continue

        acc_sum += accel
        valid += 1
        time.sleep(0.005)

    if valid <= 0:
        raise RuntimeError("acc init failed: no valid frame")

    acc_mean = acc_sum / valid
    print(f"acc_init[mg] = {acc_mean}, norm={np.linalg.norm(acc_mean) / 1000.0:.4f}g")
    return acc_mean

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G354 6-axis Attitude EKF test")
    parser.add_argument("port", nargs="?", default=PORT, help="serial port, default /dev/ttyUSB0")
    parser.add_argument("baudrate", nargs="?", type=int, default=BAUDRATE, help="baudrate, default 460800")
    parser.add_argument("--verify-burst", action="store_true", help="read back BURST_CTRL registers after config")
    parser.add_argument("--calib-samples", type=int, default=GYRO_CALIB_SAMPLES, help="gyro calibration samples")
    parser.add_argument("--warmup-samples", type=int, default=WARMUP_SAMPLES, help="attitude EKF warm-up samples")
    parser.add_argument("--acc-init-samples", type=int, default=ACC_INIT_SAMPLES, help="accel init samples")
    parser.add_argument("--no-gz-calib", action="store_true", help="do not subtract z gyro bias")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrate_gz = not args.no_gz_calib

    print(f"Open serial: {args.port} baudrate={args.baudrate}")
    ser = open_serial(args.port, args.baudrate)

    try:
        time.sleep(0.8)
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        init_g354_sampling_mode(ser, verify=args.verify_burst)

        ahrs = AttitudeEKF(
            acc_noise_unit=ATT_EKF_ACC_NOISE_UNIT,
            gyro_noise_dps_sqrt_hz=ATT_EKF_GYRO_NOISE_DPS_SQRT_HZ,
            gyro_bias_rw_dps_sqrt_s=ATT_EKF_BIAS_RW_DPS_SQRT_S,
            acc_norm_min_g=ACC_NORM_MIN_G,
            acc_norm_max_g=ACC_NORM_MAX_G,
        )

        gyro_bias = calibrate_gyro_bias(ser, samples=args.calib_samples)
        print(f"Z gyro bias correction: {'enabled' if calibrate_gz else 'disabled, gz free integration'}")

        acc_init = average_accel_for_init(ser, samples=args.acc_init_samples)
        if not ahrs.init_from_accel(acc_init, yaw_deg=0.0):
            raise RuntimeError("AttitudeEKF init_from_accel failed")

        roll, pitch, yaw = ahrs.get_euler()
        print(f"初始姿态 roll:{roll:.2f} pitch:{pitch:.2f} yaw:{yaw:.2f}")

        print(f"开始 Attitude EKF warm-up：{args.warmup_samples} samples，只更新姿态。你竟有这等实力")
        last_t = time.perf_counter()
        warm_bad = 0

        for _ in range(args.warmup_samples):
            loop_start = time.perf_counter()
            try:
                gyro_raw, accel = read_g354_raw_frame(ser)
            except RuntimeError:
                warm_bad += 1
                continue

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            dt_used = TARGET_DT if dt < DT_MIN or dt > DT_MAX else dt

            gyro = gyro_raw - gyro_bias
            if not calibrate_gz:
                gyro[2] = gyro_raw[2]

            ahrs.update_imu(gyro, accel, dt_used)

            elapsed = time.perf_counter() - loop_start
            sleep_time = TARGET_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        roll, pitch, yaw = ahrs.get_euler()
        print(f"warm-up 后姿态 roll:{roll:.2f} pitch:{pitch:.2f} yaw:{yaw:.2f} warm_bad:{warm_bad}")
        print("进入 Attitude EKF 姿态解算循环。Ctrl+C 结束。蹦蹦炸弹")

        last_t = time.perf_counter()
        frame_count = 0
        bad_frame_count = 0

        while True:
            loop_start = time.perf_counter()

            try:
                gyro_raw, accel = read_g354_raw_frame(ser)
            except RuntimeError as e:
                bad_frame_count += 1
                print(f"bad frame: {e}")
                continue

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            dt_used = TARGET_DT if dt < DT_MIN or dt > DT_MAX else dt

            gyro = gyro_raw - gyro_bias
            if not calibrate_gz:
                gyro[2] = gyro_raw[2]

            ahrs.update_imu(gyro, accel, dt_used)
            roll, pitch, yaw = ahrs.get_euler()
            att_bg = ahrs.get_gyro_bias_dps()
            acc_norm_g = np.linalg.norm(accel) / 1000.0

            frame_count += 1

            print(
                "roll:%8.2f pitch:%8.2f yaw:%8.2f | "
                "gyro[dps]=%8.3f,%8.3f,%8.3f | "
                "att_bg[dps]=%8.4f,%8.4f,%8.4f | "
                "acc[mG]=%8.1f,%8.1f,%8.1f acc_norm:%5.3fg acc_used:%d | "
                "dt:%7.4f bad:%d" % (
                    roll, pitch, yaw,
                    gyro[0], gyro[1], gyro[2],
                    att_bg[0], att_bg[1], att_bg[2],
                    accel[0], accel[1], accel[2], acc_norm_g,
                    1 if ahrs.last_acc_used else 0,
                    dt_used,
                    bad_frame_count,
                )
            )

            elapsed = time.perf_counter() - loop_start
            sleep_time = TARGET_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        ser.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n你要跟可莉玩吗？嗒嗒嗒嗒嗒，好想玩原神（云原神！）")
