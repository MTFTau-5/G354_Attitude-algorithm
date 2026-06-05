import serial
import numpy as np
import math
import time

"""
G354 + 6轴 Mahony 姿态解算测试版。

坐标约定：
    x：前方
    y：右手边
    z：向下，和真实重力方向相同

重要约定：
    加速度计原始输出是 specific force，比力。
    静止时 raw_acc ≈ -gravity_body。
    所以 Mahony 用于姿态修正的重力方向必须使用：gravity_meas = -raw_acc / |raw_acc|。
"""

PORT = "/dev/ttyUSB0"
BAUDRATE = 460800
BYTEORDER = "big"
SF_GYRO = 0.016  # (deg/s)/LSB, 16bit scale
SF_ACC = 0.2     # mG/LSB, 16bit scale
SCALE_DIV = 65536

TARGET_HZ = 100.0
TARGET_DT = 1.0 / TARGET_HZ

MAHONY_KP = 1.0
MAHONY_KI = 0.005

ACC_NORM_MIN_G = 0.85
ACC_NORM_MAX_G = 1.15
GYRO_CALIB_SAMPLES = 500
ACC_INIT_SAMPLES = 200
WARMUP_SAMPLES = 300

DT_MIN = 0.001
DT_MAX = 0.05
G354_FRAME_LEN = 38


class MahonyAHRS:
    """
    6轴 Mahony AHRS：gyro + accel。

    注意：
    1. 6轴只能用重力约束 roll/pitch，无法从根本上约束 yaw。
    2. 本代码把四元数 q=[1,0,0,0] 定义为：真实重力方向在机体系下为 +Z。
    3. G354 原始加速度静止时约为 -Z，因此 update_imu 内部会取反。
    """

    def __init__(self, kp=1.0, ki=0.0,
                 acc_norm_min_g=0.85, acc_norm_max_g=1.15):
        self.kp = float(kp)
        self.ki = float(ki)
        self.acc_norm_min_g = float(acc_norm_min_g)
        self.acc_norm_max_g = float(acc_norm_max_g)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.integral_error = np.zeros(3, dtype=float)
        self.last_acc_norm_g = 0.0
        self.last_acc_used = False

    @staticmethod
    def _normalize_quaternion(q):
        n = np.linalg.norm(q)
        if n <= 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / n

    def init_from_accel(self, accel_mg, yaw_deg=0.0):
        """
        用静止状态下的加速度初始化 roll/pitch。

        accel_mg 是 G354 原始加速度，比力 specific force。
        静止时 raw_acc ≈ -gravity_body，因此这里先取反得到真实重力方向。
        yaw 无法由 6轴 IMU 观测，这里只给一个初始值。
        """
        acc = np.asarray(accel_mg, dtype=float) / 1000.0
        n = np.linalg.norm(acc)
        if n <= 1e-12:
            return False

        gx, gy, gz = -acc / n  # 真实重力方向，机体系，z 向下为正

        roll = math.atan2(gy, gz)
        pitch = math.atan2(-gx, math.sqrt(gy * gy + gz * gz))
        yaw = math.radians(yaw_deg)

        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        q0 = cy * cp * cr + sy * sp * sr
        q1 = cy * cp * sr - sy * sp * cr
        q2 = sy * cp * sr + cy * sp * cr
        q3 = sy * cp * cr - cy * sp * sr

        self.q = self._normalize_quaternion(np.array([q0, q1, q2, q3], dtype=float))
        self.integral_error[:] = 0.0
        return True

    def update_imu(self, gyro_dps, accel_mg, dt):
        if dt <= 0:
            return False

        gyro_rad = np.deg2rad(np.asarray(gyro_dps, dtype=float))
        accel_g = np.asarray(accel_mg, dtype=float) / 1000.0

        acc_norm = np.linalg.norm(accel_g)
        self.last_acc_norm_g = float(acc_norm)

        use_acc = self.acc_norm_min_g <= acc_norm <= self.acc_norm_max_g
        self.last_acc_used = bool(use_acc)

        error = np.zeros(3, dtype=float)

        if use_acc and acc_norm > 1e-12:
            # G354 原始加速度是 specific force，静止时与真实重力方向相反。
            # Mahony 这里需要的是真实重力方向，所以必须取反。
            accel_norm = -accel_g / acc_norm

            q0, q1, q2, q3 = self.q

            # 当前姿态预测出的真实重力方向，表示在机体系下。
            vx = 2.0 * (q1 * q3 - q0 * q2)
            vy = 2.0 * (q0 * q1 + q2 * q3)
            vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
            gravity_body = np.array([vx, vy, vz], dtype=float)

            error = np.cross(accel_norm, gravity_body)

            if self.ki > 0.0:
                self.integral_error += error * self.ki * dt
            else:
                self.integral_error[:] = 0.0

        gyro_corrected = gyro_rad + self.kp * error + self.integral_error

        q0, q1, q2, q3 = self.q
        wx, wy, wz = gyro_corrected

        qdot = np.array([
            -0.5 * (q1 * wx + q2 * wy + q3 * wz),
             0.5 * (q0 * wx + q2 * wz - q3 * wy),
             0.5 * (q0 * wy - q1 * wz + q3 * wx),
             0.5 * (q0 * wz + q1 * wy - q2 * wx)
        ], dtype=float)

        self.q = self._normalize_quaternion(self.q + qdot * dt)
        return True

    def get_euler(self):
        """返回 roll, pitch, yaw，单位 degree。"""
        q0, q1, q2, q3 = self.q

        roll = math.atan2(
            2.0 * (q0 * q1 + q2 * q3),
            1.0 - 2.0 * (q1 * q1 + q2 * q2)
        )

        pitch_arg = 2.0 * (q0 * q2 - q3 * q1)
        pitch_arg = max(-1.0, min(1.0, pitch_arg))
        pitch = math.asin(pitch_arg)

        yaw = math.atan2(
            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q2 * q2 + q3 * q3)
        )

        return np.rad2deg([roll, pitch, yaw])

    def get_quaternion(self):
        return self.q.copy()


def i32_from(data, start):
    """从 data[start:start+4] 解析 int32。"""
    return int.from_bytes(data[start:start + 4], byteorder=BYTEORDER, signed=True)


def read_g354_raw_frame(ser):
    ser.write(bytes([0x80, 0x00, 0x0D]))
    data = ser.read(G354_FRAME_LEN)

    if len(data) != G354_FRAME_LEN:
        raise RuntimeError(f"G354 frame length error: got {len(data)}, expected {G354_FRAME_LEN}")

    if data[0] != 0x80 or data[-1] != 0x0D:
        raise RuntimeError(f"G354 frame header/tail error: head=0x{data[0]:02X}, tail=0x{data[-1]:02X}")

    gyro_x = i32_from(data, 7)  * (SF_GYRO / SCALE_DIV)
    gyro_y = i32_from(data, 11) * (SF_GYRO / SCALE_DIV)
    gyro_z = i32_from(data, 15) * (SF_GYRO / SCALE_DIV)

    acc_x = i32_from(data, 19) * (SF_ACC / SCALE_DIV)
    acc_y = i32_from(data, 23) * (SF_ACC / SCALE_DIV)
    acc_z = i32_from(data, 27) * (SF_ACC / SCALE_DIV)

    gyro = np.array([gyro_x, gyro_y, gyro_z], dtype=float)
    accel = np.array([acc_x, acc_y, acc_z], dtype=float)
    return gyro, accel


def calibrate_gyro_bias(ser, samples=500):
    print(f"开始陀螺仪零偏校准：{samples} samples。请保持 IMU 完全静止。")

    bias_sum = np.zeros(3, dtype=float)
    valid = 0

    for i in range(samples):
        try:
            gyro, accel = read_g354_raw_frame(ser)
        except RuntimeError as e:
            print(f"  calib bad frame: {e}")
            continue

        bias_sum += gyro
        valid += 1

        if (i + 1) % 100 == 0:
            acc_norm = np.linalg.norm(accel) / 1000.0
            print(f"  calib {i + 1}/{samples}, gyro={gyro}, acc_norm={acc_norm:.3f}g")

        time.sleep(0.002)

    bias = bias_sum / max(valid, 1)
    print(f"陀螺仪零偏校准完成 bias[dps] = gx:{bias[0]:.6f}, gy:{bias[1]:.6f}, gz:{bias[2]:.6f}")
    return bias


def average_accel_for_init(ser, samples=200):
    print(f"开始加速度姿态初始化：{samples} samples。请保持 IMU 静止。")

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


def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.2)
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # 兼容你原来的简化配置命令。
    # C++ 版 main_fixed.cpp 里有更完整的寄存器配置与回读校验。
    ser.write(bytes([0xFE, 0x00, 0x0D, 0x83, 0x01, 0x0D]))
    time.sleep(0.1)
    ser.reset_input_buffer()

    mahony = MahonyAHRS(
        kp=MAHONY_KP,
        ki=MAHONY_KI,
        acc_norm_min_g=ACC_NORM_MIN_G,
        acc_norm_max_g=ACC_NORM_MAX_G,
    )

    gyro_bias = calibrate_gyro_bias(ser, samples=GYRO_CALIB_SAMPLES)

    acc_init = average_accel_for_init(ser, samples=ACC_INIT_SAMPLES)
    mahony.init_from_accel(acc_init, yaw_deg=0.0)
    roll, pitch, yaw = mahony.get_euler()
    print(f"初始姿态 roll:{roll:.2f} pitch:{pitch:.2f} yaw:{yaw:.2f}")

    print(f"开始 AHRS warm-up：{WARMUP_SAMPLES} samples，只更新姿态，不做路径积分。")
    last_t = time.perf_counter()
    for _ in range(WARMUP_SAMPLES):
        try:
            gyro_raw, accel = read_g354_raw_frame(ser)
        except RuntimeError:
            continue

        now = time.perf_counter()
        dt = now - last_t
        last_t = now
        dt_used = TARGET_DT if dt < DT_MIN or dt > DT_MAX else dt
        gyro = gyro_raw - gyro_bias
        mahony.update_imu(gyro, accel, dt_used)
        time.sleep(max(0.0, TARGET_DT - (time.perf_counter() - now)))

    roll, pitch, yaw = mahony.get_euler()
    print(f"warm-up 后姿态 roll:{roll:.2f} pitch:{pitch:.2f} yaw:{yaw:.2f}")
    print("进入姿态解算循环。")

    last_t = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        try:
            gyro_raw, accel = read_g354_raw_frame(ser)
        except RuntimeError as e:
            print(f"bad frame: {e}")
            continue

        now = time.perf_counter()
        dt = now - last_t
        last_t = now

        dt_used = TARGET_DT if dt < DT_MIN or dt > DT_MAX else dt

        gyro = gyro_raw - gyro_bias
        mahony.update_imu(gyro, accel, dt_used)

        roll, pitch, yaw = mahony.get_euler()
        acc_norm_g = np.linalg.norm(accel) / 1000.0

        print(
            "roll:%8.2f pitch:%8.2f yaw:%8.2f | "
            "gyro[dps]=%8.3f,%8.3f,%8.3f | "
            "acc[mG]=%8.1f,%8.1f,%8.1f acc_norm:%5.3fg acc_used:%d | "
            "dt:%7.4f" % (
                roll, pitch, yaw,
                gyro[0], gyro[1], gyro[2],
                accel[0], accel[1], accel[2], acc_norm_g,
                1 if mahony.last_acc_used else 0,
                dt_used
            )
        )

        elapsed = time.perf_counter() - loop_start
        sleep_time = TARGET_DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户终止。")
