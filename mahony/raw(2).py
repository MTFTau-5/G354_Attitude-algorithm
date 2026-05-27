import serial
import numpy as np
import math
import time

"""配置参数写在前，
   养成代码好习惯"""
    
PORT = "/dev/cu.usbmodem59090479771"
BAUDRATE = 230400
BYTEORDER = "big"
SF_GYRO = 0.016  # (deg/s)/LSB
SF_ACC = 0.2     # mG/LSB
SCALE_DIV = 65536
TARGET_HZ = 100.0
TARGET_DT = 1.0 / TARGET_HZ
MAHONY_KP = 1.0
MAHONY_KI = 0.005
ACC_NORM_MIN_G = 0.85
ACC_NORM_MAX_G = 1.15
GYRO_CALIB_SAMPLES = 500
DT_MIN = 0.001
DT_MAX = 0.05


class MahonyAHRS:
    """
    6轴 Mahony AHRS：gyro + accel。

    注意：6轴只能用重力约束 roll/pitch，无法从根本上约束 yaw。
    yaw 长时间漂移属于物理观测不足，不是单纯调参能彻底解决的问题。leader老师我做不到啊。
        - 不过你可以通过增加陀螺积分的 KI 来减缓漂移，但过大可能导致响应变慢，甚至不稳定。
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
            accel_norm = accel_g / acc_norm

            q0, q1, q2, q3 = self.q

            # 当前姿态预测出的重力方向，表示在机体系下
            vx = 2.0 * (q1 * q3 - q0 * q2)
            vy = 2.0 * (q0 * q1 + q2 * q3)
            vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3
            gravity_body = np.array([vx, vy, vz], dtype=float)
            error = np.cross(accel_norm, gravity_body)

            if self.ki > 0.0:
                self.integral_error += error * self.ki * dt
            else:
                self.integral_error[:] = 0.0
        else:
            pass

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
    ser.write(bytes([0x80, 0x00, 0x0d]))
    data = ser.read(38)

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
    print(f"开始陀螺仪零偏校准：{samples} samples。请保持 IMU 完全静止，不要动他口牙...")

    bias_sum = np.zeros(3, dtype=float)
    valid = 0

    for i in range(samples):
        gyro, accel = read_g354_raw_frame(ser)
        bias_sum += gyro
        valid += 1

        if (i + 1) % 100 == 0:
            acc_norm = np.linalg.norm(accel) / 1000.0
            print(f"  calib {i + 1}/{samples}, gyro={gyro}, acc_norm={acc_norm:.3f}g")

        time.sleep(0.002)

    bias = bias_sum / max(valid, 1)
    print(f"陀螺仪零偏校准完成 bias[dps] = gx:{bias[0]:.6f}, gy:{bias[1]:.6f}, gz:{bias[2]:.6f}")
    print("如果 gz 静止零偏很大，yaw 漂移会非常明显，这个随便喵。")
    return bias


def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=0.2)
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # change to conf mode / disable uart auto mode / change to sample mode
    ser.write(bytes([0xfe, 0x00, 0x0d, 0x83, 0x01, 0x0d]))
    time.sleep(0.1)
    ser.reset_input_buffer()

    mahony = MahonyAHRS(
        kp=MAHONY_KP,
        ki=MAHONY_KI,
        acc_norm_min_g=ACC_NORM_MIN_G,
        acc_norm_max_g=ACC_NORM_MAX_G,
    )

    gyro_bias = calibrate_gyro_bias(ser, samples=GYRO_CALIB_SAMPLES)

    print("滴滴滴进入姿态解算循环楼。")

    last_t = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        gyro_raw, accel = read_g354_raw_frame(ser)
        now = time.perf_counter()
        dt = now - last_t
        last_t = now

        # dt 异常保护
        if dt < DT_MIN or dt > DT_MAX:
            dt_used = TARGET_DT
        else:
            dt_used = dt

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
