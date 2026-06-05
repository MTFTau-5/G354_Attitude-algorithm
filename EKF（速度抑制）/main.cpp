#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <termios.h>
#include <unistd.h>
#include <sys/select.h>

#include "mahony.h"

static std::atomic<bool> g_running(true);

static constexpr int G354_FRAME_LEN = 38;

// Epson G354 UART register map pieces used here.
// UART write command format: [0x80 | addr, value, 0x0D]
// UART read command format : [addr, 0x00, 0x0D] -> [addr, high, low, 0x0D]
static constexpr uint8_t G354_ADDR_WIN_CTRL        = 0x7E;
static constexpr uint8_t G354_ADDR_MODE_CTRL_HI    = 0x03;
static constexpr uint8_t G354_ADDR_UART_CTRL_LO    = 0x08;
static constexpr uint8_t G354_ADDR_BURST_CTRL1_LO  = 0x0C;
static constexpr uint8_t G354_ADDR_BURST_CTRL2_LO  = 0x0E;

static constexpr uint8_t G354_CMD_WINDOW0          = 0x00;
static constexpr uint8_t G354_CMD_WINDOW1          = 0x01;
static constexpr uint8_t G354_CMD_BEGIN_SAMPLING   = 0x01;
static constexpr uint8_t G354_CMD_END_SAMPLING     = 0x02;

// Fixed burst format for the parser below:
// BURST_CTRL1 = 0xF007: FLAG + TEMP + GYRO + ACCEL + GPIO + COUNT + CHECKSUM
// BURST_CTRL2 = 0x7000: TEMP/GYRO/ACCEL all output in 32-bit format
static constexpr uint16_t G354_BURST_CTRL1_FIXED   = 0xF007;
static constexpr uint16_t G354_BURST_CTRL2_FIXED   = 0x7000;

static constexpr float SF_GYRO = 0.016f;       // (deg/s)/LSB, 16bit scale
static constexpr float SF_ACC  = 0.2f;         // mG/LSB, 16bit scale
static constexpr float SCALE_DIV = 65536.0f;   // 32bit high/low combined correction

static constexpr float G0 = 9.7970f;  // 太原本地重力加速度

// Z 轴陀螺参与零偏扣除，降低 yaw/Z 漂移。
// 注意：6 轴 IMU 没有绝对航向，Z 轴校准只能去掉静态零偏，不能长期锁定 yaw。
static constexpr bool CALIBRATE_GYRO_Z = true;

// 转圈/改变朝向运动时，必须用 yaw 把机体系 XY 加速度旋转到初始世界 XY。
// false: 输出机体系/初始局部轴下的 XY；true: 用相对 yaw 旋转到世界 XY。
static constexpr bool USE_YAW_FOR_XY_ROTATION = true;

static constexpr float MAHONY_KP = 1.0f;
static constexpr float MAHONY_KI = 0.005f;

static constexpr int GYRO_CALIB_SAMPLES = 500;
static constexpr int ACC_INIT_SAMPLES = 200;
static constexpr int AHRS_WARMUP_SAMPLES = 300;

static constexpr float TARGET_HZ = 100.0f;
static constexpr float TARGET_DT = 1.0f / TARGET_HZ;

static constexpr float DT_MIN = 0.001f;
static constexpr float DT_MAX = 0.05f;

static constexpr float STATIONARY_GYRO_DPS = 1.2f;
static constexpr float STATIONARY_ACC_NORM_ERR_MG = 35.0f;
static constexpr float STATIONARY_LIN_ACC_XY_MG = 60.0f;
static constexpr int STATIONARY_CONFIRM_COUNT = 5;

struct G354Data {
    float gx = 0.0f;  // deg/s
    float gy = 0.0f;
    float gz = 0.0f;

    float ax = 0.0f;  // mg, raw specific force
    float ay = 0.0f;
    float az = 0.0f;
};

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

static void onSignal(int) {
    g_running = false;
}

static speed_t baudToConstant(int baud) {
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
#ifdef B230400
        case 230400: return B230400;
#endif
#ifdef B460800
        case 460800: return B460800;
#endif
#ifdef B921600
        case 921600: return B921600;
#endif
        default:
            throw std::runtime_error("Unsupported baudrate on this system.");
    }
}

class SerialPort {
public:
    SerialPort(const std::string& dev, int baudrate) {
        fd_ = open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open serial port: " + dev);
        }

        configure(baudrate);
    }

    ~SerialPort() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    bool writeAll(const uint8_t* data, size_t len) {
        size_t sent = 0;

        while (sent < len) {
            ssize_t n = write(fd_, data + sent, len - sent);
            if (n > 0) {
                sent += static_cast<size_t>(n);
            } else {
                return false;
            }
        }

        return true;
    }

    bool readExact(uint8_t* data, size_t len, int timeoutMs) {
        size_t got = 0;
        auto start = std::chrono::steady_clock::now();

        while (got < len) {
            auto now = std::chrono::steady_clock::now();
            int elapsedMs = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count()
            );

            int remainMs = timeoutMs - elapsedMs;
            if (remainMs <= 0) {
                return false;
            }

            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(fd_, &readfds);

            struct timeval tv {};
            tv.tv_sec = remainMs / 1000;
            tv.tv_usec = (remainMs % 1000) * 1000;

            int ret = select(fd_ + 1, &readfds, nullptr, nullptr, &tv);
            if (ret > 0 && FD_ISSET(fd_, &readfds)) {
                ssize_t n = read(fd_, data + got, len - got);
                if (n > 0) {
                    got += static_cast<size_t>(n);
                }
            } else {
                return false;
            }
        }

        return true;
    }

    void flushIO() {
        tcflush(fd_, TCIOFLUSH);
    }

private:
    int fd_ = -1;

    void configure(int baudrate) {
        struct termios tty {};
        if (tcgetattr(fd_, &tty) != 0) {
            throw std::runtime_error("tcgetattr failed.");
        }

        cfmakeraw(&tty);

        speed_t baud = baudToConstant(baudrate);
        cfsetispeed(&tty, baud);
        cfsetospeed(&tty, baud);

        tty.c_cflag |= CLOCAL | CREAD;
        tty.c_cflag &= ~PARENB;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;
        tty.c_cflag &= ~CRTSCTS;

        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 2;

        if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
            throw std::runtime_error("tcsetattr failed.");
        }

        tcflush(fd_, TCIOFLUSH);
    }
};

static int32_t i32FromBE(const uint8_t* data, int start) {
    uint32_t u =
        (static_cast<uint32_t>(data[start])     << 24) |
        (static_cast<uint32_t>(data[start + 1]) << 16) |
        (static_cast<uint32_t>(data[start + 2]) << 8)  |
        (static_cast<uint32_t>(data[start + 3]));

    return static_cast<int32_t>(u);
}

static G354Data parseG354Frame(const uint8_t frame[G354_FRAME_LEN]) {
    G354Data imu {};

    int32_t rawGx = i32FromBE(frame, 7);
    int32_t rawGy = i32FromBE(frame, 11);
    int32_t rawGz = i32FromBE(frame, 15);

    int32_t rawAx = i32FromBE(frame, 19);
    int32_t rawAy = i32FromBE(frame, 23);
    int32_t rawAz = i32FromBE(frame, 27);

    imu.gx = rawGx * (SF_GYRO / SCALE_DIV);
    imu.gy = rawGy * (SF_GYRO / SCALE_DIV);
    imu.gz = rawGz * (SF_GYRO / SCALE_DIV);

    imu.ax = rawAx * (SF_ACC / SCALE_DIV);
    imu.ay = rawAy * (SF_ACC / SCALE_DIV);
    imu.az = rawAz * (SF_ACC / SCALE_DIV);

    return imu;
}

static bool readG354RawFrame(SerialPort& ser, uint8_t frame[G354_FRAME_LEN]) {
    const uint8_t burstCmd[3] = {0x80, 0x00, 0x0D};

    if (!ser.writeAll(burstCmd, sizeof(burstCmd))) {
        return false;
    }

    if (!ser.readExact(frame, G354_FRAME_LEN, 200)) {
        return false;
    }

    if (frame[0] != 0x80 || frame[G354_FRAME_LEN - 1] != 0x0D) {
        return false;
    }

    return true;
}

static bool writeG354Reg8(SerialPort& ser, uint8_t addr, uint8_t value) {
    const uint8_t cmd[3] = {
        static_cast<uint8_t>(0x80u | (addr & 0x7Fu)),
        value,
        0x0D
    };

    if (!ser.writeAll(cmd, sizeof(cmd))) {
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return true;
}

static bool readG354Reg16(SerialPort& ser, uint8_t addrEven, uint16_t& value) {
    const uint8_t cmd[3] = {
        static_cast<uint8_t>(addrEven & 0x7Eu),
        0x00,
        0x0D
    };
    uint8_t resp[4] = {};

    ser.flushIO();

    if (!ser.writeAll(cmd, sizeof(cmd))) {
        return false;
    }

    if (!ser.readExact(resp, sizeof(resp), 200)) {
        return false;
    }

    if (resp[0] != (addrEven & 0x7Eu) || resp[3] != 0x0D) {
        return false;
    }

    value = static_cast<uint16_t>((static_cast<uint16_t>(resp[1]) << 8) |
                                  static_cast<uint16_t>(resp[2]));
    return true;
}

static bool setG354Window(SerialPort& ser, uint8_t window) {
    return writeG354Reg8(ser, G354_ADDR_WIN_CTRL, window);
}

static bool writeG354Reg16Parts(SerialPort& ser, uint8_t addrLo, uint16_t value) {
    // Register address xxx_LO stores D[7:0], xxx_HI stores D[15:8].
    return writeG354Reg8(ser, addrLo, static_cast<uint8_t>(value & 0xFFu)) &&
           writeG354Reg8(ser, static_cast<uint8_t>(addrLo + 1u), static_cast<uint8_t>((value >> 8) & 0xFFu));
}

static bool initG354SamplingMode(SerialPort& ser) {
    std::cout << "Configuring G354 fixed UART burst format: "
              << "BURST_CTRL1=0xF007, BURST_CTRL2=0x7000, frame_len=38"
              << std::endl;

    if (!setG354Window(ser, G354_CMD_WINDOW0) ||
        !writeG354Reg8(ser, G354_ADDR_MODE_CTRL_HI, G354_CMD_END_SAMPLING)) {
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ser.flushIO();

    if (!setG354Window(ser, G354_CMD_WINDOW1)) {
        return false;
    }

    if (!writeG354Reg8(ser, G354_ADDR_UART_CTRL_LO, 0x00)) {
        return false;
    }

    if (!writeG354Reg16Parts(ser, G354_ADDR_BURST_CTRL1_LO, G354_BURST_CTRL1_FIXED) ||
        !writeG354Reg16Parts(ser, G354_ADDR_BURST_CTRL2_LO, G354_BURST_CTRL2_FIXED)) {
        return false;
    }

    uint16_t burstCtrl1 = 0;
    uint16_t burstCtrl2 = 0;
    if (!readG354Reg16(ser, G354_ADDR_BURST_CTRL1_LO, burstCtrl1) ||
        !readG354Reg16(ser, G354_ADDR_BURST_CTRL2_LO, burstCtrl2)) {
        std::cerr << "G354 burst register readback failed." << std::endl;
        return false;
    }

    if (burstCtrl1 != G354_BURST_CTRL1_FIXED || burstCtrl2 != G354_BURST_CTRL2_FIXED) {
        std::cerr << "G354 burst register mismatch: BURST_CTRL1=0x"
                  << std::hex << burstCtrl1
                  << " BURST_CTRL2=0x" << burstCtrl2
                  << std::dec << std::endl;
        return false;
    }

    if (!setG354Window(ser, G354_CMD_WINDOW0) ||
        !writeG354Reg8(ser, G354_ADDR_MODE_CTRL_HI, G354_CMD_BEGIN_SAMPLING)) {
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ser.flushIO();

    return true;
}

static float vec3Norm(float x, float y, float z) {
    return std::sqrt(x * x + y * y + z * z);
}

static G354Data calibrateGyroBias(SerialPort& ser, int samples) {
    std::cout << "开始陀螺仪零偏校准，保持 IMU 完全静止。samples="
              << samples << std::endl;

    double sumGx = 0.0;
    double sumGy = 0.0;
    double sumGz = 0.0;

    int valid = 0;

    for (int i = 0; i < samples; ++i) {
        uint8_t frame[G354_FRAME_LEN];

        if (!readG354RawFrame(ser, frame)) {
            continue;
        }

        G354Data imu = parseG354Frame(frame);

        sumGx += imu.gx;
        sumGy += imu.gy;
        sumGz += imu.gz;
        valid++;

        if ((i + 1) % 100 == 0) {
            float accNormG = vec3Norm(imu.ax, imu.ay, imu.az) / 1000.0f;

            std::cout << "calib " << (i + 1) << "/" << samples
                      << " gyro[dps]="
                      << imu.gx << ", " << imu.gy << ", " << imu.gz
                      << " acc_norm=" << accNormG << "g"
                      << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    G354Data bias {};
    if (valid > 0) {
        bias.gx = static_cast<float>(sumGx / valid);
        bias.gy = static_cast<float>(sumGy / valid);
        bias.gz = static_cast<float>(sumGz / valid);
    }

    std::cout << "陀螺仪零偏完成 bias[dps] = "
              << bias.gx << ", " << bias.gy << ", " << bias.gz
              << ", valid=" << valid
              << std::endl;

    return bias;
}

static bool averageAccelForInit(SerialPort& ser, int samples, G354Data& accMean) {
    std::cout << "开始加速度姿态初始化，保持 IMU 静止。samples="
              << samples << std::endl;

    double sumAx = 0.0;
    double sumAy = 0.0;
    double sumAz = 0.0;
    int valid = 0;

    for (int i = 0; i < samples; ++i) {
        uint8_t frame[G354_FRAME_LEN];
        if (!readG354RawFrame(ser, frame)) {
            continue;
        }

        G354Data imu = parseG354Frame(frame);
        sumAx += imu.ax;
        sumAy += imu.ay;
        sumAz += imu.az;
        valid++;

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    if (valid <= 0) {
        return false;
    }

    accMean.ax = static_cast<float>(sumAx / valid);
    accMean.ay = static_cast<float>(sumAy / valid);
    accMean.az = static_cast<float>(sumAz / valid);

    std::cout << "acc_init[mg]="
              << accMean.ax << "," << accMean.ay << "," << accMean.az
              << " norm=" << vec3Norm(accMean.ax, accMean.ay, accMean.az) / 1000.0f << "g"
              << ", valid=" << valid
              << std::endl;

    return true;
}

static void gravityBodyFromQuaternion(const float q[4], float& gxBody, float& gyBody, float& gzBody) {
    gxBody = 2.0f * (q[1] * q[3] - q[0] * q[2]);
    gyBody = 2.0f * (q[0] * q[1] + q[2] * q[3]);
    gzBody = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
}

class PlanarEKF {
public:
    void reset() {
        for (int i = 0; i < STATE_N; ++i) {
            x_[i] = 0.0;
            for (int j = 0; j < STATE_N; ++j) {
                P_[i][j] = 0.0;
            }
        }

        // 状态量：
        // x[0] = px [m]
        // x[1] = py [m]
        // x[2] = vx [m/s]
        // x[3] = vy [m/s]
        // x[4] = bax [mg]，XY 线加速度残余零偏
        // x[5] = bay [mg]
        P_[0][0] = 1e-4;     // 初始位置不确定度
        P_[1][1] = 1e-4;
        P_[2][2] = 1e-2;     // 初始速度不确定度
        P_[3][3] = 1e-2;
        P_[4][4] = 200.0;    // bias 初值未知，单位 mg^2
        P_[5][5] = 200.0;

        pathLength_ = 0.0f;
        rawPathLength_ = 0.0f;
        pathSpeedLp_ = 0.0;
        stationaryCount_ = 0;
        lastStationary_ = false;
            }

    void update(
        float fxRawMg,
        float fyRawMg,
        float fzRawMg,
        float gxBody,
        float gyBody,
        float yawDeg,
        float gxDps,
        float gyDps,
        float gzDps,
        float dt
    ) {
        if (dt <= 0.0f || dt > DT_MAX) {
            return;
        }

        Vec2 oldPos = position();

        // 加速度计输出是比力 specific force：f = a - g。
        // 因此真实线加速度 a = f + g。EKF 的控制输入在这里按比力关系生成。
        float linAxMg = fxRawMg + gxBody * 1000.0f;
        float linAyMg = fyRawMg + gyBody * 1000.0f;

        bool stationary = detectStationary(
            linAxMg, linAyMg,
            gxDps, gyDps, gzDps,
            fxRawMg, fyRawMg, fzRawMg
        );

        // 用当前 bias 估计得到预测前的 body/local XY 加速度，用于换向检测。
        // 注意：这里仍是比力补偿后的线加速度，不直接使用 raw acc。
        double axBodyBeforePredict = (static_cast<double>(linAxMg) - x_[4]) * MG_TO_MPS2;
        double ayBodyBeforePredict = (static_cast<double>(linAyMg) - x_[5]) * MG_TO_MPS2;

        predict(linAxMg, linAyMg, yawDeg, dt);

        if (stationary) {
            stationaryCount_++;

            // 静止时速度观测为 0：强 ZUPT。
            updateVelocityZero();

            // 静止时真实 XY 线加速度应为 0，
            // 因此 linA ≈ accel_bias，用它观测 bias。
            updateAccelBias(linAxMg, linAyMg);

            if (stationaryCount_ >= STATIONARY_CONFIRM_COUNT) {
                x_[2] = 0.0;
                x_[3] = 0.0;
            }

            lastStationary_ = true;
        } else {
            stationaryCount_ = 0;
            lastStationary_ = false;

            // ZUPT 改回只由“静止检测”触发。
            // 运动过程中的加速度符号翻转不再触发弱 ZUPT，避免转圈/摆动时被误判为速度归零。
        }

        symmetrizeP();

        Vec2 newPos = position();
        float dx = newPos.x - oldPos.x;
        float dy = newPos.y - oldPos.y;

        // rawPathLength_ 保留原始逐帧位置差累加，只用于调试；它很容易被位置抖动放大。
        if (!lastStationary_) {
            rawPathLength_ += std::sqrt(dx * dx + dy * dy);
        }

        // 对外显示的 pathLength_ 改为速度积分 + 门限 + 低通，避免 EKF 位置微抖被累计成很大的路径。
        // 这对“最后落点准，但总路径巨大”的情况尤其关键。
        updateGatedPathLength(axBodyBeforePredict, ayBodyBeforePredict, dt);
    }

    Vec2 position() const {
        return Vec2{static_cast<float>(x_[0]), static_cast<float>(x_[1])};
    }

    Vec2 velocity() const {
        return Vec2{static_cast<float>(x_[2]), static_cast<float>(x_[3])};
    }

    Vec2 accelBiasMg() const {
        return Vec2{static_cast<float>(x_[4]), static_cast<float>(x_[5])};
    }

    float displacement() const {
        Vec2 p = position();
        return std::sqrt(p.x * p.x + p.y * p.y);
    }

    float pathLength() const {
        return pathLength_;
    }

    float rawPathLength() const {
        return rawPathLength_;
    }

    bool lastStationary() const {
        return lastStationary_;
    }

    Vec2 positionStd() const {
        return Vec2{
            static_cast<float>(std::sqrt(std::fmax(P_[0][0], 0.0))),
            static_cast<float>(std::sqrt(std::fmax(P_[1][1], 0.0)))
        };
    }

    Vec2 velocityStd() const {
        return Vec2{
            static_cast<float>(std::sqrt(std::fmax(P_[2][2], 0.0))),
            static_cast<float>(std::sqrt(std::fmax(P_[3][3], 0.0)))
        };
    }

private:
    static constexpr int STATE_N = 6;

    // EKF 参数需要按实测调。
    // 这里单位分清楚：bias 状态是 mg，速度/位置是 SI 单位。
    static constexpr double MG_TO_MPS2 = G0 / 1000.0;
    static constexpr double EKF_ACC_NOISE_MPS2 = 0.12;       // 约 12 mg 的等效加速度噪声
    static constexpr double EKF_BIAS_RW_MG_SQRT_S = 0.20;    // bias 随机游走，mg/sqrt(s)
    static constexpr double EKF_ZUPT_VEL_NOISE = 0.012;      // 静止速度观测噪声，m/s
    static constexpr double EKF_BIAS_MEAS_NOISE_MG = 3.0;    // 静止时 bias 观测噪声，mg

    // path_length 统计参数。不要用逐帧位置差直接作为最终路径，会把 EKF 的微小抖动累计放大。
    static constexpr double PATH_SPEED_LP_ALPHA = 0.18;       // 速度低通系数
    static constexpr double PATH_SPEED_DEADBAND_MPS = 0.035;  // 低于 3.5 cm/s 当作噪声
    static constexpr double PATH_ACC_GATE_MPS2 = 0.06;        // 加速度太小时不认为在有效运动
    static constexpr double PATH_MAX_SPEED_MPS = 1.20;        // 防止瞬时速度尖峰污染路径

    double x_[STATE_N] {};       // [px, py, vx, vy, bax_mg, bay_mg]
    double P_[STATE_N][STATE_N] {};

    float pathLength_ = 0.0f;
    float rawPathLength_ = 0.0f;
    double pathSpeedLp_ = 0.0;
    int stationaryCount_ = 0;
    bool lastStationary_ = false;

    bool detectStationary(
        float linAxMg,
        float linAyMg,
        float gxDps,
        float gyDps,
        float gzDps,
        float axRawMg,
        float ayRawMg,
        float azRawMg
    ) const {
        float accNormMg = vec3Norm(axRawMg, ayRawMg, azRawMg);

        // 如果 Z 轴陀螺已做零偏校准，则静止检测应包含 gz；
        // 如果关闭 Z 轴校准，gz 原始零偏可能很大，则不能参与静止判断。
        float gyroNorm = 0.0f;
        if constexpr (CALIBRATE_GYRO_Z) {
            gyroNorm = vec3Norm(gxDps, gyDps, gzDps);
        } else {
            gyroNorm = std::sqrt(gxDps * gxDps + gyDps * gyDps);
            (void)gzDps;
        }

        float linAxCorr = linAxMg - static_cast<float>(x_[4]);
        float linAyCorr = linAyMg - static_cast<float>(x_[5]);
        float linNormMg = std::sqrt(linAxCorr * linAxCorr + linAyCorr * linAyCorr);

        return gyroNorm < STATIONARY_GYRO_DPS &&
               std::fabs(accNormMg - 1000.0f) < STATIONARY_ACC_NORM_ERR_MG &&
               linNormMg < STATIONARY_LIN_ACC_XY_MG;
    }

    void updateGatedPathLength(double axBodyMps2, double ayBodyMps2, float dtF) {
        if (dtF <= 0.0f || dtF > DT_MAX) {
            return;
        }

        double speed = std::sqrt(x_[2] * x_[2] + x_[3] * x_[3]);
        if (speed > PATH_MAX_SPEED_MPS) {
            speed = PATH_MAX_SPEED_MPS;
        }

        pathSpeedLp_ = (1.0 - PATH_SPEED_LP_ALPHA) * pathSpeedLp_ + PATH_SPEED_LP_ALPHA * speed;

        double linAcc = std::sqrt(axBodyMps2 * axBodyMps2 + ayBodyMps2 * ayBodyMps2);

        if (lastStationary_ || pathSpeedLp_ <= PATH_SPEED_DEADBAND_MPS || linAcc <= PATH_ACC_GATE_MPS2) {
            return;
        }

        double effectiveSpeed = pathSpeedLp_ - PATH_SPEED_DEADBAND_MPS;
        pathLength_ += static_cast<float>(effectiveSpeed * static_cast<double>(dtF));
    }

    void predict(float linAxMg, float linAyMg, float yawDeg, float dtF) {
        double dt = static_cast<double>(dtF);
        double dt2 = dt * dt;

        double c = 1.0;
        double s = 0.0;

        if constexpr (USE_YAW_FOR_XY_ROTATION) {
            double yawRad = static_cast<double>(yawDeg) * 0.017453292519943295;
            c = std::cos(yawRad);
            s = std::sin(yawRad);
        } else {
            (void)yawDeg;
        }

        double axBody = (static_cast<double>(linAxMg) - x_[4]) * MG_TO_MPS2;
        double ayBody = (static_cast<double>(linAyMg) - x_[5]) * MG_TO_MPS2;

        double axWorld = c * axBody - s * ayBody;
        double ayWorld = s * axBody + c * ayBody;

        x_[0] += x_[2] * dt + 0.5 * axWorld * dt2;
        x_[1] += x_[3] * dt + 0.5 * ayWorld * dt2;
        x_[2] += axWorld * dt;
        x_[3] += ayWorld * dt;

        double F[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            F[i][i] = 1.0;
        }

        F[0][2] = dt;
        F[1][3] = dt;

        // a_world = R * (lin - bias) * MG_TO_MPS2
        double dax_dbx = -MG_TO_MPS2 * c;
        double dax_dby =  MG_TO_MPS2 * s;
        double day_dbx = -MG_TO_MPS2 * s;
        double day_dby = -MG_TO_MPS2 * c;

        F[0][4] = 0.5 * dt2 * dax_dbx;
        F[0][5] = 0.5 * dt2 * dax_dby;
        F[1][4] = 0.5 * dt2 * day_dbx;
        F[1][5] = 0.5 * dt2 * day_dby;
        F[2][4] = dt * dax_dbx;
        F[2][5] = dt * dax_dby;
        F[3][4] = dt * day_dbx;
        F[3][5] = dt * day_dby;

        double FP[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                for (int k = 0; k < STATE_N; ++k) {
                    FP[i][j] += F[i][k] * P_[k][j];
                }
            }
        }

        double newP[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                for (int k = 0; k < STATE_N; ++k) {
                    newP[i][j] += FP[i][k] * F[j][k];
                }
            }
        }

        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                P_[i][j] = newP[i][j];
            }
        }

        // 加速度白噪声进入位置/速度。
        double qa = EKF_ACC_NOISE_MPS2 * EKF_ACC_NOISE_MPS2;
        addAccelProcessNoise(0, 2, qa, dt);
        addAccelProcessNoise(1, 3, qa, dt);

        // bias 随机游走。
        double qb = EKF_BIAS_RW_MG_SQRT_S * EKF_BIAS_RW_MG_SQRT_S * dt;
        P_[4][4] += qb;
        P_[5][5] += qb;
    }

    void addAccelProcessNoise(int posIdx, int velIdx, double qa, double dt) {
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt2 * dt2;

        P_[posIdx][posIdx] += 0.25 * dt4 * qa;
        P_[posIdx][velIdx] += 0.5 * dt3 * qa;
        P_[velIdx][posIdx] += 0.5 * dt3 * qa;
        P_[velIdx][velIdx] += dt2 * qa;
    }

    void updateVelocityZero() {
        double H[2][STATE_N] {};
        H[0][2] = 1.0;
        H[1][3] = 1.0;

        double z[2] = {0.0, 0.0};
        double r = EKF_ZUPT_VEL_NOISE * EKF_ZUPT_VEL_NOISE;
        update2(H, z, r, r);
    }

    void updateAccelBias(float linAxMg, float linAyMg) {
        double H[2][STATE_N] {};
        H[0][4] = 1.0;
        H[1][5] = 1.0;

        double z[2] = {
            static_cast<double>(linAxMg),
            static_cast<double>(linAyMg)
        };

        double r = EKF_BIAS_MEAS_NOISE_MG * EKF_BIAS_MEAS_NOISE_MG;
        update2(H, z, r, r);
    }


    void update1(const double H[STATE_N], double z, double r) {
        double hx = 0.0;
        for (int i = 0; i < STATE_N; ++i) {
            hx += H[i] * x_[i];
        }

        double y = z - hx;

        double S = r;
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                S += H[i] * P_[i][j] * H[j];
            }
        }

        if (std::fabs(S) < 1e-18) {
            return;
        }

        double PHt[STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                PHt[i] += P_[i][j] * H[j];
            }
        }

        double K[STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            K[i] = PHt[i] / S;
        }

        for (int i = 0; i < STATE_N; ++i) {
            x_[i] += K[i] * y;
        }

        double KH[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                KH[i][j] = K[i] * H[j];
            }
        }

        double IminusKH[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                IminusKH[i][j] = (i == j ? 1.0 : 0.0) - KH[i][j];
            }
        }

        double newP[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                for (int k = 0; k < STATE_N; ++k) {
                    newP[i][j] += IminusKH[i][k] * P_[k][j];
                }
            }
        }

        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                P_[i][j] = newP[i][j];
            }
        }
    }

    void update2(const double H[2][STATE_N], const double z[2], double r0, double r1) {
        double hx[2] {};
        for (int row = 0; row < 2; ++row) {
            for (int i = 0; i < STATE_N; ++i) {
                hx[row] += H[row][i] * x_[i];
            }
        }

        double y[2] = {z[0] - hx[0], z[1] - hx[1]};

        double S[2][2] {};
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 2; ++col) {
                for (int i = 0; i < STATE_N; ++i) {
                    for (int j = 0; j < STATE_N; ++j) {
                        S[row][col] += H[row][i] * P_[i][j] * H[col][j];
                    }
                }
            }
        }
        S[0][0] += r0;
        S[1][1] += r1;

        double det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
        if (std::fabs(det) < 1e-18) {
            return;
        }

        double invS[2][2] {
            { S[1][1] / det, -S[0][1] / det },
            { -S[1][0] / det, S[0][0] / det }
        };

        double PHt[STATE_N][2] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int row = 0; row < 2; ++row) {
                for (int j = 0; j < STATE_N; ++j) {
                    PHt[i][row] += P_[i][j] * H[row][j];
                }
            }
        }

        double K[STATE_N][2] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int col = 0; col < 2; ++col) {
                for (int k = 0; k < 2; ++k) {
                    K[i][col] += PHt[i][k] * invS[k][col];
                }
            }
        }

        for (int i = 0; i < STATE_N; ++i) {
            x_[i] += K[i][0] * y[0] + K[i][1] * y[1];
        }

        double KH[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                KH[i][j] = K[i][0] * H[0][j] + K[i][1] * H[1][j];
            }
        }

        double IminusKH[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                IminusKH[i][j] = (i == j ? 1.0 : 0.0) - KH[i][j];
            }
        }

        double newP[STATE_N][STATE_N] {};
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                for (int k = 0; k < STATE_N; ++k) {
                    newP[i][j] += IminusKH[i][k] * P_[k][j];
                }
            }
        }

        for (int i = 0; i < STATE_N; ++i) {
            for (int j = 0; j < STATE_N; ++j) {
                P_[i][j] = newP[i][j];
            }
        }
    }

    void symmetrizeP() {
        for (int i = 0; i < STATE_N; ++i) {
            for (int j = i + 1; j < STATE_N; ++j) {
                double v = 0.5 * (P_[i][j] + P_[j][i]);
                P_[i][j] = v;
                P_[j][i] = v;
            }

            if (P_[i][i] < 1e-12) {
                P_[i][i] = 1e-12;
            }
        }
    }
};

int main(int argc, char** argv) {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    std::string port = "/dev/ttyUSB0";
    int baudrate = 460800;

    if (argc >= 2) {
        port = argv[1];
    }

    if (argc >= 3) {
        baudrate = std::stoi(argv[2]);
    }

    try {
        std::cout << "Open serial: " << port
                  << " baudrate=" << baudrate << std::endl;

        SerialPort ser(port, baudrate);

        if (!initG354SamplingMode(ser)) {
            std::cerr << "G354 init command failed." << std::endl;
            return 1;
        }

        MahonyAHRS ahrs(MAHONY_KP, MAHONY_KI);

        G354Data gyroBias = calibrateGyroBias(ser, GYRO_CALIB_SAMPLES);
        std::cout << "Z gyro bias correction: "
                  << (CALIBRATE_GYRO_Z ? "enabled" : "disabled, gz free integration")
                  << std::endl;

        G354Data accInit {};
        if (!averageAccelForInit(ser, ACC_INIT_SAMPLES, accInit)) {
            std::cerr << "acc init failed." << std::endl;
            return 1;
        }

        if (!ahrs.initFromAccel(accInit.ax, accInit.ay, accInit.az, 0.0f)) {
            std::cerr << "Mahony initFromAccel failed." << std::endl;
            return 1;
        }

        float initRoll = 0.0f;
        float initPitch = 0.0f;
        float initYaw = 0.0f;
        ahrs.getEuler(initRoll, initPitch, initYaw);
        std::cout << "初始姿态 roll=" << initRoll
                  << " pitch=" << initPitch
                  << " yaw=" << initYaw
                  << std::endl;

        std::cout << "开始 AHRS warm-up: " << AHRS_WARMUP_SAMPLES
                  << " samples。此阶段只更新姿态，不积分路径。" << std::endl;

        auto lastT = std::chrono::steady_clock::now();
        int warmBad = 0;

        for (int i = 0; i < AHRS_WARMUP_SAMPLES && g_running; ++i) {
            auto loopStart = std::chrono::steady_clock::now();

            uint8_t frame[G354_FRAME_LEN];
            if (!readG354RawFrame(ser, frame)) {
                warmBad++;
                continue;
            }

            G354Data imu = parseG354Frame(frame);

            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastT).count();
            lastT = now;

            if (dt < DT_MIN || dt > DT_MAX) {
                dt = TARGET_DT;
            }

            float gxCal = imu.gx - gyroBias.gx;
            float gyCal = imu.gy - gyroBias.gy;
            float gzCal = CALIBRATE_GYRO_Z ? (imu.gz - gyroBias.gz) : imu.gz;

            ahrs.updateIMU(gxCal, gyCal, gzCal, imu.ax, imu.ay, imu.az, dt);

            auto loopEnd = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(loopEnd - loopStart).count();
            float sleepSec = TARGET_DT - elapsed;
            if (sleepSec > 0.0f) {
                std::this_thread::sleep_for(std::chrono::duration<float>(sleepSec));
            }
        }

        float warmRoll = 0.0f;
        float warmPitch = 0.0f;
        float warmYaw = 0.0f;
        ahrs.getEuler(warmRoll, warmPitch, warmYaw);
        std::cout << "warm-up 后姿态 roll=" << warmRoll
                  << " pitch=" << warmPitch
                  << " yaw=" << warmYaw
                  << " warm_bad=" << warmBad
                  << std::endl;

        PlanarEKF path;
        path.reset();

        std::cout << "进入姿态解算 + XY 重力补偿 + 路径积分循环。Ctrl+C 结束并输出最终位移。"
                  << " USE_YAW_FOR_XY_ROTATION=" << (USE_YAW_FOR_XY_ROTATION ? 1 : 0)
                  << std::endl;

        lastT = std::chrono::steady_clock::now();

        int frameCount = 0;
        int badFrameCount = 0;

        while (g_running) {
            auto loopStart = std::chrono::steady_clock::now();

            uint8_t frame[G354_FRAME_LEN];

            if (!readG354RawFrame(ser, frame)) {
                badFrameCount++;
                continue;
            }

            G354Data imu = parseG354Frame(frame);

            auto now = std::chrono::steady_clock::now();
            float dt = std::chrono::duration<float>(now - lastT).count();
            lastT = now;

            if (dt < DT_MIN || dt > DT_MAX) {
                dt = TARGET_DT;
            }

            float gxCal = imu.gx - gyroBias.gx;
            float gyCal = imu.gy - gyroBias.gy;
            float gzCal = CALIBRATE_GYRO_Z ? (imu.gz - gyroBias.gz) : imu.gz;

            ahrs.updateIMU(
                gxCal, gyCal, gzCal,
                imu.ax, imu.ay, imu.az,
                dt
            );

            float roll = 0.0f;
            float pitch = 0.0f;
            float yaw = 0.0f;
            ahrs.getEuler(roll, pitch, yaw);

            float q[4];
            ahrs.getQuaternion(q);

            float gxBody = 0.0f;
            float gyBody = 0.0f;
            float gzBody = 0.0f;
            gravityBodyFromQuaternion(q, gxBody, gyBody, gzBody);

            // raw_acc 是比力：raw_acc = linear_acc - gravity。
            // 因此 linear_acc = raw_acc + gravity。
            float linAxMg = imu.ax + gxBody * 1000.0f;
            float linAyMg = imu.ay + gyBody * 1000.0f;

            path.update(
                imu.ax,
                imu.ay,
                imu.az,
                gxBody,
                gyBody,
                yaw,
                gxCal,
                gyCal,
                gzCal,
                dt
            );

            frameCount++;

            if (frameCount % 10 == 0) {
                Vec2 pos = path.position();
                Vec2 vel = path.velocity();
                Vec2 accBias = path.accelBiasMg();
                Vec2 posStd = path.positionStd();
                Vec2 velStd = path.velocityStd();

                float accNormG = vec3Norm(imu.ax, imu.ay, imu.az) / 1000.0f;

                std::cout
                    << "roll:" << roll
                    << " pitch:" << pitch
                    << " yaw:" << yaw
                    << " | gyro[dps]="
                    << gxCal << "," << gyCal << "," << gzCal
                    << " | acc[mg]="
                    << imu.ax << "," << imu.ay << "," << imu.az
                    << " norm:" << accNormG << "g"
                    << " | gBody="
                    << gxBody << "," << gyBody << "," << gzBody
                    << " | linXY[mg]="
                    << linAxMg << "," << linAyMg
                    << " | vel[m/s]="
                    << vel.x << "," << vel.y
                    << " | pos[m]="
                    << pos.x << "," << pos.y
                    << " | disp[m]=" << path.displacement()
                    << " path[m]=" << path.pathLength()
                    << " rawPath[m]=" << path.rawPathLength()
                    << " | zupt=" << (path.lastStationary() ? 1 : 0)
                    << " | accBiasXY[mg]="
                    << accBias.x << "," << accBias.y
                    << " | ekfStdPos[m]="
                    << posStd.x << "," << posStd.y
                    << " | ekfStdVel[m/s]="
                    << velStd.x << "," << velStd.y
                    << " | dt=" << dt
                    << " | bad=" << badFrameCount
                    << std::endl;
            }

            auto loopEnd = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(loopEnd - loopStart).count();
            float sleepSec = TARGET_DT - elapsed;

            if (sleepSec > 0.0f) {
                std::this_thread::sleep_for(
                    std::chrono::duration<float>(sleepSec)
                );
            }
        }

        Vec2 finalPos = path.position();
        Vec2 finalVel = path.velocity();

        std::cout << "\n========== FINAL PATH RESULT ==========\n";
        std::cout << "final_x[m](最终估计位置的 X 坐标): " << finalPos.x << "\n";
        std::cout << "final_y[m](最终估计位置的 Y 坐标): " << finalPos.y << "\n";
        std::cout << "final_velocity_x[m/s](最终x轴速度): " << finalVel.x << "\n";
        std::cout << "final_velocity_y[m/s](最终y轴速度): " << finalVel.y << "\n";
        std::cout << "final_displacement[m](最终位移): " << path.displacement() << "\n";
        std::cout << "path_length[m](门限速度积分路径长度): " << path.pathLength() << "\n";
        std::cout << "raw_path_length[m](原始逐帧位置差路径，仅调试): " << path.rawPathLength() << "\n";
        std::cout << "valid_frames(有效帧数): " << frameCount << "\n";
        std::cout << "bad_frames(无效帧数): " << badFrameCount << "\n";
        std::cout << "=======================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
