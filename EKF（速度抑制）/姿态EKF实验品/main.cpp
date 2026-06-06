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

#include "attitude_ekf_roll_pitch.h"

static std::atomic<bool> g_running(true);

static constexpr int G354_FRAME_LEN = 38;
static constexpr uint8_t G354_ADDR_WIN_CTRL        = 0x7E;
static constexpr uint8_t G354_ADDR_MODE_CTRL_HI    = 0x03;
static constexpr uint8_t G354_ADDR_UART_CTRL_LO    = 0x08;
static constexpr uint8_t G354_ADDR_BURST_CTRL1_LO  = 0x0C;
static constexpr uint8_t G354_ADDR_BURST_CTRL2_LO  = 0x0E;

static constexpr uint8_t G354_CMD_WINDOW0          = 0x00;
static constexpr uint8_t G354_CMD_WINDOW1          = 0x01;
static constexpr uint8_t G354_CMD_BEGIN_SAMPLING   = 0x01;
static constexpr uint8_t G354_CMD_END_SAMPLING     = 0x02;
static constexpr uint16_t G354_BURST_CTRL1_FIXED   = 0xF007;
static constexpr uint16_t G354_BURST_CTRL2_FIXED   = 0x7000;

static constexpr float SF_GYRO = 0.016f;       
static constexpr float SF_ACC  = 0.2f;      
static constexpr float SCALE_DIV = 65536.0f;   

static constexpr float G0 = 9.7970f;  // 太原本地重力加速度
static constexpr bool CALIBRATE_GYRO_Z = true;
static constexpr bool USE_YAW_FOR_XY_ROTATION = true;
static constexpr float ATT_EKF_ACC_NOISE_UNIT = 0.035f;         
static constexpr float ATT_EKF_GYRO_NOISE_DPS_SQRT_HZ = 0.10f;
static constexpr float ATT_EKF_BIAS_RW_DPS_SQRT_S = 0.005f; 
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

        AttitudeEKF ahrs(
            ATT_EKF_ACC_NOISE_UNIT,
            ATT_EKF_GYRO_NOISE_DPS_SQRT_HZ,
            ATT_EKF_BIAS_RW_DPS_SQRT_S
        );

        G354Data gyroBias = calibrateGyroBias(ser, GYRO_CALIB_SAMPLES);
        std::cout << "Roll/Pitch-only mode: yaw estimation disabled, z gyro residual bias locked." << std::endl;

        G354Data accInit {};
        if (!averageAccelForInit(ser, ACC_INIT_SAMPLES, accInit)) {
            std::cerr << "acc init failed." << std::endl;
            return 1;
        }

        if (!ahrs.initFromAccel(accInit.ax, accInit.ay, accInit.az, 0.0f)) {
            std::cerr << "AttitudeEKF initFromAccel failed." << std::endl;
            return 1;
        }

        float initRoll = 0.0f;
        float initPitch = 0.0f;
        ahrs.getRollPitch(initRoll, initPitch);
        std::cout << "初始姿态 roll=" << initRoll
                  << " pitch=" << initPitch
                  << " yaw=disabled"
                  << std::endl;

        std::cout << "开始 Roll/Pitch EKF warm-up: " << AHRS_WARMUP_SAMPLES
                  << " samples。只估计 roll/pitch，不估计 yaw，不做路径积分。" << std::endl;

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

            // Roll/Pitch-only：不使用 gz 进行 yaw 传播。
            ahrs.updateIMU(gxCal, gyCal, 0.0f, imu.ax, imu.ay, imu.az, dt);

            auto loopEnd = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(loopEnd - loopStart).count();
            float sleepSec = TARGET_DT - elapsed;
            if (sleepSec > 0.0f) {
                std::this_thread::sleep_for(std::chrono::duration<float>(sleepSec));
            }
        }

        float warmRoll = 0.0f;
        float warmPitch = 0.0f;
        ahrs.getRollPitch(warmRoll, warmPitch);
        std::cout << "warm-up 后姿态 roll=" << warmRoll
                  << " pitch=" << warmPitch
                  << " yaw=disabled"
                  << " warm_bad=" << warmBad
                  << std::endl;

        std::cout << "进入 Roll/Pitch-only EKF 姿态解算循环。Ctrl+C 结束。" << std::endl;

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
            float gzCalForLog = imu.gz - gyroBias.gz;

            ahrs.updateIMU(
                gxCal, gyCal, 0.0f,
                imu.ax, imu.ay, imu.az,
                dt
            );

            frameCount++;

            if (frameCount % 10 == 0) {
                float roll = 0.0f;
                float pitch = 0.0f;
                ahrs.getRollPitch(roll, pitch);

                float attBg[3] = {0.0f, 0.0f, 0.0f};
                ahrs.getGyroBiasDps(attBg);

                float accNormG = vec3Norm(imu.ax, imu.ay, imu.az) / 1000.0f;

                std::cout
                    << "roll:" << roll
                    << " pitch:" << pitch
                    << " | gyro[dps]="
                    << gxCal << "," << gyCal << "," << gzCalForLog
                    << " | attBg[dps]="
                    << attBg[0] << "," << attBg[1] << "," << attBg[2]
                    << " | acc[mg]="
                    << imu.ax << "," << imu.ay << "," << imu.az
                    << " norm:" << accNormG << "g"
                    << " acc_used:" << (ahrs.getLastAccUsed() ? 1 : 0)
                    << " | dt=" << dt
                    << " | bad=" << badFrameCount
                    << std::endl;
            }

            auto loopEnd = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(loopEnd - loopStart).count();
            float sleepSec = TARGET_DT - elapsed;

            if (sleepSec > 0.0f) {
                std::this_thread::sleep_for(std::chrono::duration<float>(sleepSec));
            }
        }

        std::cout << "\n========== FINAL ROLL/PITCH RESULT ==========" << std::endl;
        float finalRoll = 0.0f;
        float finalPitch = 0.0f;
        ahrs.getRollPitch(finalRoll, finalPitch);
        std::cout << "final_roll[deg]: " << finalRoll << std::endl;
        std::cout << "final_pitch[deg]: " << finalPitch << std::endl;
        std::cout << "valid_frames: " << frameCount << std::endl;
        std::cout << "bad_frames: " << badFrameCount << std::endl;
        std::cout << "============================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
