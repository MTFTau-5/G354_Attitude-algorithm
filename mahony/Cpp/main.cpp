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
static constexpr uint8_t G354_ADDR_BURST_CTRL1_HI  = 0x0D;
static constexpr uint8_t G354_ADDR_BURST_CTRL2_LO  = 0x0E;
static constexpr uint8_t G354_ADDR_BURST_CTRL2_HI  = 0x0F;

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

static constexpr float G0 = 9.80665f;

static constexpr float MAHONY_KP = 1.0f;
static constexpr float MAHONY_KI = 0.005f;

static constexpr int GYRO_CALIB_SAMPLES = 500;
static constexpr float TARGET_HZ = 100.0f;
static constexpr float TARGET_DT = 1.0f / TARGET_HZ;

static constexpr float DT_MIN = 0.001f;
static constexpr float DT_MAX = 0.05f;
static constexpr float STATIONARY_GYRO_DPS = 0.08f;
static constexpr float STATIONARY_ACC_NORM_ERR_MG = 35.0f;
static constexpr float STATIONARY_LIN_ACC_XY_MG = 60.0f;
static constexpr int STATIONARY_CONFIRM_COUNT = 5;

struct G354Data {
    float gx;  // deg/s
    float gy;
    float gz;

    float ax;  // mg
    float ay;
    float az;
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
            } else if (ret == 0) {
                return false;
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

    /*
     * 解析 G354 帧数据：
     *
     * gyro_x: data[7:11]
     * gyro_y: data[11:15]
     * gyro_z: data[15:19]
     * acc_x : data[19:23]
     * acc_y : data[23:27]
     * acc_z : data[27:31]
     *
     */
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
    /*
     * 等价于 Python：
     * ser.write(bytes([0x80, 0x00, 0x0d]))
     * data = ser.read(38)
     */
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

    // G354 UART register writes need a small inter-command gap.
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

    // 1) Force exit from sampling mode first. This is the real "stop sampling" command.
    if (!setG354Window(ser, G354_CMD_WINDOW0) ||
        !writeG354Reg8(ser, G354_ADDR_MODE_CTRL_HI, G354_CMD_END_SAMPLING)) {
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ser.flushIO();

    // 2) Window 1 contains UART_CTRL and BURST_CTRL registers.
    if (!setG354Window(ser, G354_CMD_WINDOW1)) {
        return false;
    }

    // 3) Manual UART burst read mode. We actively send 0x80 0x00 0x0D for every sample,
    //    so disable UART auto output to avoid unsolicited bytes in the serial buffer.
    if (!writeG354Reg8(ser, G354_ADDR_UART_CTRL_LO, 0x00)) {
        return false;
    }

    // 4) Fix the burst packet to exactly match parseG354Frame():
    //    [0] 0x80
    //    [1..2]   FLAG
    //    [3..6]   TEMP 32-bit
    //    [7..18]  GYRO XYZ 32-bit each
    //    [19..30] ACC  XYZ 32-bit each
    //    [31..32] GPIO
    //    [33..34] COUNT
    //    [35..36] CHECKSUM
    //    [37] 0x0D
    if (!writeG354Reg16Parts(ser, G354_ADDR_BURST_CTRL1_LO, G354_BURST_CTRL1_FIXED) ||
        !writeG354Reg16Parts(ser, G354_ADDR_BURST_CTRL2_LO, G354_BURST_CTRL2_FIXED)) {
        return false;
    }

    // 5) Read back and verify the two burst control registers while still in config mode.
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

    // 6) Return to Window 0 and enter sampling mode.
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

    std::cout << "陀螺仪零偏完成（吗） bias[dps] = "
              << bias.gx << ", " << bias.gy << ", " << bias.gz
              << std::endl;

    return bias;
}

class PlanarPathIntegrator {
public:
    void update(
        float linAxMg,
        float linAyMg,
        float yawDeg,
        float gxDps,
        float gyDps,
        float gzDps,
        float axRawMg,
        float ayRawMg,
        float azRawMg,
        float dt
    ) {
        if (dt <= 0.0f || dt > DT_MAX) {
            return;
        }

        float accNormMg = vec3Norm(axRawMg, ayRawMg, azRawMg);
        float gyroNorm = vec3Norm(gxDps, gyDps, gzDps);
        float linNormMg = std::sqrt(linAxMg * linAxMg + linAyMg * linAyMg);

        bool stationary =
            gyroNorm < STATIONARY_GYRO_DPS &&
            std::fabs(accNormMg - 1000.0f) < STATIONARY_ACC_NORM_ERR_MG &&
            linNormMg < STATIONARY_LIN_ACC_XY_MG;

        if (stationary) {
            stationaryCount_++;
            const float alpha = 0.01f;
            accBiasMg_.x = (1.0f - alpha) * accBiasMg_.x + alpha * linAxMg;
            accBiasMg_.y = (1.0f - alpha) * accBiasMg_.y + alpha * linAyMg;

            if (stationaryCount_ >= STATIONARY_CONFIRM_COUNT) {
                vel_.x = 0.0f;
                vel_.y = 0.0f;
            }

            lastStationary_ = true;
            return;
        }

        stationaryCount_ = 0;
        lastStationary_ = false;

        float axBodyMps2 = (linAxMg - accBiasMg_.x) * G0 / 1000.0f;
        float ayBodyMps2 = (linAyMg - accBiasMg_.y) * G0 / 1000.0f;
        float yawRad = yawDeg * 0.0174532925f;
        float c = std::cos(yawRad);
        float s = std::sin(yawRad);

        float axWorld = c * axBodyMps2 - s * ayBodyMps2;
        float ayWorld = s * axBodyMps2 + c * ayBodyMps2;

        Vec2 oldPos = pos_;

        pos_.x += vel_.x * dt + 0.5f * axWorld * dt * dt;
        pos_.y += vel_.y * dt + 0.5f * ayWorld * dt * dt;

        vel_.x += axWorld * dt;
        vel_.y += ayWorld * dt;

        float dx = pos_.x - oldPos.x;
        float dy = pos_.y - oldPos.y;
        pathLength_ += std::sqrt(dx * dx + dy * dy);
    }

    Vec2 position() const {
        return pos_;
    }

    Vec2 velocity() const {
        return vel_;
    }

    Vec2 accelBiasMg() const {
        return accBiasMg_;
    }

    float displacement() const {
        return std::sqrt(pos_.x * pos_.x + pos_.y * pos_.y);
    }

    float pathLength() const {
        return pathLength_;
    }

    bool lastStationary() const {
        return lastStationary_;
    }

private:
    Vec2 pos_ {};
    Vec2 vel_ {};
    Vec2 accBiasMg_ {};

    float pathLength_ = 0.0f;

    int stationaryCount_ = 0;
    bool lastStationary_ = false;
};

int main(int argc, char** argv) {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    std::string port = "/dev/cu.usbmodem59090479771";
    int baudrate = 230400;

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
        ahrs.setGyroBias(gyroBias.gx, gyroBias.gy, gyroBias.gz);

        PlanarPathIntegrator path;

        std::cout << "进入姿态解算 + XY 重力补偿 + 路径积分循环。Ctrl+C 结束并输出最终位移,可以看精度的说。"
                  << std::endl;

        auto lastT = std::chrono::steady_clock::now();

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

            ahrs.updateIMU(
                imu.gx, imu.gy, imu.gz,
                imu.ax, imu.ay, imu.az,
                dt
            );

            float roll, pitch, yaw;
            ahrs.getEuler(roll, pitch, yaw);

            float linAxMg, linAyMg;
            ahrs.getLinearAccelerationXYMg(imu.ax, imu.ay, linAxMg, linAyMg);

            float gxCal = imu.gx - gyroBias.gx;
            float gyCal = imu.gy - gyroBias.gy;
            float gzCal = imu.gz - gyroBias.gz;

            path.update(
                linAxMg,
                linAyMg,
                yaw,
                gxCal,
                gyCal,
                gzCal,
                imu.ax,
                imu.ay,
                imu.az,
                dt
            );

            frameCount++;

            if (frameCount % 10 == 0) {
                Vec2 pos = path.position();
                Vec2 vel = path.velocity();
                Vec2 accBias = path.accelBiasMg();

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
                    << " | linXY[mg]="
                    << linAxMg << "," << linAyMg
                    << " | vel[m/s]="
                    << vel.x << "," << vel.y
                    << " | pos[m]="
                    << pos.x << "," << pos.y
                    << " | disp[m]=" << path.displacement()
                    << " path[m]=" << path.pathLength()
                    << " | zupt=" << (path.lastStationary() ? 1 : 0)
                    << " | accBiasXY[mg]="
                    << accBias.x << "," << accBias.y
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
        std::cout << "path_length[m](路径长度): " << path.pathLength() << "\n";
        std::cout << "valid_frames(有效帧数): " << frameCount << "\n";
        std::cout << "bad_frames(无效帧数): " << badFrameCount << "\n";
        std::cout << "=======================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}