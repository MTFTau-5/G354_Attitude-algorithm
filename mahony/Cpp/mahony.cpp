// 有关yaw请看94行代码
// 因为只有加速度计无法约束yaw，所以陀螺仪的零偏对yaw估计影响较大。
// 如果陀螺仪零偏不准确，可能会导致yaw估计出现较大误差，进而影响路径积分的准确性。
// 所以没有用gz修正

#include "mahony.h"

MahonyAHRS::MahonyAHRS(float kp_, float ki_)
    : q0(1.0f), q1(0.0f), q2(0.0f), q3(0.0f),
      kp(kp_), ki(ki_)
{
    integralError[0] = 0.0f;
    integralError[1] = 0.0f;
    integralError[2] = 0.0f;

    gyroBias[0] = 0.0f;
    gyroBias[1] = 0.0f;
    gyroBias[2] = 0.0f;
}

void MahonyAHRS::setGyroBias(float bx, float by, float bz) {
    gyroBias[0] = bx;
    gyroBias[1] = by;
    gyroBias[2] = bz;
}

void MahonyAHRS::setGains(float kp_, float ki_) {
    kp = kp_;
    ki = ki_;
}

void MahonyAHRS::reset() {
    q0 = 1.0f;
    q1 = 0.0f;
    q2 = 0.0f;
    q3 = 0.0f;

    integralError[0] = 0.0f;
    integralError[1] = 0.0f;
    integralError[2] = 0.0f;
}

void MahonyAHRS::updateIMU(
    float gx, float gy, float gz,
    float ax, float ay, float az,
    float dt
) {
    if (dt <= 0.0f || dt > 0.05f) {
        return;
    }
    gx -= gyroBias[0];
    gy -= gyroBias[1];
    gz -= gyroBias[2];

    /*
     * 2. deg/s -> rad/s
     */
    gx *= 0.0174532925f;
    gy *= 0.0174532925f;
    gz *= 0.0174532925f;

    float accNorm = std::sqrt(ax * ax + ay * ay + az * az);

    bool accReliable = (accNorm > 850.0f && accNorm < 1150.0f);

    if (accReliable) {
        ax /= accNorm;
        ay /= accNorm;
        az /= accNorm;
        float vx = 2.0f * (q1 * q3 - q0 * q2);
        float vy = 2.0f * (q0 * q1 + q2 * q3);
        float vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
        float ex = ay * vz - az * vy;
        float ey = az * vx - ax * vz;
        [[maybe_unused]] float ez = ax * vy - ay * vx;

        if (ki > 0.0f) {
            integralError[0] += ki * ex * dt;
            integralError[1] += ki * ey * dt;
            integralError[2] = 0.0f;

            const float iLimit = 0.1f;

            integralError[0] = clampFloat(integralError[0], -iLimit, iLimit);
            integralError[1] = clampFloat(integralError[1], -iLimit, iLimit);
        } else {
            integralError[0] = 0.0f;
            integralError[1] = 0.0f;
            integralError[2] = 0.0f;
        }

        gx += kp * ex + integralError[0];
        gy += kp * ey + integralError[1];

        
        // 不建议：
        // gz += kp * ez + integralError[2];
        // 因为只有加速度计无法约束 yaw。
         
    } else {
        integralError[0] *= 0.99f;
        integralError[1] *= 0.99f;
        integralError[2] = 0.0f;
    }

    float qDot0 = -0.5f * (q1 * gx + q2 * gy + q3 * gz);
    float qDot1 =  0.5f * (q0 * gx + q2 * gz - q3 * gy);
    float qDot2 =  0.5f * (q0 * gy - q1 * gz + q3 * gx);
    float qDot3 =  0.5f * (q0 * gz + q1 * gy - q2 * gx);

    q0 += qDot0 * dt;
    q1 += qDot1 * dt;
    q2 += qDot2 * dt;
    q3 += qDot3 * dt;

    float norm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);

    q0 *= norm;
    q1 *= norm;
    q2 *= norm;
    q3 *= norm;
}

void MahonyAHRS::getQuaternion(float q[4]) const {
    q[0] = q0;
    q[1] = q1;
    q[2] = q2;
    q[3] = q3;
}

void MahonyAHRS::getEuler(float& roll, float& pitch, float& yaw) const {
    roll = std::atan2(
        2.0f * (q0 * q1 + q2 * q3),
        1.0f - 2.0f * (q1 * q1 + q2 * q2)
    );

    float sinPitch = 2.0f * (q0 * q2 - q3 * q1);
    sinPitch = clampFloat(sinPitch, -1.0f, 1.0f);
    pitch = std::asin(sinPitch);

    yaw = std::atan2(
        2.0f * (q1 * q2 + q0 * q3),
        1.0f - 2.0f * (q2 * q2 + q3 * q3)
    );

    roll  *= 57.2957795f;
    pitch *= 57.2957795f;
    yaw   *= 57.2957795f;
}

void MahonyAHRS::getLinearAccelerationXYMg(
    float axRaw, float ayRaw,
    float& linAx, float& linAy
) const {
    const float GRAVITY_MG = 1000.0f;
    float gxBody = 2.0f * (q1 * q3 - q0 * q2);
    float gyBody = 2.0f * (q0 * q1 + q2 * q3);

    linAx = axRaw - gxBody * GRAVITY_MG;
    linAy = ayRaw - gyBody * GRAVITY_MG;
}

float MahonyAHRS::invSqrt(float x) {
    if (x <= 0.0f) {
        return 1.0f;
    }

    return 1.0f / std::sqrt(x);
}

float MahonyAHRS::clampFloat(float x, float minVal, float maxVal) {
    if (x < minVal) {
        return minVal;
    }

    if (x > maxVal) {
        return maxVal;
    }

    return x;
}