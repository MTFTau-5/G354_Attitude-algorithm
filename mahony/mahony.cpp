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

    // 先扣陀螺仪零偏。输入单位仍为 deg/s。
    gx -= gyroBias[0];
    gy -= gyroBias[1];
    gz -= gyroBias[2];

    const float gyroNormDps = std::sqrt(gx * gx + gy * gy + gz * gz);
    const float accNorm = std::sqrt(ax * ax + ay * ay + az * az);
    const float accErrMg = std::fabs(accNorm - 1000.0f);
    const bool accStrictReliable =
        accNorm > 1.0f &&
        accErrMg < 30.0f &&
        gyroNormDps < 0.25f;

    const bool accWeakReliable =
        accNorm > 1.0f &&
        accErrMg < 70.0f &&
        gyroNormDps < 3.0f;

    float kpEff = 0.0f;
    float kiEff = 0.0f;

    if (accStrictReliable) {
        kpEff = kp;
        kiEff = ki;
    } else if (accWeakReliable) {
        kpEff = kp * 0.08f;
        kiEff = 0.0f;
    }

    // deg/s -> rad/s
    gx *= 0.0174532925f;
    gy *= 0.0174532925f;
    gz *= 0.0174532925f;

    if (kpEff > 0.0f) {
        ax /= accNorm;
        ay /= accNorm;
        az /= accNorm;

        // 当前姿态下，世界重力方向 [0,0,1] 投影到机体系。
        const float vx = 2.0f * (q1 * q3 - q0 * q2);
        const float vy = 2.0f * (q0 * q1 + q2 * q3);
        const float vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

        const float ex = ay * vz - az * vy;
        const float ey = az * vx - ax * vz;
        [[maybe_unused]] const float ez = ax * vy - ay * vx;

        if (kiEff > 0.0f) {
            integralError[0] += kiEff * ex * dt;
            integralError[1] += kiEff * ey * dt;
            integralError[2] = 0.0f;

            const float iLimit = 0.05f;
            integralError[0] = clampFloat(integralError[0], -iLimit, iLimit);
            integralError[1] = clampFloat(integralError[1], -iLimit, iLimit);
        } else {
            // 运动时不让积分项继续积累，防止被动态加速度污染。
            integralError[0] *= 0.98f;
            integralError[1] *= 0.98f;
            integralError[2] = 0.0f;
        }

        gx += kpEff * ex + integralError[0];
        gy += kpEff * ey + integralError[1];

        // 不用加速度修正 yaw：只有加速度计无法观测航向。
        // gz += kpEff * ez + integralError[2];
    } else {
        integralError[0] *= 0.98f;
        integralError[1] *= 0.98f;
        integralError[2] = 0.0f;
    }

    const float qDot0 = -0.5f * (q1 * gx + q2 * gy + q3 * gz);
    const float qDot1 =  0.5f * (q0 * gx + q2 * gz - q3 * gy);
    const float qDot2 =  0.5f * (q0 * gy - q1 * gz + q3 * gx);
    const float qDot3 =  0.5f * (q0 * gz + q1 * gy - q2 * gx);

    q0 += qDot0 * dt;
    q1 += qDot1 * dt;
    q2 += qDot2 * dt;
    q3 += qDot3 * dt;

    const float norm = invSqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
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
    const float gxBody = 2.0f * (q1 * q3 - q0 * q2);
    const float gyBody = 2.0f * (q0 * q1 + q2 * q3);

    linAx = axRaw - gxBody * GRAVITY_MG;
    linAy = ayRaw - gyBody * GRAVITY_MG;
}

void MahonyAHRS::getLinearAccelerationWorldMg(
    float axRaw, float ayRaw, float azRaw,
    float& linAxWorld, float& linAyWorld, float& linAzWorld
) const {
    // R_body_to_world(q) * acc_body
    const float r00 = 1.0f - 2.0f * (q2 * q2 + q3 * q3);
    const float r01 = 2.0f * (q1 * q2 - q0 * q3);
    const float r02 = 2.0f * (q1 * q3 + q0 * q2);

    const float r10 = 2.0f * (q1 * q2 + q0 * q3);
    const float r11 = 1.0f - 2.0f * (q1 * q1 + q3 * q3);
    const float r12 = 2.0f * (q2 * q3 - q0 * q1);

    const float r20 = 2.0f * (q1 * q3 - q0 * q2);
    const float r21 = 2.0f * (q2 * q3 + q0 * q1);
    const float r22 = 1.0f - 2.0f * (q1 * q1 + q2 * q2);

    const float axWorld = r00 * axRaw + r01 * ayRaw + r02 * azRaw;
    const float ayWorld = r10 * axRaw + r11 * ayRaw + r12 * azRaw;
    const float azWorld = r20 * axRaw + r21 * ayRaw + r22 * azRaw;

    linAxWorld = axWorld;
    linAyWorld = ayWorld;
    linAzWorld = azWorld - 1000.0f;
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
