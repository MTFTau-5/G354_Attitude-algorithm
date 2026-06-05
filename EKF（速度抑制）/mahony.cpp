#include "mahony.h"

MahonyAHRS::MahonyAHRS(float kp_, float ki_,
                       float acc_norm_min_g_, float acc_norm_max_g_)
    : q0(1.0f), q1(0.0f), q2(0.0f), q3(0.0f),
      kp(kp_), ki(ki_),
      acc_norm_min_g(acc_norm_min_g_), acc_norm_max_g(acc_norm_max_g_),
      last_acc_norm_g(0.0f), last_acc_used(false)
{
    integralError[0] = 0.0f;
    integralError[1] = 0.0f;
    integralError[2] = 0.0f;
}

void MahonyAHRS::normalizeQuaternion(float& q0, float& q1, float& q2, float& q3) {
    float n = std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    if (n <= 1e-12f) {
        q0 = 1.0f;
        q1 = 0.0f;
        q2 = 0.0f;
        q3 = 0.0f;
        return;
    }

    float inv_n = 1.0f / n;
    q0 *= inv_n;
    q1 *= inv_n;
    q2 *= inv_n;
    q3 *= inv_n;
}

bool MahonyAHRS::initFromAccel(float axMg, float ayMg, float azMg, float yawDeg) {
    float axG = axMg / 1000.0f;
    float ayG = ayMg / 1000.0f;
    float azG = azMg / 1000.0f;

    float n = std::sqrt(axG * axG + ayG * ayG + azG * azG);
    if (n <= 1e-12f) {
        return false;
    }

    // G354 原始加速度是 specific force。
    // 静止时 raw_acc ≈ -gravity_body，因此这里取反得到真实重力方向。
    float gx = -axG / n;
    float gy = -ayG / n;
    float gz = -azG / n;

    float roll = std::atan2(gy, gz);
    float pitch = std::atan2(-gx, std::sqrt(gy * gy + gz * gz));

    const float DEG2RAD = 0.0174532925f;
    float yaw = yawDeg * DEG2RAD;

    float cr = std::cos(roll * 0.5f);
    float sr = std::sin(roll * 0.5f);
    float cp = std::cos(pitch * 0.5f);
    float sp = std::sin(pitch * 0.5f);
    float cy = std::cos(yaw * 0.5f);
    float sy = std::sin(yaw * 0.5f);

    q0 = cy * cp * cr + sy * sp * sr;
    q1 = cy * cp * sr - sy * sp * cr;
    q2 = sy * cp * sr + cy * sp * cr;
    q3 = sy * cp * cr - cy * sp * sr;

    normalizeQuaternion(q0, q1, q2, q3);

    integralError[0] = 0.0f;
    integralError[1] = 0.0f;
    integralError[2] = 0.0f;

    return true;
}

bool MahonyAHRS::updateIMU(
    float gx, float gy, float gz,
    float ax, float ay, float az,
    float dt
) {
    if (dt <= 0.0f) {
        return false;
    }

    // 陀螺仪 deg/s -> rad/s
    const float DEG2RAD = 0.0174532925f;
    float gx_rad = gx * DEG2RAD;
    float gy_rad = gy * DEG2RAD;
    float gz_rad = gz * DEG2RAD;

    // 加速度 mg -> g
    float ax_g = ax / 1000.0f;
    float ay_g = ay / 1000.0f;
    float az_g = az / 1000.0f;

    float acc_norm = std::sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
    last_acc_norm_g = acc_norm;

    bool use_acc = (acc_norm >= acc_norm_min_g && acc_norm <= acc_norm_max_g);
    last_acc_used = use_acc;

    float ex = 0.0f;
    float ey = 0.0f;
    float ez = 0.0f;

    if (use_acc && acc_norm > 1e-12f) {
        float inv_norm = 1.0f / acc_norm;

        // G354 原始加速度是 specific force，静止时与真实重力方向相反。
        // Mahony 这里比较的应该是真实重力方向，所以必须取反。
        float ax_n = -ax_g * inv_norm;
        float ay_n = -ay_g * inv_norm;
        float az_n = -az_g * inv_norm;

        // 从当前四元数推导出的真实重力方向，机体系。
        // q=[1,0,0,0] 时 gravity_body=[0,0,+1]，即 z 向下。
        float vx = 2.0f * (q1 * q3 - q0 * q2);
        float vy = 2.0f * (q0 * q1 + q2 * q3);
        float vz = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;

        // error = cross(measured_gravity, estimated_gravity)
        ex = ay_n * vz - az_n * vy;
        ey = az_n * vx - ax_n * vz;
        ez = ax_n * vy - ay_n * vx;

        if (ki > 0.0f) {
            integralError[0] += ki * ex * dt;
            integralError[1] += ki * ey * dt;
            integralError[2] += ki * ez * dt;
        } else {
            integralError[0] = 0.0f;
            integralError[1] = 0.0f;
            integralError[2] = 0.0f;
        }
    }

    // 6轴只能用重力约束 roll/pitch，无法约束 yaw。
    float wx = gx_rad + kp * ex + integralError[0];
    float wy = gy_rad + kp * ey + integralError[1];
    float wz = gz_rad + kp * ez + integralError[2];

    float qDot0 = -0.5f * (q1 * wx + q2 * wy + q3 * wz);
    float qDot1 =  0.5f * (q0 * wx + q2 * wz - q3 * wy);
    float qDot2 =  0.5f * (q0 * wy - q1 * wz + q3 * wx);
    float qDot3 =  0.5f * (q0 * wz + q1 * wy - q2 * wx);

    q0 += qDot0 * dt;
    q1 += qDot1 * dt;
    q2 += qDot2 * dt;
    q3 += qDot3 * dt;

    normalizeQuaternion(q0, q1, q2, q3);

    return true;
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
    if (sinPitch > 1.0f) {
        sinPitch = 1.0f;
    }
    if (sinPitch < -1.0f) {
        sinPitch = -1.0f;
    }
    pitch = std::asin(sinPitch);

    yaw = std::atan2(
        2.0f * (q1 * q2 + q0 * q3),
        1.0f - 2.0f * (q2 * q2 + q3 * q3)
    );

    const float RAD2DEG = 57.2957795f;
    roll  *= RAD2DEG;
    pitch *= RAD2DEG;
    yaw   *= RAD2DEG;
}
