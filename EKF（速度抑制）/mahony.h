#ifndef MAHONY_AHRS_H
#define MAHONY_AHRS_H

#include <cmath>

class MahonyAHRS {
public:
    MahonyAHRS(float kp = 1.0f, float ki = 0.0f,
               float acc_norm_min_g = 0.85f, float acc_norm_max_g = 1.15f);

    // 用 G354 原始加速度初始化 roll/pitch。
    // 注意：G354 原始加速度是 specific force，静止时与真实重力方向相反。
    bool initFromAccel(float axMg, float ayMg, float azMg, float yawDeg = 0.0f);

    bool updateIMU(
        float gx, float gy, float gz,
        float ax, float ay, float az,
        float dt
    );

    void getQuaternion(float q[4]) const;
    void getEuler(float& roll, float& pitch, float& yaw) const;

    float getLastAccNormG() const { return last_acc_norm_g; }
    bool getLastAccUsed() const { return last_acc_used; }

private:
    float q0, q1, q2, q3;

    float integralError[3];

    float kp;
    float ki;

    float acc_norm_min_g;
    float acc_norm_max_g;

    float last_acc_norm_g;
    bool last_acc_used;

    static void normalizeQuaternion(float& q0, float& q1, float& q2, float& q3);
};

#endif
