#ifndef MAHONY_AHRS_H
#define MAHONY_AHRS_H

#include <cmath>

class MahonyAHRS {
public:
    MahonyAHRS(float kp = 1.0f, float ki = 0.0f);

    void updateIMU(
        float gx, float gy, float gz,
        float ax, float ay, float az,
        float dt
    );

    void getQuaternion(float q[4]) const;
    void getEuler(float& roll, float& pitch, float& yaw) const;
    void setGyroBias(float bx, float by, float bz);
    void setGains(float kp_, float ki_);
    void reset();

    // 兼容旧接口：返回机体系 XY 线加速度，不建议继续用于路径积分。
    void getLinearAccelerationXYMg(
        float axRaw, float ayRaw,
        float& linAx, float& linAy
    ) const;

    // 新接口：完整旋转到世界系，然后扣除世界 Z 轴重力。
    // 输入单位：mg；输出单位：mg。
    void getLinearAccelerationWorldMg(
        float axRaw, float ayRaw, float azRaw,
        float& linAxWorld, float& linAyWorld, float& linAzWorld
    ) const;

private:
    float q0, q1, q2, q3;

    float integralError[3];

    float kp;
    float ki;

    float gyroBias[3];

    static float invSqrt(float x);
    static float clampFloat(float x, float minVal, float maxVal);
};

#endif
