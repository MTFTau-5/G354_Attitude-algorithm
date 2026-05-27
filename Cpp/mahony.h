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
    void getLinearAccelerationXYMg(
        float axRaw, float ayRaw,
        float& linAx, float& linAy
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