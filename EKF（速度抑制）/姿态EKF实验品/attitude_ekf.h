#ifndef ATTITUDE_EKF_H
#define ATTITUDE_EKF_H

#include <cmath>

class AttitudeEKF {
public:
    // 姿态 EKF：名义状态 q + gyro residual bias，误差状态 [dtheta, dbg]。
    // acc_noise_unit：归一化重力方向观测噪声，0.03 约等效 30mg/约1.7deg 量级。
    // gyro_noise_dps_sqrt_hz：陀螺白噪声，用于姿态预测协方差。
    // gyro_bias_rw_dps_sqrt_s：陀螺残余零偏随机游走。
    AttitudeEKF(float acc_noise_unit = 0.035f,
                float gyro_noise_dps_sqrt_hz = 0.10f,
                float gyro_bias_rw_dps_sqrt_s = 0.005f,
                float acc_norm_min_g = 0.85f,
                float acc_norm_max_g = 1.15f);

    bool initFromAccel(float axMg, float ayMg, float azMg, float yawDeg = 0.0f);

    bool updateIMU(
        float gxDps, float gyDps, float gzDps,
        float axMg, float ayMg, float azMg,
        float dt
    );

    void getQuaternion(float q[4]) const;
    void getEuler(float& rollDeg, float& pitchDeg, float& yawDeg) const;

    float getLastAccNormG() const { return last_acc_norm_g_; }
    bool getLastAccUsed() const { return last_acc_used_; }

    void getGyroBiasDps(float bias[3]) const;

private:
    static constexpr int ERR_N = 6;

    double q_[4];        // [qw, qx, qy, qz]
    double bg_[3];       // residual gyro bias, rad/s
    double P_[ERR_N][ERR_N];

    double acc_noise_unit_;
    double gyro_noise_rad_sqrt_hz_;
    double gyro_bias_rw_rad_sqrt_s_;
    double acc_norm_min_g_;
    double acc_norm_max_g_;

    float last_acc_norm_g_;
    bool last_acc_used_;

    void resetCovariance();

    static void normalizeQuaternion(double q[4]);
    static void quatMultiply(const double a[4], const double b[4], double out[4]);
    static void deltaQuatFromRotVec(const double rv[3], double dq[4]);
    static void gravityBodyFromQuaternion(const double q[4], double gBody[3]);
    static void skew(const double v[3], double S[3][3]);

    void predict(const double gyroRad[3], double dt);
    void updateAccelGravity(const double measuredGravityBody[3]);
    void applyErrorState(const double dx[ERR_N]);
    void symmetrizeP();
};

#endif
