#include "attitude_ekf.h"

#include <algorithm>
#include <cstring>

namespace {
constexpr double DEG2RAD = 0.017453292519943295;
constexpr double RAD2DEG = 57.29577951308232;
}

AttitudeEKF::AttitudeEKF(float acc_noise_unit,
                         float gyro_noise_dps_sqrt_hz,
                         float gyro_bias_rw_dps_sqrt_s,
                         float acc_norm_min_g,
                         float acc_norm_max_g)
    : acc_noise_unit_(acc_noise_unit),
      gyro_noise_rad_sqrt_hz_(gyro_noise_dps_sqrt_hz * DEG2RAD),
      gyro_bias_rw_rad_sqrt_s_(gyro_bias_rw_dps_sqrt_s * DEG2RAD),
      acc_norm_min_g_(acc_norm_min_g),
      acc_norm_max_g_(acc_norm_max_g),
      last_acc_norm_g_(0.0f),
      last_acc_used_(false)
{
    q_[0] = 1.0;
    q_[1] = 0.0;
    q_[2] = 0.0;
    q_[3] = 0.0;
    bg_[0] = 0.0;
    bg_[1] = 0.0;
    bg_[2] = 0.0;
    resetCovariance();
}

void AttitudeEKF::resetCovariance() {
    for (int i = 0; i < ERR_N; ++i) {
        for (int j = 0; j < ERR_N; ++j) {
            P_[i][j] = 0.0;
        }
    }

    // 初始角度误差标准差约 5deg；残余陀螺零偏约 0.2dps。
    const double angleStd = 5.0 * DEG2RAD;
    const double biasStd = 0.20 * DEG2RAD;

    P_[0][0] = angleStd * angleStd;
    P_[1][1] = angleStd * angleStd;
    P_[2][2] = angleStd * angleStd;
    P_[3][3] = biasStd * biasStd;
    P_[4][4] = biasStd * biasStd;
    P_[5][5] = biasStd * biasStd;
}

void AttitudeEKF::normalizeQuaternion(double q[4]) {
    double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n <= 1e-18) {
        q[0] = 1.0;
        q[1] = 0.0;
        q[2] = 0.0;
        q[3] = 0.0;
        return;
    }
    double inv = 1.0 / n;
    q[0] *= inv;
    q[1] *= inv;
    q[2] *= inv;
    q[3] *= inv;
}

void AttitudeEKF::quatMultiply(const double a[4], const double b[4], double out[4]) {
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

void AttitudeEKF::deltaQuatFromRotVec(const double rv[3], double dq[4]) {
    double angle = std::sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
    if (angle < 1e-12) {
        dq[0] = 1.0;
        dq[1] = 0.5 * rv[0];
        dq[2] = 0.5 * rv[1];
        dq[3] = 0.5 * rv[2];
        normalizeQuaternion(dq);
        return;
    }

    double half = 0.5 * angle;
    double s = std::sin(half) / angle;
    dq[0] = std::cos(half);
    dq[1] = rv[0] * s;
    dq[2] = rv[1] * s;
    dq[3] = rv[2] * s;
    normalizeQuaternion(dq);
}

void AttitudeEKF::gravityBodyFromQuaternion(const double q[4], double gBody[3]) {
    // 与原 Mahony 版本保持一致：q=[1,0,0,0] 时 gravity_body=[0,0,+1]。
    gBody[0] = 2.0 * (q[1] * q[3] - q[0] * q[2]);
    gBody[1] = 2.0 * (q[0] * q[1] + q[2] * q[3]);
    gBody[2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
}

void AttitudeEKF::skew(const double v[3], double S[3][3]) {
    S[0][0] = 0.0;     S[0][1] = -v[2];  S[0][2] = v[1];
    S[1][0] = v[2];    S[1][1] = 0.0;    S[1][2] = -v[0];
    S[2][0] = -v[1];   S[2][1] = v[0];   S[2][2] = 0.0;
}

bool AttitudeEKF::initFromAccel(float axMg, float ayMg, float azMg, float yawDeg) {
    double axG = static_cast<double>(axMg) / 1000.0;
    double ayG = static_cast<double>(ayMg) / 1000.0;
    double azG = static_cast<double>(azMg) / 1000.0;

    double n = std::sqrt(axG * axG + ayG * ayG + azG * azG);
    if (n <= 1e-12) {
        return false;
    }

    // G354 原始加速度是 specific force，静止时 raw_acc ≈ -gravity_body。
    double gx = -axG / n;
    double gy = -ayG / n;
    double gz = -azG / n;

    double roll = std::atan2(gy, gz);
    double pitch = std::atan2(-gx, std::sqrt(gy * gy + gz * gz));
    double yaw = static_cast<double>(yawDeg) * DEG2RAD;

    double cr = std::cos(roll * 0.5);
    double sr = std::sin(roll * 0.5);
    double cp = std::cos(pitch * 0.5);
    double sp = std::sin(pitch * 0.5);
    double cy = std::cos(yaw * 0.5);
    double sy = std::sin(yaw * 0.5);

    q_[0] = cy * cp * cr + sy * sp * sr;
    q_[1] = cy * cp * sr - sy * sp * cr;
    q_[2] = sy * cp * sr + cy * sp * cr;
    q_[3] = sy * cp * cr - cy * sp * sr;
    normalizeQuaternion(q_);

    bg_[0] = 0.0;
    bg_[1] = 0.0;
    bg_[2] = 0.0;
    resetCovariance();
    return true;
}

bool AttitudeEKF::updateIMU(
    float gxDps, float gyDps, float gzDps,
    float axMg, float ayMg, float azMg,
    float dt
) {
    if (dt <= 0.0f) {
        return false;
    }

    double gyroRad[3] = {
        static_cast<double>(gxDps) * DEG2RAD,
        static_cast<double>(gyDps) * DEG2RAD,
        static_cast<double>(gzDps) * DEG2RAD
    };

    predict(gyroRad, static_cast<double>(dt));

    double axG = static_cast<double>(axMg) / 1000.0;
    double ayG = static_cast<double>(ayMg) / 1000.0;
    double azG = static_cast<double>(azMg) / 1000.0;
    double accNorm = std::sqrt(axG * axG + ayG * ayG + azG * azG);
    last_acc_norm_g_ = static_cast<float>(accNorm);

    bool useAcc = (accNorm >= acc_norm_min_g_ && accNorm <= acc_norm_max_g_ && accNorm > 1e-12);
    last_acc_used_ = useAcc;

    if (useAcc) {
        // 观测量是真实重力方向，body系，归一化。
        double z[3] = {-axG / accNorm, -ayG / accNorm, -azG / accNorm};
        updateAccelGravity(z);
    }

    symmetrizeP();
    return true;
}

void AttitudeEKF::predict(const double gyroRad[3], double dt) {
    if (dt <= 0.0 || dt > 0.2) {
        return;
    }

    double omega[3] = {
        gyroRad[0] - bg_[0],
        gyroRad[1] - bg_[1],
        gyroRad[2] - bg_[2]
    };

    double rv[3] = {omega[0] * dt, omega[1] * dt, omega[2] * dt};
    double dq[4];
    deltaQuatFromRotVec(rv, dq);

    double qNew[4];
    quatMultiply(q_, dq, qNew);
    std::memcpy(q_, qNew, sizeof(qNew));
    normalizeQuaternion(q_);

    double F[ERR_N][ERR_N] {};
    for (int i = 0; i < ERR_N; ++i) {
        F[i][i] = 1.0;
    }

    double W[3][3];
    skew(omega, W);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            F[r][c] += -W[r][c] * dt;
        }
        F[r][r + 3] = -dt;
    }

    double FP[ERR_N][ERR_N] {};
    for (int i = 0; i < ERR_N; ++i) {
        for (int j = 0; j < ERR_N; ++j) {
            for (int k = 0; k < ERR_N; ++k) {
                FP[i][j] += F[i][k] * P_[k][j];
            }
        }
    }

    double newP[ERR_N][ERR_N] {};
    for (int i = 0; i < ERR_N; ++i) {
        for (int j = 0; j < ERR_N; ++j) {
            for (int k = 0; k < ERR_N; ++k) {
                newP[i][j] += FP[i][k] * F[j][k];
            }
        }
    }

    for (int i = 0; i < ERR_N; ++i) {
        for (int j = 0; j < ERR_N; ++j) {
            P_[i][j] = newP[i][j];
        }
    }

    const double qg = gyro_noise_rad_sqrt_hz_ * gyro_noise_rad_sqrt_hz_ * dt;
    const double qbg = gyro_bias_rw_rad_sqrt_s_ * gyro_bias_rw_rad_sqrt_s_ * dt;

    P_[0][0] += qg;
    P_[1][1] += qg;
    P_[2][2] += qg;
    P_[3][3] += qbg;
    P_[4][4] += qbg;
    P_[5][5] += qbg;
}

void AttitudeEKF::updateAccelGravity(const double measuredGravityBody[3]) {
    double h[3];
    gravityBodyFromQuaternion(q_, h);

    double H[3][ERR_N] {};
    const double eps = 1e-5;

    for (int col = 0; col < 3; ++col) {
        double rv[3] = {0.0, 0.0, 0.0};
        rv[col] = eps;
        double dq[4];
        deltaQuatFromRotVec(rv, dq);

        double qPert[4];
        quatMultiply(q_, dq, qPert);
        normalizeQuaternion(qPert);

        double hp[3];
        gravityBodyFromQuaternion(qPert, hp);

        for (int row = 0; row < 3; ++row) {
            H[row][col] = (hp[row] - h[row]) / eps;
        }
    }

    // gyro bias 对当前重力方向观测无直接一阶影响。
    for (int row = 0; row < 3; ++row) {
        for (int col = 3; col < ERR_N; ++col) {
            H[row][col] = 0.0;
        }
    }

    double y[3] = {
        measuredGravityBody[0] - h[0],
        measuredGravityBody[1] - h[1],
        measuredGravityBody[2] - h[2]
    };

    double R[3][3] {};
    const double r = acc_noise_unit_ * acc_noise_unit_;
    R[0][0] = r;
    R[1][1] = r;
    R[2][2] = r;

    double S[3][3] {};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S[i][j] = R[i][j];
            for (int a = 0; a < ERR_N; ++a) {
                for (int b = 0; b < ERR_N; ++b) {
                    S[i][j] += H[i][a] * P_[a][b] * H[j][b];
                }
            }
        }
    }

    double det =
        S[0][0] * (S[1][1] * S[2][2] - S[1][2] * S[2][1]) -
        S[0][1] * (S[1][0] * S[2][2] - S[1][2] * S[2][0]) +
        S[0][2] * (S[1][0] * S[2][1] - S[1][1] * S[2][0]);

    if (std::fabs(det) < 1e-18) {
        return;
    }

    double invS[3][3];
    invS[0][0] =  (S[1][1] * S[2][2] - S[1][2] * S[2][1]) / det;
    invS[0][1] = -(S[0][1] * S[2][2] - S[0][2] * S[2][1]) / det;
    invS[0][2] =  (S[0][1] * S[1][2] - S[0][2] * S[1][1]) / det;
    invS[1][0] = -(S[1][0] * S[2][2] - S[1][2] * S[2][0]) / det;
    invS[1][1] =  (S[0][0] * S[2][2] - S[0][2] * S[2][0]) / det;
    invS[1][2] = -(S[0][0] * S[1][2] - S[0][2] * S[1][0]) / det;
    invS[2][0] =  (S[1][0] * S[2][1] - S[1][1] * S[2][0]) / det;
    invS[2][1] = -(S[0][0] * S[2][1] - S[0][1] * S[2][0]) / det;
    invS[2][2] =  (S[0][0] * S[1][1] - S[0][1] * S[1][0]) / det;

    double PHt[ERR_N][3] {};
    for (int i = 0; i < ERR_N; ++i) {
        for (int row = 0; row < 3; ++row) {
            for (int j = 0; j < ERR_N; ++j) {
                PHt[i][row] += P_[i][j] * H[row][j];
            }
        }
    }

    double K[ERR_N][3] {};
    for (int i = 0; i < ERR_N; ++i) {
        for (int col = 0; col < 3; ++col) {
            for (int k = 0; k < 3; ++k) {
                K[i][col] += PHt[i][k] * invS[k][col];
            }
        }
    }

    double dx[ERR_N] {};
    for (int i = 0; i < ERR_N; ++i) {
        for (int row = 0; row < 3; ++row) {
            dx[i] += K[i][row] * y[row];
        }
    }

    applyErrorState(dx);

    double KH[ERR_N][ERR_N] {};
    for (int i = 0; i < ERR_N; ++i) {
        for (int j = 0; j < ERR_N; ++j) {
            for (int row = 0; row < 3; ++row) {
                KH[i][j] += K[i][row] * H[row][j];
            }
        }
    }

    double IminusKH[ERR_N][ERR_N] {};
    for (int i = 0; i < ERR_N; ++i) {
        for (int j = 0; j < ERR_N; ++j) {
            IminusKH[i][j] = (i == j ? 1.0 : 0.0) - KH[i][j];
        }
    }

    double newP[ERR_N][ERR_N] {};
    for (int i = 0; i < ERR_N; ++i) {
        for (int j = 0; j < ERR_N; ++j) {
            for (int k = 0; k < ERR_N; ++k) {
                newP[i][j] += IminusKH[i][k] * P_[k][j];
            }
        }
    }

    for (int i = 0; i < ERR_N; ++i) {
        for (int j = 0; j < ERR_N; ++j) {
            P_[i][j] = newP[i][j];
        }
    }
}

void AttitudeEKF::applyErrorState(const double dx[ERR_N]) {
    double rv[3] = {dx[0], dx[1], dx[2]};
    double dq[4];
    deltaQuatFromRotVec(rv, dq);

    double qNew[4];
    quatMultiply(q_, dq, qNew);
    std::memcpy(q_, qNew, sizeof(qNew));
    normalizeQuaternion(q_);

    bg_[0] += dx[3];
    bg_[1] += dx[4];
    bg_[2] += dx[5];
}

void AttitudeEKF::symmetrizeP() {
    for (int i = 0; i < ERR_N; ++i) {
        for (int j = i + 1; j < ERR_N; ++j) {
            double v = 0.5 * (P_[i][j] + P_[j][i]);
            P_[i][j] = v;
            P_[j][i] = v;
        }
        if (P_[i][i] < 1e-14) {
            P_[i][i] = 1e-14;
        }
    }
}

void AttitudeEKF::getQuaternion(float q[4]) const {
    q[0] = static_cast<float>(q_[0]);
    q[1] = static_cast<float>(q_[1]);
    q[2] = static_cast<float>(q_[2]);
    q[3] = static_cast<float>(q_[3]);
}

void AttitudeEKF::getEuler(float& rollDeg, float& pitchDeg, float& yawDeg) const {
    double roll = std::atan2(
        2.0 * (q_[0] * q_[1] + q_[2] * q_[3]),
        1.0 - 2.0 * (q_[1] * q_[1] + q_[2] * q_[2])
    );

    double sinPitch = 2.0 * (q_[0] * q_[2] - q_[3] * q_[1]);
    sinPitch = std::max(-1.0, std::min(1.0, sinPitch));
    double pitch = std::asin(sinPitch);

    double yaw = std::atan2(
        2.0 * (q_[1] * q_[2] + q_[0] * q_[3]),
        1.0 - 2.0 * (q_[2] * q_[2] + q_[3] * q_[3])
    );

    rollDeg = static_cast<float>(roll * RAD2DEG);
    pitchDeg = static_cast<float>(pitch * RAD2DEG);
    yawDeg = static_cast<float>(yaw * RAD2DEG);
}

void AttitudeEKF::getGyroBiasDps(float bias[3]) const {
    bias[0] = static_cast<float>(bg_[0] * RAD2DEG);
    bias[1] = static_cast<float>(bg_[1] * RAD2DEG);
    bias[2] = static_cast<float>(bg_[2] * RAD2DEG);
}
