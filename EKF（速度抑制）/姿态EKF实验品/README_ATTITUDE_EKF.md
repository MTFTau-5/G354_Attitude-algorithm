# G354 Attitude EKF 版说明

这一版把姿态解算从 `MahonyAHRS` 替换成 `AttitudeEKF`。

## 编译

```bash
g++ -std=c++17 -Wall -Wextra -O2 main.cpp attitude_ekf.cpp -o g354_attitude_ekf
```

## 运行

```bash
./g354_attitude_ekf /dev/ttyUSB0 460800
```

## 姿态 EKF 状态

名义状态：

```text
q  = [qw, qx, qy, qz]
bg = [bgx, bgy, bgz]  // residual gyro bias, rad/s
```

误差状态：

```text
δx = [δθx, δθy, δθz, δbgx, δbgy, δbgz]
```

## 预测

使用 G354 陀螺积分姿态：

```text
omega = gyro - bg
q[k+1] = q[k] ⊗ exp(omega * dt)
```

## 加速度观测

G354 原始加速度是比力 specific force：

```text
f = a - g
```

静止或低动态时：

```text
raw_acc ≈ -gravity_body
```

所以姿态 EKF 使用：

```text
measured_gravity_body = -normalize(raw_acc)
```

来修正 roll / pitch。

## 注意

- 6轴 IMU 仍然无法绝对约束 yaw。
- yaw 只能靠 gz 积分和零偏校准维持，长期仍会漂。
- 当前版本只是把姿态部分从 Mahony 换成 EKF，平面位置仍由原来的 PlanarEKF 估计。
- 输出中的 `attBg[dps]` 是姿态 EKF 自己估计的 residual gyro bias。

## 主要新增参数

在 `main.cpp` 顶部：

```cpp
static constexpr float ATT_EKF_ACC_NOISE_UNIT = 0.035f;
static constexpr float ATT_EKF_GYRO_NOISE_DPS_SQRT_HZ = 0.10f;
static constexpr float ATT_EKF_BIAS_RW_DPS_SQRT_S = 0.005f;
```

含义：

- `ATT_EKF_ACC_NOISE_UNIT`：归一化重力方向观测噪声，越大越不信加速度修正。
- `ATT_EKF_GYRO_NOISE_DPS_SQRT_HZ`：陀螺预测噪声，越大姿态预测协方差增长越快。
- `ATT_EKF_BIAS_RW_DPS_SQRT_S`：残余陀螺零偏随机游走，越大 bias 变化越快。
