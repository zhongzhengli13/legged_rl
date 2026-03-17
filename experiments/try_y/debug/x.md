# 双足机器人直线行走优化分析报告

> 数据来源：`debug_0.xlsx` | 总时间步：1999 步 | 分析日期：2026-03-16

---

## 一、核心问题总览

| 问题 | 严重程度 | 关键数据 |
|------|---------|---------|
| 侧向漂移严重 | ★★★★★ | y方向漂移 -3.15m（同期x仅前进7.35m），漂移率 vy/vx≈78% |
| 足端接触力异常（代码BUG） | ★★★★★ | feet_frc 惩罚 = -0.819，是所有奖励项中最大负值 |
| 动作幅度过大 | ★★★☆☆ | act_const 惩罚 = -0.157，膝关节动作均值 0.39 rad |
| 身体扭转/晃动 | ★★★☆☆ | twist 惩罚 = -0.124，roll角速度 std=0.358 rad/s |
| 步宽偏大 | ★★★☆☆ | 实际步宽 0.378m vs 期望 0.25m |
| 步态相位不标准 | ★★☆☆☆ | 左右脚相位差 2.78 rad vs 理想 π≈3.14 rad |
| 着地冲击大 | ★★☆☆☆ | 足力峰值 4499N（≈459kg等效力），foot_sft = -0.113 |

---

## 二、逐项深度分析

### 2.1 侧向漂移（最严重问题）

这是当前策略最大的缺陷——机器人根本没有走直线。

**数据证据：**

```
整体 episode:
  x 位移: +7.35m    y 位移: -3.15m
  → 侧移量 = x方向位移的 43%，严重偏航

直线行走段（fwd_vel>0.1, yaw_rate=0, 共200步）:
  实际 vx 均值: 0.318 m/s
  实际 vy 均值: 0.230 m/s   ← 命令是0，但实际侧向速度很大
  侧向漂移率 vy/vx: 0.784   ← 几乎横着走！

侧向速度分布:
  10th percentile: -0.489 m/s
  90th percentile: +0.244 m/s
  → 方向不稳定，时左时右大幅摆动
```

**根因分析：**

1. **lateral_vel_rew 权重不足**：系数仅为 2，而 fwd_vel 系数为 5.5。前进速度奖励远强于侧向约束，策略发现"即使侧着走，只要vx大，总奖励就高"。

2. **左右髋关节不对称**：
   ```
   left_hip_yaw  均值: 0.056 rad
   right_hip_yaw 均值: 0.013 rad   ← 差4倍！
   left_hip_roll  均值: -0.062 rad
   right_hip_roll 均值: -0.027 rad  ← 差2倍！
   ```
   左髋yaw偏转远大于右髋，导致机器人整体偏向一侧。

3. **yaw命令采样比例不合理**：
   ```
   yaw_rate=0.0: 1098步 (55%)
   yaw_rate=0.3: 901步  (45%)
   ```
   将近一半时间在命令转弯，策略大量学习了"转弯模式"，在直线段残留了转弯习惯。

4. **缺少全局位置约束**：当前奖励函数只约束瞬时侧向速度 vy，没有约束累积y偏移。策略可以每步偏一点vy，只要瞬时vy不太大就不受重罚，但累积起来就严重跑偏。

---

### 2.2 足端接触力异常（代码BUG）

`feet_frc` 是整个奖励中**最大的负值项**（-0.819），但它实际上包含一个代码重复BUG。

**BUG位置**：`locomotion_task.py` 第 1092-1109 行

```python
# 第2条惩罚（line 1092-1095）
feet_contact_frc_rew += -torch.sum(
    (20.0 - self.env.foot_frc).clip(min=0.0) * self.foot_support_mask,
    dim=1, keepdim=True
)

# 第3条惩罚（line 1106-1109）—— 与上面完全相同！
feet_contact_frc_rew += -torch.sum(
    (20.0 - self.env.foot_frc).clip(min=0.0) * self.foot_support_mask,
    dim=1, keepdim=True
)
```

**问题**：第 3 条注释说要修复"静止模式惩罚"，但代码实际上是第 2 条的完全复制粘贴。导致：
- 支撑相"力不够20N"的惩罚被施加了 **2倍**
- 原本想区分"运动模式"和"静止模式"的逻辑完全丢失
- 该项成为最大负奖励来源，扭曲了整个奖励信号

**足力数据异常：**

```
左脚力: 均值 271N, 最大 4499N, std 320N
右脚力: 均值 239N, 最大 3521N, std 291N
双脚同时支撑比例: 11.9%   ← 过低，说明单腿支撑时间过长
左脚支撑时间比: 56.5%
右脚支撑时间比: 55.3%
```

力的峰值达到4499N（约460kg等效力），说明着地瞬间有巨大冲击，这不是正常的行走模式。

---

### 2.3 动作幅度与能量消耗

**关节动作(action)分析：**

| 关节 | 动作均值(|action|) | 网络输出均值(|net_out|) | 角速度(rad/s) | 力矩峰值(N·m) |
|------|-------------------|----------------------|--------------|--------------|
| left_hip_pitch | 0.221 | 0.395 | 1.149 | 93.7 |
| left_knee | 0.387 | 0.466 | 1.402 | **220.6** |
| left_ankle | 0.256 | **0.592** | 0.865 | 40.0 |
| right_hip_pitch | 0.146 | 0.388 | 0.778 | 95.7 |
| right_knee | 0.373 | 0.502 | 1.480 | **186.9** |
| right_ankle | 0.328 | 0.464 | 0.938 | 33.3 |

**问题**：
- 膝关节力矩峰值极高（220N·m），说明关节被推到了极限
- 左右不对称明显：left_hip_pitch 动作 0.221 vs right 0.146
- 踝关节网络输出最大（0.592），说明策略过度依赖踝关节调节

---

### 2.4 身体姿态晃动

```
欧拉角 roll:  均值 0.028 rad (1.6°),  std 0.036 rad (2.1°)
欧拉角 pitch: 均值 -0.011 rad (0.6°), std 0.033 rad (1.9°)
欧拉角 yaw:   均值 2.418 rad (138.5°), std 0.073 rad (4.2°)

角速度 roll:  std = 0.358 rad/s  ← 最大的晃动源
角速度 pitch: std = 0.181 rad/s
角速度 yaw:   std = 0.114 rad/s
```

- yaw均值 2.418 rad（≈138.5°）说明机器人整体朝向偏离了初始方向约138°，即严重跑偏了方向
- roll 角速度波动最大（std=0.358），说明身体左右摇摆严重
- 这和步宽偏大（0.378m vs 0.25m）是相关的：步宽越大，重心转移越大，晃动越严重

---

### 2.5 步态相位分析

```
左右脚相位差均值: 2.78 rad
理想交替步态相位差: π ≈ 3.14 rad
偏离: 0.37 rad (≈21°)

左脚相位 std: 1.29
右脚相位 std: 2.31  ← 右脚相位变化远大于左脚
```

右脚相位的标准差（2.31）远大于左脚（1.29），说明右脚的步态节奏不稳定。这和前面观察到的左右不对称一致——策略对左右腿的控制方式不同，导致步态不协调。

---

## 三、奖励项权重分析（当前 vs 建议）

```
当前奖励字典 rew_dict:
  fwd_vel     = forward_vel_rew    * 5.5      ← 前进速度权重最高
  yaw_rat     = yaw_rate_rew       * 2
  lateral_vel = lateral_vel_rew    * 2        ← 侧向约束弱
  twist       = twist_rew          * 2.5
  ang_vel     = ang_vel_rew        * 0.8
  balance     = balance_rew        * 0.5
  vertical_vel= vertical_vel_rew   * 0.5
  foot_clr    = foot_clear_rew     * balance * 5
  foot_heit   = foot_height_rew    * balance * 0.8
  foot_supt   = foot_support_rew   * balance * 0.7
  leg_width   = leg_width_rew      * balance * 1.2
  act_const   = action_constraint  * balance * 0.4
  foot_slip   = foot_slip_rew      * balance * 1.2
  foot_sft    = foot_soft_rew      * balance * 1.0
  foot_phase  = foot_phase_rew     * balance * 0.5
  jnt_pos_err = joint_pos_error    * balance * 0.3
  sa_const    = sa_constraint      * balance * 0.2
  act_smo     = action_smooth_rew  * balance * 0.05
  jnt_vel     = joint_velocity     * balance * 0.01   ← 极小
  feet_frc    = feet_contact_frc   * 0.003
  joint_tor   = joint_tor_rew      * 0.001            ← 极小
  net_smo     = net_out_smooth     * balance * 0.00002 ← 几乎无效
  net_out_val = net_out_val_rew    * balance * 0.00001 ← 几乎无效
  pmf         = pmf_rew            * balance * 0.03
  feet_py     = foot_py_rew        * balance * 0.5
  foot_vz     = foot_vz_rew        * balance * 0.3
  foot_acc    = foot_acc_rew       * balance * 0.05
```

**问题诊断**：前进速度（5.5）远大于侧向约束（2），策略自然会选择"先跑快、不管方向"。同时 jnt_vel(0.01)、joint_tor(0.001)、net_smo(0.00002) 几乎为零，对运动质量没有约束力。

---

## 四、具体修改建议

### 修改 1：修复 feet_frc 代码重复BUG（优先级：P0）

**文件**：`locomotion_task.py`，第 1096-1109 行

将第 3 条惩罚修改为原本的设计意图——静止模式下约束过大的力：

```python
# 第2条：支撑相力不够 → 惩罚（保留）
feet_contact_frc_rew += -torch.sum(
    (20.0 - self.env.foot_frc).clip(min=0.0) * self.foot_support_mask,
    dim=1, keepdim=True
)

# 第3条：修复为静止模式惩罚 —— 静止时力不应太大
feet_contact_frc_rew += -torch.sum(
    (self.env.foot_frc - 350.0).clip(min=0.0),
    dim=1, keepdim=True
) * torch.logical_not(self.static_flag)
```

**预期效果**：feet_frc 惩罚将大幅下降，奖励信号恢复正常。

---

### 修改 2：增强侧向约束（优先级：P0）

**文件**：`locomotion_task.py`，奖励字典部分

```python
# 当前
lateral_vel=lateral_vel_rew * 2,

# 建议修改为
lateral_vel=lateral_vel_rew * 4,     # 翻倍，让侧向约束与前进速度更均衡
```

同时强化 `lateral_vel_rew` 的指数惩罚系数（第 904-906 行）：

```python
# 当前: 20 / lin_vel_x_norm → k ∈ [3, 15]
lateral_vel_rew = torch.exp(
    -torch.clip(20 / lin_vel_x_norm, min=3.0, max=15.0)  # ← 已改为20，OK
    * torch.norm(self.env.base_lin_vel[:, [1]], dim=1, keepdim=True) ** 2
)
# 线性惩罚从 2.0 增强到 3.0
lateral_vel_rew -= 3.0 * torch.abs(self.env.base_lin_vel[:, [1]])
```

**预期效果**：侧向速度受到更强约束，减少漂移。

---

### 修改 3：调整命令采样（优先级：P1）

如果目标是"走直线"，应在 `_resample_commands` 中调整 yaw_rate 的采样分布：

```python
# 建议：增加 yaw_rate=0 的采样比例
# 当前: yaw_rate 在 [0, 0.3] 均匀采样，导致45%时间在转弯
# 修改: 70%概率 yaw_rate=0，30%概率 yaw_rate ∈ [0, 0.3]
```

或者在 command 配置文件中将 `yaw_rate` 的范围缩小到 `[0, 0.15]`。

**预期效果**：策略将更多地学习直线行走，减少转弯习惯残留。

---

### 修改 4：增加步宽约束（优先级：P1）

```python
# 当前
leg_width_rew=leg_width_rew * balance_rew * 1.2,

# 建议修改为
leg_width_rew=leg_width_rew * balance_rew * 2.0,
```

**预期效果**：步宽从 0.378m 向期望值 0.25m 收敛，减少身体左右晃动。

---

### 修改 5：增加运动平滑性约束（优先级：P2 — 提升优雅度）

```python
# 当前权重太小，对优雅度几乎没有约束力
jnt_vel   = joint_velocity_rew * balance_rew * 0.01,   # → 改为 0.05
joint_tor = joint_tor_rew * 0.001,                      # → 改为 0.01
act_smo   = action_smooth_rew * balance_rew * 0.05,     # → 改为 0.15
net_smo   = net_out_smooth_rew * balance_rew * 0.00002, # → 改为 0.0005
```

**原理**：
- `jnt_vel`（关节角速度惩罚）：限制关节转速，让动作不那么急躁
- `joint_tor`（力矩惩罚）：减小峰值力矩（当前220N·m太大），让出力更均匀
- `act_smo`（动作平滑）：减小帧间动作跳变，让运动连贯
- `net_smo`（网络输出平滑）：让策略输出更稳定

**预期效果**：运动更柔和、优雅，减少"机械抽搐"感。

---

### 修改 6：增强步态对称性（优先级：P2）

```python
# 当前
foot_phase=foot_phase_rew * balance_rew * 0.5,

# 建议修改为
foot_phase=foot_phase_rew * balance_rew * 1.5,
```

**预期效果**：左右脚相位差向 π 收敛（从 2.78 → 3.14），步态更对称协调。

---

### 修改 7：增加着地柔和度约束（优先级：P2）

```python
# 当前
foot_sft=foot_soft_rew * 1 * balance_rew,

# 建议修改为
foot_sft=foot_soft_rew * 2.0 * balance_rew,
```

**预期效果**：减小着地冲击力（从峰值4499N降低），让落脚更轻柔。

---

## 五、修改优先级路线图

```
阶段一（立即修复 — 解决根本问题）：
  ┌─────────────────────────────────────────────────────┐
  │ ① 修复 feet_frc 重复BUG（P0）                       │
  │ ② 增强 lateral_vel 权重 2→4（P0）                   │
  │ ③ lateral_vel 线性惩罚 2.0→3.0（P0）                │
  └─────────────────────────────────────────────────────┘
                          ↓
阶段二（训练调优 — 改善行走质量）：
  ┌─────────────────────────────────────────────────────┐
  │ ④ 调整 yaw_rate 命令采样（P1）                      │
  │ ⑤ 增大 leg_width_rew 权重 1.2→2.0（P1）            │
  └─────────────────────────────────────────────────────┘
                          ↓
阶段三（精细调整 — 提升优雅度）：
  ┌─────────────────────────────────────────────────────┐
  │ ⑥ 增大 jnt_vel / joint_tor / act_smo / net_smo（P2）│
  │ ⑦ 增大 foot_phase 权重 0.5→1.5（P2）               │
  │ ⑧ 增大 foot_sft 权重 1.0→2.0（P2）                 │
  └─────────────────────────────────────────────────────┘
```

**重要**：每个阶段修改后都应训练验证，不要一次改太多。建议先只做阶段一的修改，训练后看直线行走是否显著改善，再进入阶段二。

---

## 六、修改前后预期对比

| 指标 | 当前值 | 目标值 | 说明 |
|------|--------|--------|------|
| 侧向漂移 vy/vx | 0.78 | < 0.10 | 直线行走段 |
| y方向总漂移 | -3.15m | < 0.5m | 整个episode |
| feet_frc 惩罚 | -0.819 | > -0.3 | 修复BUG后 |
| 步宽 | 0.378m | ~0.25m | 接近期望值 |
| 步态相位差 | 2.78 rad | ~3.14 rad | 对称交替步态 |
| 足力峰值 | 4499N | < 1500N | 柔和着地 |
| 膝关节力矩峰值 | 220 N·m | < 100 N·m | 关节保护 |
| roll 角速度 std | 0.358 | < 0.15 | 减少晃动 |

---

## 七、附录：奖励项均值排名

```
排名（从最差到最好）：

feet_frc        -0.819   ← 最大负值（含BUG）
act_const       -0.157
twist           -0.124
leg_width_rew   -0.115
foot_sft        -0.113
jnt_vel         -0.068
foot_vz         -0.064
sa_const        -0.061
jnt_pos_err     -0.058
foot_slip       -0.037
pmf             -0.031
foot_phase      -0.029
feet_py         -0.028
act_smo         -0.015
foot_acc        -0.013
net_smo         -0.001
net_out_val     -0.000
joint_tor        0.000
vertical_vel     0.225
foot_clr         0.254
lateral_vel      0.275
balance          0.415
foot_supt        0.501
ang_vel          0.553
foot_heit        0.966
yaw_rat          1.237
fwd_vel          1.299   ← 最大正值
```
