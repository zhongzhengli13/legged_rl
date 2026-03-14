# try_y 模型效果分析与调试指南

## 一、问题诊断

### 1.1 当前模型表现分析

根据 `experiments/try_y/debug/debug_0.xlsx` 的数据分析，发现以下关键问题：

#### 关键指标异常：

| 指标 | 当前值 | 正常范围 | 问题严重程度 |
|------|--------|----------|--------------|
| **平衡奖励 (balance)** | 0.365 | 应接近 1.0 | ⚠️ 严重 |
| 平衡最小值 | 0.250 | > 0.8 | ⚠️ 严重 |
| 前向速度 (fwd_vel) | 0.860 m/s | 0.3-0.7 m/s | ⚠️ 超出目标 |
| 前向速度最大值 | 3.518 m/s | < 1.0 m/s | ⚠️ 严重超速 |
| 足部滑动 (foot_slip) | -0.085 | 应接近 0 | ⚠️ 中等 |
| 动作平滑度 (act_smo) | -0.001 | 应为正值 | ⚠️ 轻微 |

**核心问题：**
1. **平衡性极差**：平均平衡奖励只有 0.365，说明机器人姿态不稳定，容易摔倒
2. **速度控制失效**：目标速度是 0.3-0.7 m/s，但实际最高达到 3.518 m/s，说明策略过于激进
3. **足部接触不良**：足部滑动值为负，说明接地不稳定

### 1.2 配置文件分析

查看 `experiments/try_y/model/cfg.yaml`，发现以下配置可能导致问题：

```yaml
# 当前配置
lin_vel_x_range: [0.3, 0.7]  # 速度目标
ang_vel_yaw_range: [0, 0]    # 不转向
episode_length_s: 10          # 每回合10秒
num_steps_per_env: 24         # 每次更新24步
```

**潜在问题：**
- 速度目标设置较高（0.3-0.7 m/s），对于训练初期可能过于激进
- 训练步数较少（24步），可能导致策略不够稳定

---

## 二、Debug 文件详解

### 2.1 Debug 文件结构

`debug_0.xlsx` 包含 27 列数据，每行代表一个时间步（共 208 步）：

#### 奖励项（Reward Components）：

| 列索引 | 列名 | 含义 | 正常范围 |
|--------|------|------|----------|
| 0 | balance | 平衡奖励（姿态稳定性） | 0.8-1.0 |
| 1 | fwd_vel | 前向速度奖励 | 根据目标速度 |
| 2 | yaw_rat | 偏航角速度奖励 | 接近0 |
| 3 | lateral_vel | 侧向速度惩罚 | 接近0 |
| 4 | vertical_vel | 垂直速度惩罚 | 接近0 |
| 5 | ang_vel | 角速度惩罚 | 接近0 |
| 6 | twist | 扭转惩罚 | 接近0 |
| 7 | foot_clr | 足部离地高度奖励 | > 0 |
| 8 | foot_supt | 足部支撑奖励 | 0.5-1.0 |
| 9 | foot_heit | 足部高度奖励 | > 0 |
| 10 | leg_width_rew | 腿部宽度奖励 | - |
| 11 | act_const | 动作约束惩罚 | 接近0 |
| 12 | sa_const | 状态-动作约束 | 接近0 |
| 13 | foot_phase | 足部相位奖励 | - |
| 14 | jnt_pos_err | 关节位置误差 | 接近0 |
| 15 | act_smo | 动作平滑度奖励 | > 0 |
| 16 | net_smo | 网络输出平滑度 | > 0 |
| 17 | net_out_val | 网络输出值 | - |
| 18 | foot_slip | 足部滑动惩罚 | 接近0 |
| 19 | foot_vz | 足部垂直速度 | - |
| 20 | foot_acc | 足部加速度 | - |
| 21 | foot_sft | 足部软着陆 | - |
| 22 | jnt_vel | 关节速度惩罚 | 接近0 |
| 23 | feet_py | 足部俯仰角 | - |
| 24 | feet_frc | 足部接触力 | - |
| 25 | joint_tor | 关节力矩惩罚 | 接近0 |
| 26 | pmf | 相位调制因子 | - |

### 2.2 如何读取和分析 Debug 文件

#### 方法1：使用 Python 脚本分析

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取debug文件
df = pd.read_excel('experiments/try_y/debug/debug_0.xlsx')

# 查看基本信息
print(f"总步数: {len(df)}")
print(f"列名: {df.columns.tolist()}")

# 分析关键指标
print("\n=== 关键指标统计 ===")
print(f"平均平衡奖励: {df['balance'].mean():.3f}")
print(f"平衡奖励最小值: {df['balance'].min():.3f}")
print(f"平均前向速度: {df['fwd_vel'].mean():.3f} m/s")
print(f"前向速度最大值: {df['fwd_vel'].max():.3f} m/s")
print(f"平均足部滑动: {df['foot_slip'].mean():.3f}")

# 绘制关键指标曲线
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# 平衡奖励
axes[0, 0].plot(df['balance'])
axes[0, 0].set_title('Balance Reward')
axes[0, 0].axhline(y=0.8, color='r', linestyle='--', label='Target')
axes[0, 0].legend()

# 前向速度
axes[0, 1].plot(df['fwd_vel'])
axes[0, 1].set_title('Forward Velocity')
axes[0, 1].axhline(y=0.5, color='g', linestyle='--', label='Target')
axes[0, 1].legend()

# 足部滑动
axes[1, 0].plot(df['foot_slip'])
axes[1, 0].set_title('Foot Slip')

# 动作平滑度
axes[1, 1].plot(df['act_smo'])
axes[1, 1].set_title('Action Smoothness')

# 足部支撑
axes[2, 0].plot(df['foot_supt'])
axes[2, 0].set_title('Foot Support')

# 关节速度
axes[2, 1].plot(df['jnt_vel'])
axes[2, 1].set_title('Joint Velocity')

plt.tight_layout()
plt.savefig('debug_analysis.png')
print("\n图表已保存到 debug_analysis.png")
```

#### 方法2：使用 Excel 直接查看

1. 打开 `experiments/try_y/debug/debug_0.xlsx`
2. 重点关注以下列：
   - **balance**：如果持续低于 0.8，说明平衡性差
   - **fwd_vel**：如果波动剧烈或超出目标范围，说明速度控制不稳定
   - **foot_slip**：如果绝对值较大，说明足部接触有问题
   - **act_smo**：如果为负值或波动大，说明动作不平滑

### 2.3 Debug 数据的时间关系

- 每行代表一个控制步（control step）
- 控制频率 = 1 / (sim.dt × decimation) = 1 / (0.001 × 10) = 100 Hz
- 208 步 ≈ 2.08 秒的仿真时间
- **注意**：如果机器人在 2 秒内就结束了（done=True），说明很快就摔倒或失败了

---

## 三、修改建议

### 3.1 立即修改项（高优先级）

#### 修改1：降低速度目标（config/loc.py:193）

**问题**：当前速度目标 0.3-0.7 m/s 对于不稳定的策略过高

```python
# 修改前
lin_vel_x_range = [0.3, 0.7]

# 修改后（降低到更保守的范围）
lin_vel_x_range = [0.1, 0.3]  # 先让机器人学会慢慢走
```

**文件位置**：`config/loc.py` 第 193 行

#### 修改2：增加训练步数（config/loc.py:22）

**问题**：24 步太少，策略更新不够充分

```python
# 修改前
num_steps_per_env = 24

# 修改后
num_steps_per_env = 48  # 增加到48步，提供更多经验
```

**文件位置**：`config/loc.py` 第 22 行

#### 修改3：调整 PD 增益（config/loc.py:98-117）

**问题**：当前 PD 参数可能导致动作过于激进

```python
# 建议降低膝关节和髋关节的刚度
stiffness = {
    "hip_yaw": 150,      # 从 200 降低到 150
    "hip_roll": 150,     # 从 200 降低到 150
    "hip_pitch": 150,    # 从 200 降低到 150
    "knee": 250,         # 从 300 降低到 250
    "ankle": 40,         # 保持不变
    "torso": 300,
    "shoulder": 200,
    "elbow": 200,
}

# 增加阻尼以提高稳定性
damping = {
    "hip_yaw": 20.0,     # 从 15.0 增加到 20.0
    "hip_roll": 35.0,    # 从 30.0 增加到 35.0
    "hip_pitch": 20.0,   # 从 15.0 增加到 20.0
    "knee": 30,          # 从 25 增加到 30
    "ankle": 5.0,        # 从 4.0 增加到 5.0
    "torso": 10,
    "shoulder": 5,
    "elbow": 5,
}
```

**文件位置**：`config/loc.py` 第 98-117 行

### 3.2 中期优化项（中优先级）

#### 修改4：启用域随机化（config/loc.py:130-165）

**目的**：提高策略的鲁棒性

```python
# 修改前
randomize_friction = False
randomize_gains = False

# 修改后
randomize_friction = True
friction_range = [0.8, 1.2]  # 增加摩擦力随机化
randomize_gains = True
gains_range = [0.95, 1.05]   # 轻微的增益随机化
```

**文件位置**：`config/loc.py` 第 131、150 行

#### 修改5：调整奖励权重

需要查看奖励函数的权重配置。通常在 `env/tasks/locomotion_task.py` 中定义。

**建议调整方向**：
- **增加平衡奖励权重**：让策略更重视保持平衡
- **增加动作平滑度权重**：减少抖动
- **降低速度奖励权重**：避免过于追求速度而忽视稳定性

### 3.3 长期改进项（低优先级）

#### 修改6：课程学习

```python
# 在 config/loc.py 中启用课程学习
curriculum = True
max_curriculum = 1.0
```

让机器人从简单任务（低速、平地）逐步过渡到复杂任务。

#### 修改7：增加训练迭代次数

```python
# config/loc.py:21
max_iterations = 2000  # 从 1100 增加到 2000
```

---

## 四、调试流程

### 4.1 标准调试步骤

1. **修改配置文件**
   ```bash
   # 编辑配置
   vim config/loc.py
   ```

2. **重新训练模型**
   ```bash
   python train.py --name try_y_v2
   ```

3. **测试新模型**
   ```bash
   python play.py --name try_y_v2 --debug --time 10
   ```

4. **分析 Debug 数据**
   ```bash
   # 查看新的 debug 文件
   python -c "
   import pandas as pd
   df = pd.read_excel('experiments/try_y_v2/debug/debug_0.xlsx')
   print('平均平衡:', df['balance'].mean())
   print('平均速度:', df['fwd_vel'].mean())
   "
   ```

5. **对比前后效果**
   - 比较 `try_y` 和 `try_y_v2` 的 debug 数据
   - 查看平衡奖励是否提升
   - 查看速度是否更稳定

### 4.2 快速验证脚本

创建一个快速分析脚本 `analyze_debug.py`：

```python
#!/usr/bin/env python3
import pandas as pd
import sys

def analyze_debug(exp_name):
    df = pd.read_excel(f'experiments/{exp_name}/debug/debug_0.xlsx')

    print(f"\n{'='*50}")
    print(f"实验: {exp_name}")
    print(f"{'='*50}")

    # 关键指标
    metrics = {
        '总步数': len(df),
        '平均平衡': df['balance'].mean(),
        '平衡最小值': df['balance'].min(),
        '平均速度': df['fwd_vel'].mean(),
        '速度最大值': df['fwd_vel'].max(),
        '平均足部滑动': df['foot_slip'].mean(),
        '平均动作平滑度': df['act_smo'].mean(),
    }

    for key, value in metrics.items():
        print(f"{key:15s}: {value:.3f}")

    # 评分
    score = 0
    if df['balance'].mean() > 0.8:
        score += 30
        print("\n✓ 平衡性良好")
    else:
        print("\n✗ 平衡性差")

    if 0.3 <= df['fwd_vel'].mean() <= 0.7:
        score += 30
        print("✓ 速度控制良好")
    else:
        print("✗ 速度控制不佳")

    if abs(df['foot_slip'].mean()) < 0.05:
        score += 20
        print("✓ 足部接触良好")
    else:
        print("✗ 足部滑动过多")

    if df['act_smo'].mean() > 0:
        score += 20
        print("✓ 动作平滑")
    else:
        print("✗ 动作不平滑")

    print(f"\n总分: {score}/100")
    return score

if __name__ == '__main__':
    if len(sys.argv) > 1:
        analyze_debug(sys.argv[1])
    else:
        print("用法: python analyze_debug.py <实验名称>")
        print("示例: python analyze_debug.py try_y")
```

使用方法：
```bash
chmod +x analyze_debug.py
python analyze_debug.py try_y
python analyze_debug.py try_y_v2
```

---

## 五、常见问题排查

### 问题1：机器人快速摔倒（< 3秒）

**症状**：debug 文件只有几十行数据

**可能原因**：
- 初始姿态不稳定
- PD 增益过高导致震荡
- 奖励函数设置不合理

**解决方案**：
1. 检查 `init_state.reset_joint_pos` 是否合理
2. 降低 PD 刚度（stiffness）
3. 增加平衡奖励权重

### 问题2：速度不受控制

**症状**：fwd_vel 波动剧烈或持续超出目标范围

**可能原因**：
- 速度奖励权重过高
- 动作范围（action_limit）过大
- 学习率过高导致策略不稳定

**解决方案**：
1. 降低速度目标范围
2. 减小 `inc_high_ranges` 和 `inc_low_ranges`
3. 降低学习率：`learning_rate = 5e-4`（从 1e-3）

### 问题3：足部滑动严重

**症状**：foot_slip 绝对值 > 0.1

**可能原因**：
- 地面摩擦力不足
- 足部接触力分布不均
- 步态相位不协调

**解决方案**：
1. 增加地面摩擦力：`static_friction = 1.2`
2. 检查足部奖励权重
3. 启用足部相位奖励

### 问题4：动作抖动

**症状**：act_smo 为负值或波动大

**可能原因**：
- 动作平滑度惩罚权重过低
- PD 阻尼不足
- 观测噪声过大

**解决方案**：
1. 增加动作平滑度奖励权重
2. 增加 PD 阻尼（damping）
3. 降低观测噪声：`dof_vel = 0.05`（从 0.1）

---

## 六、进阶调试技巧

### 6.1 使用 TensorBoard 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir experiments/try_y/log

# 在浏览器打开 http://localhost:6006
```

**关注指标**：
- `rewards/total`：总奖励应该逐渐上升
- `losses/policy_loss`：策略损失应该逐渐下降
- `losses/value_loss`：价值损失应该逐渐下降

### 6.2 对比不同迭代的模型

```bash
# 测试不同迭代的模型
python play.py --name try_y --iter 200 --debug
python play.py --name try_y --iter 400 --debug
python play.py --name try_y --iter 600 --debug
python play.py --name try_y --iter 800 --debug
python play.py --name try_y --iter 1000 --debug

# 对比 debug 数据
python -c "
import pandas as pd
for iter in [200, 400, 600, 800, 1000]:
    df = pd.read_excel(f'experiments/try_y/debug/debug_{iter}.xlsx')
    print(f'Iter {iter}: balance={df[\"balance\"].mean():.3f}, vel={df[\"fwd_vel\"].mean():.3f}')
"
```

### 6.3 录制视频分析

```bash
# 录制视频
python play.py --name try_y --video --time 10

# 视频保存在 experiments/try_y/debug/try_y.mp4
```

观察视频时注意：
- 机器人是否保持直立
- 步态是否协调
- 是否有明显的抖动或不自然的动作

---

## 七、推荐的修改顺序

### 第一轮修改（最小改动）

1. 降低速度目标：`lin_vel_x_range = [0.1, 0.3]`
2. 增加训练步数：`num_steps_per_env = 48`
3. 测试并分析 debug 数据

### 第二轮修改（如果第一轮效果不佳）

4. 调整 PD 增益（降低刚度，增加阻尼）
5. 增加训练迭代次数：`max_iterations = 2000`
6. 测试并对比

### 第三轮修改（精细调优）

7. 启用域随机化
8. 调整奖励权重
9. 启用课程学习

---

## 八、总结

### 当前问题根源

try_y 模型效果不好的主要原因是：
1. **平衡性差**：平均平衡奖励只有 0.365（正常应 > 0.8）
2. **速度失控**：最高速度达到 3.5 m/s，远超目标范围
3. **训练不充分**：可能需要更多训练迭代和更保守的初始目标

### 核心修改建议

**立即修改**：
- 降低速度目标到 0.1-0.3 m/s
- 增加训练步数到 48
- 调整 PD 增益（降低刚度，增加阻尼）

**验证方法**：
- 使用 `analyze_debug.py` 脚本快速评估
- 对比修改前后的 balance、fwd_vel、foot_slip 指标
- 目标：balance > 0.8, fwd_vel 在目标范围内，foot_slip < 0.05

### 学习资源

- Debug 文件位置：`experiments/<实验名>/debug/debug_0.xlsx`
- 配置文件：`config/loc.py`
- 任务定义：`env/tasks/locomotion_task.py`
- 训练日志：`experiments/<实验名>/log/`（使用 TensorBoard 查看）

---

**文档创建时间**：2026-03-13
**适用版本**：legged_rl 项目
**作者**：AI Assistant
