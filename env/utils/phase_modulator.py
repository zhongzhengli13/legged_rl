# from math import sin, pi, tau
# import numpy as np
# from isaacgym.torch_utils import to_torch
# import torch


# class PhaseModulator:
#     def __init__(self, time_step, num_envs, num_legs, device):
#         self.num_legs = num_legs
#         self._phase = torch.zeros(
#             num_envs, num_legs, dtype=torch.float, device=device, requires_grad=False)
#         self._frequency = torch.ones(
#             num_envs, num_legs, dtype=torch.float, device=device, requires_grad=False) * 0.5
#         self._time_step = time_step
#         self.device = device
#         self.reset(env_ids=torch.arange(num_envs))
#     # 原始
#     # def reset(self, convert_phi=pi, env_ids=None, render=False):
#     #     if render:
#     #         init_phase = to_torch([[0. for _ in range(self.num_legs)]], device=self.device)
#     #     else:
#     #         init_phase = to_torch([[np.random.uniform(0, 2 * pi) for _ in range(self.num_legs)]], device=self.device)
#     #     self._phase[env_ids] = init_phase % tau
#     #     self._frequency[env_ids] = torch.ones(len(env_ids), self.num_legs, dtype=torch.float, device=self.device, requires_grad=False) * 0.5

#     # 修改 ggg
#     # 修改后 phase_modulator.py → reset() 函数
#     def reset(self, convert_phi=pi, env_ids=None, render=False):
#         if render:
#             # render 模式：左腿从 0 开始，右腿从 π 开始（标准对称步态）
#             init_phase = to_torch(
#                 [[0., pi]],       # 原来是 [[0., 0.]]
#                 device=self.device
#             )
#         else:
#             # 训练模式：左腿随机，右腿 = 左腿 + π（强制保证反相）
#             left_phases = np.random.uniform(0, 2 * pi, size=len(env_ids))
#             right_phases = left_phases + pi   # ← 关键：右腿强制比左腿晚半个周期

#             # shape: [len(env_ids), 2]，每行是 [左腿相位, 右腿相位]
#             init_phase_np = np.stack([left_phases, right_phases], axis=1)
#             init_phase = torch.tensor(
#                 init_phase_np, dtype=torch.float, device=self.device)

#             self._phase[env_ids] = init_phase % tau
#             self._frequency[env_ids] = torch.ones(
#                 len(env_ids), self.num_legs, dtype=torch.float,
#                 device=self.device, requires_grad=False
#             ) * 0.5
#             return   # ← 训练模式在这里提前返回，不走下面的赋值

#         # render 模式才走到这里（init_phase 是 shape [1, 2] 的张量）
#         self._phase[env_ids] = init_phase % tau
#         self._frequency[env_ids] = torch.ones(
#             len(env_ids), self.num_legs, dtype=torch.float,
#             device=self.device, requires_grad=False
#         ) * 0.5

#     def compute(self, frequency):
#         self._frequency = frequency
#         self._phase = (self._phase + tau * frequency * self._time_step) % tau
#         return self._phase

#     @property
#     def frequency(self):
#         return self._frequency

#     @property
#     def phase(self):
#         return self._phase


from math import sin, pi, tau
import numpy as np
from isaacgym.torch_utils import to_torch
import torch


class PhaseModulator:
    def __init__(self, time_step, num_envs, num_legs, device):
        self.num_legs = num_legs
        self._phase = torch.zeros(
            num_envs, num_legs, dtype=torch.float, device=device, requires_grad=False)
        self._frequency = torch.ones(
            num_envs, num_legs, dtype=torch.float, device=device, requires_grad=False) * 0.5
        self._time_step = time_step
        self.device = device
        self.reset(env_ids=torch.arange(num_envs))

    def reset(self, convert_phi=pi, env_ids=None, render=False):
        n = len(env_ids)

        if render:
            # render / 推理模式：左腿从 0 开始，右腿从 π 开始
            # 原代码是 [[0., 0.]]，两腿同相，这里改成标准对称初始化
            left_phases = np.zeros(n)
            right_phases = np.full(n, pi)
        else:
            # 训练模式：左腿随机，右腿 = 左腿 + π
            # 原代码是两腿各自独立随机，不保证反相
            left_phases = np.random.uniform(0, 2 * pi, size=n)
            right_phases = left_phases + pi   # 强制相差半个周期

        init_phase_np = np.stack(
            [left_phases, right_phases], axis=1)  # shape: [n, 2]
        init_phase = torch.tensor(
            init_phase_np, dtype=torch.float, device=self.device)

        self._phase[env_ids] = init_phase % tau
        self._frequency[env_ids] = torch.ones(
            n, self.num_legs, dtype=torch.float,
            device=self.device, requires_grad=False
        ) * 0.5

    def compute(self, frequency):
        self._frequency = frequency
        self._phase = (self._phase + tau * frequency * self._time_step) % tau
        return self._phase

    @property
    def frequency(self):
        return self._frequency

    @property
    def phase(self):
        return self._phase
