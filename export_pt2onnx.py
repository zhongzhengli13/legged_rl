# -*- coding: utf-8 -*-
import argparse
import isaacgym
import numpy as np
import os
from os.path import exists, join
import torch.nn as nn
from env.utils.helpers import class_to_dict

from model import load_actor
from env.utils import get_args
import importlib
from utils.yaml import ParamsProcess
import onnxruntime as ort
import torch

args = get_args()
exp_dir = join('experiments', args.name)
model_dir = join(exp_dir, 'model')
deploy_dir = join(exp_dir, 'deploy')
os.makedirs(deploy_dir, exist_ok=True)

paramsProcess = ParamsProcess()
params = paramsProcess.read_param(join(model_dir, 'cfg.yaml'))
cfg = getattr(importlib.import_module('.'.join(['config', params['env']['cfg']])), 'H1Config')
cfg = paramsProcess.dict2class(cfg, params)


def convert(name: str, model: nn.Module, input: np.ndarray):
    print(f'\n******************************** {name} ********************************************\n')
    deploy_path = join(deploy_dir, f'{name}.onnx')
    torch.onnx.export(model, torch.from_numpy(input), deploy_path, verbose=False, opset_version=12, input_names=['input'], output_names=['output'])
    print('Pytorch')
    print(model(torch.from_numpy(input)).detach().cpu().numpy())
    ort_session = ort.InferenceSession(deploy_path)
    print('Onnx')
    print(ort_session.run(None, {'input': input})[0])
    gap = model(torch.from_numpy(input)).detach().cpu().numpy() - ort_session.run(None, {'input': input})[0]
    print('Gap')
    print(gap)


actor_policy = load_actor(class_to_dict(cfg.policy), deploy=True).eval()
policy_path = join(model_dir, 'all/policy_2000.pt')
# policy_path = join(model_dir, 'policy.pt')
assert exists(policy_path), policy_path
saved_model = torch.load(policy_path, map_location='cpu')
actor_policy.load_state_dict(saved_model['actor'], strict=False)

for i in range(2):
    input = torch.rand([1, cfg.policy.num_observations]).cpu().numpy()
    convert('actor', actor_policy, input)
input = torch.ones([1, cfg.policy.num_observations]).cpu().numpy()
convert('actor', actor_policy, input)


