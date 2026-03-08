import os
# 强制让 JAX 不要在启动时抢占所有显存，防止 OOM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# 如果你想用 GPU 0，这里改成 '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0' 

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import argparse
import optax
from functools import partial
import time
from models import DeepOHeat_ST, DeepOHeat_v1
from hvp import hvp_fwdfwd
from train import train_loop, update

from physics_config import LayerProperty, BoundaryProperty, ChipletStack



layers = [
    LayerProperty(name="layer1", k=130.0, thickness=50e-6),
    LayerProperty(name="layer2", k=1.5,   thickness=20e-6),
    LayerProperty(name="layer3", k=5.0,   thickness=40e-6),
    LayerProperty(name="layer4", k=120.0, thickness=60e-6),
    LayerProperty(name="layer5", k=2.0,   thickness=30e-6),
    LayerProperty(name="layer6", k=150.0, thickness=80e-6),
    LayerProperty(name="layer7", k=10.0,  thickness=100e-6),
]

boundary = BoundaryProperty(
    h_top=1.0e4,
    h_bottom=5.0e3,
    T_amb_k=298.15, #环境温度
)

stack = ChipletStack(
    layers=layers,
    boundary=boundary,
    q_ref=2.0e8,
    T_shift_k=293.15, #偏移温度
)

def generate_power_map(n_train=100, n_test=20, nc=101):
    print("--- 正在生成热源数据 ---")
    nx, ny = nc, nc
    nz = max(2, int(round(stack.total_thickness / stack.min_sampling_spacing_m)) + 1)
    dim_branch = nx * ny
    
    # 模拟热源 f (Batch, 101*101)
    # 随机生成一些热源值，模拟原本的数据分布
    key = jax.random.PRNGKey(0)
    fs_train = jax.random.uniform(key, (n_train, dim_branch), minval=0.1, maxval=2.0)
    fs_test = jax.random.uniform(key, (n_test, dim_branch), minval=0.1, maxval=2.0)
    
    # 模拟真实场 
    u_test = jax.random.uniform(key, (n_test, nx, ny, nz, 1))
    
    print(f"功耗数据生成完毕:")
    print(f"fs_train: {fs_train.shape}")
    print(f"fs_test:  {fs_test.shape}")
    print(f"u_test:   {u_test.shape}")
    return fs_train, fs_test, u_test
# print(stack.summary())

# params_star = stack.build_pinn_parameters()
# print("source_star =", params_star["source_star"])

# T_k = np.array([293.15, 303.15, 323.15])
# T_star = stack.normalize_temperature(T_k)
# print("T_star =", T_star)

# T_back = stack.denormalize_temperature(T_star)
# print("T_back =", T_back)
