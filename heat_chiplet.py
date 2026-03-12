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
    LayerProperty(name="layer0", k=130.0, thickness=1000e-6),
    LayerProperty(name="layer1", k=1.5,   thickness=50e-6),
    LayerProperty(name="layer2", k=5.0,   thickness=100e-6),
    LayerProperty(name="layer3", k=120.0, thickness=50e-6),
    LayerProperty(name="layer4", k=2.0,   thickness=200e-6),
    LayerProperty(name="layer5", k=150.0, thickness=100e-6),
    LayerProperty(name="layer6", k=10.0,  thickness=1000e-6),
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
    x_size_m=1.0e-2,
    y_size_m=1.0e-2,
    T_shift_k=293.15, #偏移温度
    power_sample_min=0.1,
    power_sample_max=2.0,
)

def generate_power_map(n_train=100, n_test=20, nc=101):
    print("--- 正在生成热源数据 ---")
    nx, ny = nc, nc
    nz = stack.nz
    dim_branch = nx * ny
    
    # 模拟热源 f (Batch, 101*101)
    # 随机生成一些热源值，模拟原本的数据分布
    key = jax.random.PRNGKey(0)
    key_train, key_test, key_u = jax.random.split(key, 3)
    fs_train = jax.random.uniform(
        key_train,
        (n_train, dim_branch),
        minval=stack.power_sample_min,
        maxval=stack.power_sample_max,
    )
    fs_test = jax.random.uniform(
        key_test,
        (n_test, dim_branch),
        minval=stack.power_sample_min,
        maxval=stack.power_sample_max,
    )
    
    # 模拟真实场 
    u_test = jax.random.uniform(key_u, (n_test, nx, ny, nz, 1))
    
    print(f"功耗数据生成完毕:")
    print(f"fs_train: {fs_train.shape}")
    print(f"fs_test:  {fs_test.shape}")
    print(f"u_test:   {u_test.shape}")
    return fs_train, fs_test, u_test

@jax.jit
def create_mesh(xi_batch, yi_batch, zi_batch):
    return jnp.meshgrid(xi_batch.ravel(), yi_batch.ravel(), zi_batch.ravel(), indexing='ij')

@eqx.filter_jit
def apply_model_deepoheat_st(model, xc, yc, zc, fc, lam_b=1.):

    def PDE_loss(model, x, y, z, f):
        # compute u
        u = model(((x, y, z), f))
    
        # tangent vector dx/dx dy/dy dz/dz
        v_x = jnp.ones(x.shape)
        v_y = jnp.ones(y.shape)
        v_z = jnp.ones(z.shape)

        # 1st, 2nd derivatives of u
        ux, uxx = hvp_fwdfwd(lambda x: model(((x, y, z), f)), (x,), (v_x,), True)
        uy, uyy = hvp_fwdfwd(lambda y: model(((x, y, z), f)), (y,), (v_y,), True)
        uz, uzz = hvp_fwdfwd(lambda z: model(((x, y, z), f)), (z,), (v_z,), True)
        
        # PDE residual (7-layer stack)
        laplacian = (uxx + uyy + uzz)
        nz = laplacian.shape[3]

        # 6 layer interfaces in normalized z* from physics_config
        z_if_star = stack.z_interfaces_norm[1:-1]
        i01, i12, i23, i34, i45, i56 = [int(round(v * (nz - 1))) for v in z_if_star]

        # per-layer interior slices
        lap_l0 = laplacian[:, :, :, 0:i01, :]
        lap_l1 = laplacian[:, :, :, i01 + 1:i12, :]
        lap_l2 = laplacian[:, :, :, i12 + 1:i23, :]
        lap_l3 = laplacian[:, :, :, i23 + 1:i34, :]
        lap_l4 = laplacian[:, :, :, i34 + 1:i45, :]
        lap_l5 = laplacian[:, :, :, i45 + 1:i56, :]
        lap_l6 = laplacian[:, :, :, i56 + 1:nz, :]

        # interface point slices
        lap_if01 = laplacian[:, :, :, i01:i01 + 1, :]
        lap_if12 = laplacian[:, :, :, i12:i12 + 1, :]
        lap_if23 = laplacian[:, :, :, i23:i23 + 1, :]
        lap_if34 = laplacian[:, :, :, i34:i34 + 1, :]
        lap_if45 = laplacian[:, :, :, i45:i45 + 1, :]
        lap_if56 = laplacian[:, :, :, i56:i56 + 1, :]

        # per-layer normalized conductivity from physics_config
        layer_props = stack.normalized_layer_properties()
        boundary_props = stack.normalized_boundary_properties()
        k_layers = jnp.asarray(
            [p["k_star"] for p in layer_props],
            dtype=laplacian.dtype,
        )
        T_amb_star = jnp.asarray(boundary_props["T_amb_star"], dtype=laplacian.dtype)
        h_top_star = jnp.asarray(boundary_props["h_top_star"], dtype=laplacian.dtype)
        h_bottom_star = jnp.asarray(boundary_props["h_bottom_star"], dtype=laplacian.dtype)

        # harmonic mean conductivity on interfaces
        k_if01 = 2.0 * k_layers[0] * k_layers[1] / (k_layers[0] + k_layers[1])
        k_if12 = 2.0 * k_layers[1] * k_layers[2] / (k_layers[1] + k_layers[2])
        k_if23 = 2.0 * k_layers[2] * k_layers[3] / (k_layers[2] + k_layers[3])
        k_if34 = 2.0 * k_layers[3] * k_layers[4] / (k_layers[3] + k_layers[4])
        k_if45 = 2.0 * k_layers[4] * k_layers[5] / (k_layers[4] + k_layers[5])
        k_if56 = 2.0 * k_layers[5] * k_layers[6] / (k_layers[5] + k_layers[6])

        # power map expanded to match (batch, nx, ny, 1, 1)
        f_xy = f.reshape(f.shape[0], x.shape[0], y.shape[0], 1, 1)

        pde_res = jnp.concatenate(
            [
                k_layers[6] * lap_l6,
                k_if56 * lap_if56,
                k_layers[5] * lap_l5,
                k_if45 * lap_if45 + f_xy,
                k_layers[4] * lap_l4 + 2*f_xy, #chiplet
                k_if34 * lap_if34 + f_xy,
                k_layers[3] * lap_l3,
                k_if23 * lap_if23,
                k_layers[2] * lap_l2,
                k_if12 * lap_if12,
                k_layers[1] * lap_l1,
                k_if01 * lap_if01,
                k_layers[0] * lap_l0,
            ],
            axis=3,
        )
        pde_res = jnp.mean(pde_res ** 2)

        # top surface
        bc_top = jnp.mean(
            #(-k_layers[-1] * uz[:, :, :, -1, :] - h_top_star * (u[:, :, :, -1, :] - T_amb_star)) ** 2
            (u[:, :, :, -1, :] - T_amb_star + k_layers[-1]/h_top_star * uz[:, :, :, -1, :]) ** 2
        )
        # bottom surface
        bc_bottom = jnp.mean(
            #(k_layers[0] * uz[:, :, :, 0, :] - h_bottom_star * (u[:, :, :, 0, :] - T_amb_star)) ** 2
            (u[:, :, :, 0, :] - T_amb_star - k_layers[0]/h_bottom_star * uz[:, :, :, 0, :]) ** 2
        )
        # other surfaces
        bc_other = (
            jnp.mean((uy[:, :, 0, :, :]) ** 2)
            + jnp.mean((uy[:, :, -1, :, :]) ** 2)
            + jnp.mean((ux[:, 0, :, :, :]) ** 2)
            + jnp.mean((ux[:, -1, :, :, :]) ** 2)
        )

        return pde_res + lam_b * (bc_top + bc_bottom + bc_other)

    # isolate loss func from redundant arguments
    loss_fn = lambda model: PDE_loss(model, xc, yc, zc, fc)
    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient

@partial(jax.jit, static_argnums=(1,2))
def deepoheat_st_train_generator(fs, batch, nc, key):
    nx = nc
    ny = nc
    nz = stack.nz
    key, _ = jax.random.split(key)

    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False) # replace=True if data is small
    fc = fs[idx,:]
    xc = jnp.linspace(0, stack.x_size_norm, nx).reshape(-1,1)
    yc = jnp.linspace(0, stack.y_size_norm, ny).reshape(-1,1)
    zc = jnp.linspace(0, stack.z_interfaces_norm[-1], nz).reshape(-1,1)
    
    return xc, yc, zc, fc

@jax.jit
def deepoheat_st_test_generator(fs, u):
    # 这里只是生成坐标网格
    nz = stack.nz
    x = jnp.linspace(0, stack.x_size_norm, 101).reshape(-1,1)
    y = jnp.linspace(0, stack.y_size_norm, 101).reshape(-1,1)
    z = jnp.linspace(0, stack.z_interfaces_norm[-1], nz).reshape(-1,1)
    return x, y, z, fs, u

if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')
    parser.add_argument('--model_name', type=str, default='DeepOHeat_v1', choices=['DeepOHeat_ST', 'DeepOHeat_v1'], help='model name')
    parser.add_argument('--device_name', type=int, default=0, choices=[0, 1], help='GPU device')

    # training data settings
    parser.add_argument('--nc', type=int, default=101, help='the number of input points for each axis')
    parser.add_argument('--batch', type=int, default=4, help='the number of train functions')
    
    # training settings
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # 为了快速看结果，把 epoch 改少
    parser.add_argument('--epochs', type=int, default=1000, help='training epochs')
    parser.add_argument('--log_epoch', type=int, default=10, help='log the loss every chosen epochs')

    # model settings (保持默认)
    parser.add_argument('--dim', type=int, default=3, help='the input size')
    parser.add_argument('--branch_dim', type=int, default=101**2, help='the number of sensors')
    parser.add_argument('--field_dim', type=int, default=1, help='the dimension of the output field')
    parser.add_argument('--branch_depth', type=int, default=3, help='reduced for debug') # 稍微改小了层数以便快速初始化
    parser.add_argument('--branch_hidden', type=int, default=32, help='reduced for debug')
    parser.add_argument('--trunk_depth', type=int, default=3, help='the number of hidden layers')
    parser.add_argument('--trunk_hidden', type=int, default=32, help='the size of each hidden layer')
    parser.add_argument('--r', type=int, default=32, help='rank')
    
    args = parser.parse_args()

    #热源分布数据生成
    fs_train, fs_test, u_test = generate_power_map(n_train=50, n_test=10, nc=args.nc)

    #result_dir
    root_dir = os.path.join(os.getcwd(), 'results', 'chiplet', args.model_name)
    result_dir = os.path.join(root_dir, 'nf'+str(args.batch)+'_nc'+str(args.nc) + '_branch_' + str(args.branch_depth) + 
                              '_'+str(args.branch_hidden)+'_trunk_' + str(args.trunk_depth) + 
                              '_'+str(args.trunk_hidden)+'_r'+ str(args.r))    
    
    # make dir
    os.makedirs(result_dir, exist_ok=True)
    # clean old logs
    for f in ['log (loss).csv', 'log (eval metrics).csv', 'log (physics_loss).csv', 
              'total parameters.csv', 'total runtime (sec).csv', 'memory usage (mb).csv']:
        p = os.path.join(result_dir, f)
        if os.path.exists(p):
            os.remove(p)   


    # update function
    update_fn = update

    # define the optimizer
    schedule = optax.exponential_decay(args.lr, 1000, 0.9)
    optimizer = optax.adam(schedule)

    # random key
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key, 2)

    print(f"初始化模型: {args.model_name}...")
    # init model
    if args.model_name == 'DeepOHeat_ST':
        model = eqx.filter_jit(DeepOHeat_ST(dim=args.dim, branch_dim=args.branch_dim, field_dim=args.field_dim, 
                                            branch_depth=args.branch_depth, branch_hidden=args.branch_hidden, trunk_depth=args.trunk_depth, 
                                            trunk_hidden=args.trunk_hidden, rank=args.r, key=subkey))
    else:
        model = eqx.filter_jit(DeepOHeat_v1(dim=args.dim, branch_dim=args.branch_dim, field_dim=args.field_dim,
                                            branch_depth=args.branch_depth, branch_hidden=args.branch_hidden, trunk_depth=args.trunk_depth, 
                                            trunk_hidden=args.trunk_hidden, rank=args.r, key=subkey))
    
    params = eqx.filter(model, eqx.is_array)
    num_params = sum(jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.size, params)))
    print(f'Total number of parameters: {num_params}')

    # init state
    key, subkey = jax.random.split(key)
    opt_state = optimizer.init(params)

    # train/test generator
    # 注意：fs_train 的 batch 采样在这里进行
    train_generator = jax.jit(lambda key: deepoheat_st_train_generator(fs_train, args.batch, args.nc, key))
    test_generator = jax.jit(deepoheat_st_test_generator)
    loss_fn = apply_model_deepoheat_st

    print("开始训练...")

    model, optimizer, opt_state, runtime = train_loop(model, optimizer, opt_state, update_fn, train_generator, loss_fn, args.epochs, args.log_epoch, result_dir, args.device_name, subkey)
# print(stack.summary())

# params_star = stack.build_pinn_parameters()
# print("source_star =", params_star["source_star"])

# T_k = np.array([293.15, 303.15, 323.15])
# T_star = stack.normalize_temperature(T_k)
# print("T_star =", T_star)

# T_back = stack.denormalize_temperature(T_star)
# print("T_back =", T_back)
