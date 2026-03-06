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

# ---------------------------------------------------------
# 假设你已经有这些文件在目录下 (原项目自带)
# 如果没有这些文件，代码会报错
# ---------------------------------------------------------
try:
    from models import DeepOHeat_ST, DeepOHeat_v1
    from hvp import hvp_fwdfwd
    from train import train_loop, update
    from eval import eval_heat3d
except ImportError as e:
    print("Error: 缺少依赖文件 (models.py, hvp.py, etc.)。")
    print("请确保将此脚本放在原项目根目录下运行。")
    raise e

# =========================================================
# 1. 新增：模拟数据生成器 (替代本地文件加载)
# =========================================================
def generate_fake_data(n_train=100, n_test=20, nc=101):
    print("--- 正在生成模拟数据 (无需下载数据集) ---")
    nx, ny = nc, nc
    nz = int(0.55 * nc + 0.45) # 56
    dim_branch = nx * ny
    
    # 模拟热源 f (Batch, 101*101)
    # 随机生成一些功率值，模拟原本的数据分布
    key = jax.random.PRNGKey(0)
    fs_train = jax.random.uniform(key, (n_train, dim_branch), minval=0.1, maxval=2.0)
    fs_test = jax.random.uniform(key, (n_test, dim_branch), minval=0.1, maxval=2.0)
    
    # 模拟真实场 u (Batch, 101, 101, 56, 1)
    # 因为我们没有真实解，这里用随机数填充，仅仅为了让代码不报错。
    # 注意：这样跑出来的 Test Error (MSE) 是没有物理意义的，但 Loss (PDE Residual) 是有意义的。
    u_test = jax.random.uniform(key, (n_test, nx, ny, nz, 1))
    
    print(f"模拟数据生成完毕:")
    print(f"fs_train: {fs_train.shape}")
    print(f"fs_test:  {fs_test.shape}")
    print(f"u_test:   {u_test.shape}")
    return fs_train, fs_test, u_test

# =========================================================
# 2. 原本的辅助函数 (保持不变)
# =========================================================
@jax.jit
def create_mesh(xi_batch, yi_batch, zi_batch):
    return jnp.meshgrid(xi_batch.ravel(), yi_batch.ravel(), zi_batch.ravel(), indexing='ij')

#########################################################################
# Loss function (保持原本逻辑，包含原本的 0.1 Bug)
#########################################################################
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
        
        # PDE residual
        laplacian = (uxx + uyy + uzz)
        laplacian_bottom_power = laplacian[:,:,:,10:11,:] # z = 0.1, bottom interface
        laplacian_interior_power = laplacian[:,:,:,11:15,:]
        laplacian_top_power = laplacian[:,:,:,15:16,:] # z = 0.15, top interface
        
        # harmonic average
        k_bottom = 2 * 0.1 * 2 / (0.1 + 2)
        k_interior = 0.1
        k_top = 0.1 
        
        # ---------------------------------------------------------
        # 注意：这里保留了你指出的潜在 Bug (0.1 而不是 2.0)
        # 以确保"尽量不改动原本代码"。
        # ---------------------------------------------------------
        pde_res = jnp.concatenate([0.1*laplacian[:,:,:,16:,:],
                                    k_top*laplacian_top_power+f.reshape(-1, 101, 101, 1, 1),
                                    k_interior*laplacian_interior_power+2*f.reshape(-1, 101, 101, 1, 1),
                                    k_bottom*laplacian_bottom_power+f.reshape(-1, 101, 101, 1, 1),
                                    2*laplacian[:,:,:,0:10,:] 
            ],axis=3)
        pde_res = jnp.mean(pde_res**2)
        
        # top surface
        bc_top = jnp.mean((u[:,:,:,-1,:] - 0.2 + 2*uz[:,:,:,-1,:])**2)
        # bottom surface
        bc_bottom = jnp.mean((u[:,:,:,0,:] - 0.2 - 40*uz[:,:,:,0,:])**2)
        # other_surfaces
        bc_other = jnp.mean((uy[:,:,0,:,:])**2) + jnp.mean((uy[:,:,-1,:,:])**2) + jnp.mean((ux[:,0,:,:,:])**2) + jnp.mean((ux[:,-1,:,:,:])**2)
      
        return pde_res + lam_b*(bc_top + bc_bottom + bc_other)

    # isolate loss func from redundant arguments
    loss_fn = lambda model: PDE_loss(model, xc, yc, zc, fc)
                        
    loss, gradient = eqx.filter_value_and_grad(loss_fn)(model)

    return loss, gradient

#########################################################################
# Train generator (保持不变)
#########################################################################
@partial(jax.jit, static_argnums=(1,2))
def deepoheat_st_train_generator(fs, batch, nc, key):
    nx = nc
    ny = nc
    nz = int(0.55*nc + 0.45)
    key, _ = jax.random.split(key)
    # 注意：如果fs样本数少于batch，这里可能会报错，所以下面主程序里我保证了生成的样本够多
    idx = jax.random.choice(key, fs.shape[0], (batch,), replace=False) # replace=True if data is small
    fc = fs[idx,:]
    xc = jnp.linspace(0, 1, nx).reshape(-1,1)
    yc = jnp.linspace(0, 1, ny).reshape(-1,1)
    zc = jnp.linspace(0, 0.55, nz).reshape(-1,1)
    
    return xc, yc, zc, fc

#########################################################################
# Test generator (保持不变)
#########################################################################
@jax.jit
def deepoheat_st_test_generator(fs, u):
    # 这里只是生成坐标网格
    x = jnp.linspace(0, 1, 101).reshape(-1,1)
    y = jnp.linspace(0, 1, 101).reshape(-1,1)
    z = jnp.linspace(0, 0.55, 56).reshape(-1,1)
    return x, y, z, fs, u


# =========================================================
# 3. Main Block (修改了数据加载和默认参数)
# =========================================================
if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')
    parser.add_argument('--model_name', type=str, default='DeepOHeat_v1', choices=['DeepOHeat_ST', 'DeepOHeat_v1'], help='model name')
    parser.add_argument('--device_name', type=int, default=0, choices=[0, 1], help='GPU device')

    # training data settings
    parser.add_argument('--nc', type=int, default=101, help='the number of input points for each axis')
    # 为了跑通，把 batch 默认改小一点，防止你的显存爆炸
    parser.add_argument('--batch', type=int, default=4, help='the number of train functions')
    
    # training settings
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # 为了快速看结果，把 epoch 改少
    parser.add_argument('--epochs', type=int, default=100000, help='training epochs')
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

    # >>>>>>>>>> 关键修改：使用模拟数据替代 jnp.load <<<<<<<<<<
    # 原始代码:
    # fs_train = jnp.load('data/fs_train_volume.npy').reshape(-1,101**2)
    # fs_test = jnp.load('data/fs_test_volume.npy').reshape(-1,101**2)
    # u_test = jnp.load('data/u_test_volume.npy')
    
    # 模拟生成足够的数据以供训练
    # 保证样本数(50)大于 batch size(4)
    fs_train, fs_test, u_test = generate_fake_data(n_train=50, n_test=10, nc=args.nc)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # result dir
    root_dir = os.path.join(os.getcwd(), 'results', 'debug_run', args.model_name)
    result_dir = os.path.join(root_dir, 'debug_test_run')
    
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
    # ------------------ 新增：开始计时 ------------------
    start_time = time.time()
    # ---------------------------------------------------

    # train the model
    # 这里的 u_test 是随机生成的，所以 Eval 出来的误差会很大，请忽略 Eval 结果，主要看 Physics Loss 是否下降
    model, optimizer, opt_state, runtime = train_loop(model, optimizer, opt_state, update_fn, train_generator, loss_fn, args.epochs, args.log_epoch, result_dir, args.device_name, subkey)
    
    # ------------------ 新增：结束计时与计算 ------------------
    end_time = time.time()
    total_duration = end_time - start_time
    
    # 打印到控制台
    print(f"训练结束。")
    print(f"总耗时 (包含编译): {total_duration:.2f} 秒")
    print(f"总耗时 (分钟): {total_duration/60:.2f} 分钟")
    # -----------------------------------------------------
    
    # save the model
    eqx.tree_serialise_leaves(os.path.join(result_dir, args.model_name+'_trained_model.eqx'), model)
    print("训练结束，模型已保存。")

# =========================================================
    # 4. 评估与可视化 (修正版：适配 Separated Trunk 架构)
    # =========================================================
    import matplotlib.pyplot as plt
    
    print("\n--- 开始评估与可视化 (Separated Trunk Mode) ---")

    # 4.1 生成测试输入
    # -----------------------------------------------------
    # 1. 生成随机功率 f (1, 101^2)
    eval_key = jax.random.PRNGKey(123)
    f_eval = jax.random.uniform(eval_key, (1, args.branch_dim), minval=0.1, maxval=2.0)
    
    # 2. 生成三轴坐标 (直接使用线性向量，不需要 Meshgrid)
    # 对应你提到的 deepoheat_st_test_generator 逻辑
    x_axis = jnp.linspace(0, 1, args.nc).reshape(-1, 1)        # (101, 1)
    y_axis = jnp.linspace(0, 1, args.nc).reshape(-1, 1)        # (101, 1)
    z_axis = jnp.linspace(0, 0.55, int(0.55*args.nc + 0.45)).reshape(-1, 1) # (56, 1)
    
    print(f"输入形状: x={x_axis.shape}, y={y_axis.shape}, z={z_axis.shape}")
    
    # 4.2 模型推理
    # -----------------------------------------------------
    @eqx.filter_jit
    def predict_st(model, x, y, z, f):
        # DeepOHeat_ST 内部会自动处理 (x, y, z) 的张量积
        # 输出形状通常应该是 (Batch, Nx, Ny, Nz, 1) 或者类似的组合
        return model(((x, y, z), f))

    print("正在进行推理...")
    # 直接一次性预测，不会 OOM，因为输入只有几百个点
    u_pred_raw = predict_st(model, x_axis, y_axis, z_axis, f_eval)
    
    # 4.3 结果形状处理
    # -----------------------------------------------------
    # 转换成 numpy
    u_pred_3d = np.array(u_pred_raw)
    
    print(f"原始输出形状: {u_pred_3d.shape}")
    
    # 这里的形状可能是 (1, 101, 101, 56, 1) 或者 (101, 101, 56, 1)
    # 我们统一 squeeze 掉维度为 1 的轴，确保剩下 (101, 101, 56)
    u_pred_3d = np.squeeze(u_pred_3d)
    
    # 二次确认形状 (防止 Batch 维还在)
    if u_pred_3d.ndim == 4: # 假设是 (Batch, x, y, z)
        u_pred_3d = u_pred_3d[0]
        
    print(f"处理后 3D 形状: {u_pred_3d.shape}") # 应该是 (101, 101, 56)
    
    # 4.4 绘制 X-Z 竖切面 (y=0.5)
    # -----------------------------------------------------
    # 找到 y 轴中间的索引
    slice_idx = args.nc // 2 
    
    # 切片: 取出 y=0.5 那个面 -> (x, z)
    # 注意：DeepOHeat_ST 的输出轴顺序通常是 (x, y, z)
    u_slice = u_pred_3d[:, slice_idx, :] 
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # u_slice.T 让 x 轴做横轴，z 轴做纵轴
    im = plt.imshow(u_slice.T, cmap='inferno', origin='lower', 
                    extent=[0, 1, 0, 0.55], aspect='auto')
    
    cbar = plt.colorbar(im)
    cbar.set_label('Temperature (Normalized)', rotation=270, labelpad=15)
    
    plt.title(f'Vertical Cross-Section (ST Model, y=0.5)')
    plt.xlabel('X Coordinate (mm)')
    plt.ylabel('Z Coordinate (mm)')
    
    # 辅助线
    plt.axhline(y=0.1, color='white', linestyle='--', alpha=0.5)
    plt.axhline(y=0.15, color='white', linestyle='--', alpha=0.5)
    plt.text(0.02, 0.05, 'Bottom Layer', color='white', fontsize=8)
    plt.text(0.02, 0.12, 'Heat Source', color='white', fontsize=8)
    
    save_path = os.path.join(result_dir, 'st_model_slice.png')
    plt.savefig(save_path, dpi=150)
    print(f"可视化完成！已保存至: {save_path}")