import jax
import jax.numpy as jnp


def scalar_model_output(model, x, y, z):
    """
    模型输入单点 (x, y, z)，输出标量 T*
    """
    xyz = jnp.array([x, y, z])
    out = model(xyz)
    return jnp.squeeze(out)


def get_layer_index_from_zstar(z_star, z_interfaces_star):
    """
    根据 z* 判断属于哪一层

    Parameters
    ----------
    z_star : scalar or array
    z_interfaces_star : shape (8,)

    Returns
    -------
    idx : int or array of int
        0~6
    """
    return jnp.sum(z_star[..., None] >= z_interfaces_star[1:-1], axis=-1)


def select_layer_property(z_star, z_interfaces_star, prop_per_layer):
    """
    根据 z* 选择所属层的物性参数
    """
    idx = get_layer_index_from_zstar(z_star, z_interfaces_star)
    return prop_per_layer[idx]


# =========================
# PDE loss: ∇*²T* + source_star = 0
# =========================

def pde_residual_single(model, x, y, z, z_interfaces_star, source_star_all):
    """
    单点 PDE 残差

    层内方程:
        T_xx* + T_yy* + T_zz* + source_star = 0
    """
    T_fn = lambda x_, y_, z_: scalar_model_output(model, x_, y_, z_)

    T_xx = jax.grad(jax.grad(T_fn, argnums=0), argnums=0)(x, y, z)
    T_yy = jax.grad(jax.grad(T_fn, argnums=1), argnums=1)(x, y, z)
    T_zz = jax.grad(jax.grad(T_fn, argnums=2), argnums=2)(x, y, z)

    s_star = select_layer_property(z, z_interfaces_star, source_star_all)

    return T_xx + T_yy + T_zz + s_star


pde_residual_batch = jax.vmap(
    pde_residual_single,
    in_axes=(None, 0, 0, 0, None, None)
)


def pde_loss(model, xyz_f, params_star):
    x = xyz_f[:, 0]
    y = xyz_f[:, 1]
    z = xyz_f[:, 2]

    res = pde_residual_batch(
        model,
        x, y, z,
        params_star["z_interfaces_star"],
        params_star["source_star"],
    )
    return jnp.mean(res ** 2)


# =========================
# Top convection BC
# -k* dT/dn = h*(T - T_amb*)
# top outward normal = +z
# residual: -k* T_z - h*(T - T_amb*) = 0
# =========================

def bc_top_residual_single(model, x, y, z, h_top_star, T_amb_star, k_top_star):
    T_fn = lambda x_, y_, z_: scalar_model_output(model, x_, y_, z_)

    T = T_fn(x, y, z)
    T_z = jax.grad(T_fn, argnums=2)(x, y, z)

    return -k_top_star * T_z - h_top_star * (T - T_amb_star)


bc_top_residual_batch = jax.vmap(
    bc_top_residual_single,
    in_axes=(None, 0, 0, 0, None, None, None)
)


def bc_top_loss(model, xyz_top, params_star):
    h_top_star = params_star["h_top_star"]
    if h_top_star is None:
        return 0.0

    k_top_star = params_star["k_star"][-1]
    T_amb_star = params_star["T_amb_star"]

    x = xyz_top[:, 0]
    y = xyz_top[:, 1]
    z = xyz_top[:, 2]

    res = bc_top_residual_batch(
        model, x, y, z,
        h_top_star, T_amb_star, k_top_star
    )
    return jnp.mean(res ** 2)


# =========================
# Bottom convection BC
# bottom outward normal = -z
# residual: k* T_z - h*(T - T_amb*) = 0
# =========================

def bc_bottom_residual_single(model, x, y, z, h_bottom_star, T_amb_star, k_bottom_star):
    T_fn = lambda x_, y_, z_: scalar_model_output(model, x_, y_, z_)

    T = T_fn(x, y, z)
    T_z = jax.grad(T_fn, argnums=2)(x, y, z)

    return k_bottom_star * T_z - h_bottom_star * (T - T_amb_star)


bc_bottom_residual_batch = jax.vmap(
    bc_bottom_residual_single,
    in_axes=(None, 0, 0, 0, None, None, None)
)


def bc_bottom_loss(model, xyz_bottom, params_star):
    h_bottom_star = params_star["h_bottom_star"]
    if h_bottom_star is None:
        return 0.0

    k_bottom_star = params_star["k_star"][0]
    T_amb_star = params_star["T_amb_star"]

    x = xyz_bottom[:, 0]
    y = xyz_bottom[:, 1]
    z = xyz_bottom[:, 2]

    res = bc_bottom_residual_batch(
        model, x, y, z,
        h_bottom_star, T_amb_star, k_bottom_star
    )
    return jnp.mean(res ** 2)


# =========================
# Interface continuity
# 1) temperature continuity
# 2) heat flux continuity
# =========================

def interface_residual_single(model, x, y, z_left, z_right, k_left, k_right):
    T_fn = lambda x_, y_, z_: scalar_model_output(model, x_, y_, z_)

    T_left = T_fn(x, y, z_left)
    T_right = T_fn(x, y, z_right)

    Tz_left = jax.grad(T_fn, argnums=2)(x, y, z_left)
    Tz_right = jax.grad(T_fn, argnums=2)(x, y, z_right)

    res_T = T_left - T_right
    res_q = k_left * Tz_left - k_right * Tz_right

    return res_T, res_q


interface_residual_batch = jax.vmap(
    interface_residual_single,
    in_axes=(None, 0, 0, 0, 0, None, None)
)


def interface_loss(model, xy_interface, z_interface, k_left, k_right, eps=1e-4):
    """
    Parameters
    ----------
    xy_interface : shape (N, 2)
    z_interface : float
        界面位置 z*
    eps : float
        界面两侧的小偏移量
    """
    x = xy_interface[:, 0]
    y = xy_interface[:, 1]

    z_left = jnp.full_like(x, z_interface - eps)
    z_right = jnp.full_like(x, z_interface + eps)

    res_T, res_q = interface_residual_batch(
        model, x, y, z_left, z_right, k_left, k_right
    )

    return jnp.mean(res_T ** 2) + jnp.mean(res_q ** 2)


# =========================
# Total loss
# =========================

def total_loss(
    model,
    xyz_f,
    xyz_top,
    xyz_bottom,
    interface_batches,
    params_star,
    w_pde=1.0,
    w_top=1.0,
    w_bottom=1.0,
    w_interface=1.0,
):
    loss_pde = pde_loss(model, xyz_f, params_star)
    loss_top = bc_top_loss(model, xyz_top, params_star)
    loss_bottom = bc_bottom_loss(model, xyz_bottom, params_star)

    loss_interface = 0.0
    z_interfaces_star = params_star["z_interfaces_star"]
    k_star = params_star["k_star"]

    for i, xy_int in enumerate(interface_batches):
        z_int = z_interfaces_star[i + 1]
        k_left = k_star[i]
        k_right = k_star[i + 1]
        loss_interface = loss_interface + interface_loss(
            model, xy_int, z_int, k_left, k_right
        )

    return (
        w_pde * loss_pde
        + w_top * loss_top
        + w_bottom * loss_bottom
        + w_interface * loss_interface
    )


def loss_breakdown(
    model,
    xyz_f,
    xyz_top,
    xyz_bottom,
    interface_batches,
    params_star,
):
    loss_pde = pde_loss(model, xyz_f, params_star)
    loss_top = bc_top_loss(model, xyz_top, params_star)
    loss_bottom = bc_bottom_loss(model, xyz_bottom, params_star)

    loss_interface = 0.0
    z_interfaces_star = params_star["z_interfaces_star"]
    k_star = params_star["k_star"]

    for i, xy_int in enumerate(interface_batches):
        z_int = z_interfaces_star[i + 1]
        k_left = k_star[i]
        k_right = k_star[i + 1]
        loss_interface = loss_interface + interface_loss(
            model, xy_int, z_int, k_left, k_right
        )

    return {
        "pde": loss_pde,
        "top_bc": loss_top,
        "bottom_bc": loss_bottom,
        "interface": loss_interface,
        "total": loss_pde + loss_top + loss_bottom + loss_interface,
    }