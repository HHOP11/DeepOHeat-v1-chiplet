from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


DEFAULT_T_SHIFT_K = 293.15
DEFAULT_MIN_SAMPLING_SPACING_M = 5e-6


@dataclass
class LayerProperty:
    """
    单层物理属性
    """
    name: str
    k: float               # W/(m·K)
    thickness: float       # m


@dataclass
class BoundaryProperty:
    """
    边界属性
    """
    h_top: Optional[float] = None        # W/(m^2·K)
    h_bottom: Optional[float] = None     # W/(m^2·K)
    T_amb_k: float = 298.15              # K


@dataclass
class NormalizationScales:
    """
    无量纲参考尺度
    """
    L_ref: float
    k_ref: float
    q_ref: float
    dT_ref: float
    T_shift_k: float

    @staticmethod
    def from_physics(
        total_thickness: float,
        k_ref: float,
        q_ref: float,
        T_shift_k: float = DEFAULT_T_SHIFT_K,
    ) -> "NormalizationScales":
        if total_thickness <= 0:
            raise ValueError("total_thickness 必须大于 0")
        if k_ref <= 0:
            raise ValueError("k_ref 必须大于 0")
        if q_ref <= 0:
            raise ValueError("q_ref 必须大于 0")

        L_ref = total_thickness
        dT_ref = q_ref * L_ref ** 2 / k_ref

        return NormalizationScales(
            L_ref=L_ref,
            k_ref=k_ref,
            q_ref=q_ref,
            dT_ref=dT_ref,
            T_shift_k=T_shift_k,
        )


@dataclass
class ChipletStack:
    """
    7层 chiplet 结构

    温度归一化:
        T* = (T - T_shift_k) / dT_ref

    层内 PDE 采用 q/k 形式:
        ∇²T + q/k = 0

    无量纲后:
        ∇*²T* + s* = 0
    其中:
        s_i* = q_i L_ref² / (k_i dT_ref)
    """
    layers: List[LayerProperty]
    boundary: BoundaryProperty
    q_ref: float
    x_size_m: float
    y_size_m: float
    T_shift_k: float = DEFAULT_T_SHIFT_K
    min_sampling_spacing_m: float = DEFAULT_MIN_SAMPLING_SPACING_M
    power_sample_min: float = 0.1
    power_sample_max: float = 2.0
    scales: NormalizationScales = field(init=False)

    def __post_init__(self):
        if len(self.layers) != 7:
            raise ValueError(f"需要 7 层，当前为 {len(self.layers)} 层")

        total_thickness = sum(layer.thickness for layer in self.layers)
        if total_thickness <= 0:
            raise ValueError("总厚度必须大于 0")
        if self.x_size_m <= 0:
            raise ValueError("x_size_m 必须大于 0")
        if self.y_size_m <= 0:
            raise ValueError("y_size_m 必须大于 0")
        if self.min_sampling_spacing_m <= 0:
            raise ValueError("min_sampling_spacing_m 必须大于 0")
        if self.min_sampling_spacing_m > total_thickness:
            raise ValueError("min_sampling_spacing_m 不能大于总厚度")
        if self.power_sample_min >= self.power_sample_max:
            raise ValueError("power_sample_min 必须小于 power_sample_max")

        for i, layer in enumerate(self.layers):
            if layer.k <= 0:
                raise ValueError(f"第 {i+1} 层热导率 k 必须大于 0")
            if layer.thickness <= 0:
                raise ValueError(f"第 {i+1} 层 thickness 必须大于 0")

        k_ref = max(layer.k for layer in self.layers)
        q_ref = self.q_ref

        self.scales = NormalizationScales.from_physics(
            total_thickness=total_thickness,
            k_ref=k_ref,
            q_ref=q_ref,
            T_shift_k=self.T_shift_k,
        )

    @property
    def total_thickness(self) -> float:
        return sum(layer.thickness for layer in self.layers)

    @property
    def x_size_norm(self) -> float:
        return self.x_size_m / self.scales.L_ref

    @property
    def y_size_norm(self) -> float:
        return self.y_size_m / self.scales.L_ref

    @property
    def nz(self) -> int:
        return max(2, int(round(self.total_thickness / self.min_sampling_spacing_m)) + 1)

    @property
    def z_interfaces(self) -> np.ndarray:
        """
        原始量纲界面位置: [0, z1, ..., z7]
        """
        z = [0.0]
        acc = 0.0
        for layer in self.layers:
            acc += layer.thickness
            z.append(acc)
        return np.array(z, dtype=np.float64)

    @property
    def z_interfaces_norm(self) -> np.ndarray:
        """
        无量纲界面位置
        """
        return self.z_interfaces / self.scales.L_ref

    def normalized_layer_properties(self) -> List[Dict[str, float]]:
        """
        返回每层无量纲参数

        k_star : 用于边界和界面热流连续
        d_star : 无量纲厚度
        """
        props = []
        for layer in self.layers:
            k_star = layer.k / self.scales.k_ref
            d_star = layer.thickness / self.scales.L_ref
            props.append({
                "name": layer.name,
                "k_star": k_star,
                "d_star": d_star,
            })
        return props

    def source_star_from_qvol(self, q_vol_per_layer: np.ndarray) -> np.ndarray:
        """
        输入每层体热源 q_vol (shape=(7,))，返回 PDE 源项 s* (shape=(7,))
        s_i* = q_i L_ref^2 / (k_i dT_ref)
        """
        q_vol_per_layer = np.asarray(q_vol_per_layer, dtype=np.float64)
        if q_vol_per_layer.shape != (7,):
            raise ValueError("q_vol_per_layer 形状必须是 (7,)")
        k = np.array([layer.k for layer in self.layers], dtype=np.float64)
        return q_vol_per_layer * self.scales.L_ref ** 2 / (k * self.scales.dT_ref)

    def q_star_from_qvol(self, q_vol_per_layer: np.ndarray) -> np.ndarray:
        """
        输入每层体热源 q_vol (shape=(7,))，返回 q* (shape=(7,))
        """
        q_vol_per_layer = np.asarray(q_vol_per_layer, dtype=np.float64)
        if q_vol_per_layer.shape != (7,):
            raise ValueError("q_vol_per_layer 形状必须是 (7,)")
        return q_vol_per_layer / self.scales.q_ref

    def normalized_boundary_properties(self) -> Dict[str, Optional[float]]:
        """
        h* = h L_ref / k_ref
        T_amb* = (T_amb_k - T_shift_k) / dT_ref
        """
        h_top_star = None
        h_bottom_star = None

        if self.boundary.h_top is not None:
            h_top_star = self.boundary.h_top * self.scales.L_ref / self.scales.k_ref

        if self.boundary.h_bottom is not None:
            h_bottom_star = self.boundary.h_bottom * self.scales.L_ref / self.scales.k_ref

        T_amb_star = (
            self.boundary.T_amb_k - self.scales.T_shift_k
        ) / self.scales.dT_ref

        return {
            "h_top_star": h_top_star,
            "h_bottom_star": h_bottom_star,
            "T_amb_star": T_amb_star,
            "T_amb_k": self.boundary.T_amb_k,
        }

    def normalize_temperature(self, T_k: np.ndarray) -> np.ndarray:
        """
        T* = (T - T_shift_k) / dT_ref
        """
        return (T_k - self.scales.T_shift_k) / self.scales.dT_ref

    def denormalize_temperature(self, T_star: np.ndarray) -> np.ndarray:
        """
        T = T_shift_k + dT_ref * T*
        """
        return self.scales.T_shift_k + self.scales.dT_ref * T_star

    def normalize_ambient_temperature(self, T_amb_k: np.ndarray) -> np.ndarray:
        return (T_amb_k - self.scales.T_shift_k) / self.scales.dT_ref

    def normalize_length(self, x: np.ndarray) -> np.ndarray:
        return x / self.scales.L_ref

    def denormalize_length(self, x_star: np.ndarray) -> np.ndarray:
        return self.scales.L_ref * x_star

    def normalize_k(self, k: np.ndarray) -> np.ndarray:
        return k / self.scales.k_ref

    def normalize_h(self, h: np.ndarray) -> np.ndarray:
        return h * self.scales.L_ref / self.scales.k_ref

    def locate_layer(self, z: np.ndarray) -> np.ndarray:
        """
        输入原始量纲 z 返回所属层编号 0~6
        """
        interfaces = self.z_interfaces
        idx = np.digitize(z, interfaces[1:-1], right=False)
        return idx

    def locate_layer_norm(self, z_star: np.ndarray) -> np.ndarray:
        """
        输入无量纲 z*，返回所属层编号 0~6
        """
        z = z_star * self.scales.L_ref
        return self.locate_layer(z)

    def build_pinn_parameters(self, q_vol_per_layer: Optional[np.ndarray] = None) -> Dict[str, Any]:
        props = self.normalized_layer_properties()
        bc = self.normalized_boundary_properties()

        source_star = None
        q_star = None
        if q_vol_per_layer is not None:
            source_star = self.source_star_from_qvol(q_vol_per_layer)
            q_star = self.q_star_from_qvol(q_vol_per_layer)

        return {
            "k_star": np.array([p["k_star"] for p in props], dtype=np.float64),
            "q_star": q_star,
            "source_star": source_star,
            "d_star": np.array([p["d_star"] for p in props], dtype=np.float64),
            "z_interfaces_star": self.z_interfaces_norm.astype(np.float64),
            "h_top_star": bc["h_top_star"],
            "h_bottom_star": bc["h_bottom_star"],
            "T_amb_star": bc["T_amb_star"],
            "T_amb_k": bc["T_amb_k"],
            "T_shift_k": self.scales.T_shift_k,
            "L_ref": self.scales.L_ref,
            "k_ref": self.scales.k_ref,
            "q_ref": self.scales.q_ref,
            "dT_ref": self.scales.dT_ref,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "total_thickness": self.total_thickness,
            "z_interfaces": self.z_interfaces,
            "z_interfaces_norm": self.z_interfaces_norm,
            "layers_norm": self.normalized_layer_properties(),
            "boundary_norm": self.normalized_boundary_properties(),
            "scales": {
                "L_ref": self.scales.L_ref,
                "k_ref": self.scales.k_ref,
                "q_ref": self.scales.q_ref,
                "dT_ref": self.scales.dT_ref,
                "T_shift_k": self.scales.T_shift_k,
            },
        }
