import numpy as np
import paddle
import paddle.nn.functional as F
from energy_equation import EnergyEquation

import ppsci
from ppsci.utils import logger


class GradientReversalLayer(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class Problems:
    def __init__(self, dim, geo_origin, geo_dim, **kwargs):
        self.dim = dim
        self.geo_origin = geo_origin
        self.geo_dim = geo_dim
        self.volume = self.get_volume()

        default_params = {
            "volume_ratio": 0.5,
            "alpha": 5.0,
            "exponent": 3.0,
            "vol_penalty_strength": 10.0,
            "nu": 0.3,
            "E": 1.0,
            "use_oc": False,
            "max_move": 0.2,
            "damping_parameter": 0.5,
            "use_mmse": False,
            "filter": "none",
            "filter_radius": 2.0,
        }
        merged_params = {**default_params, **kwargs}
        for key, value in merged_params.items():
            setattr(self, key, value)

        lambda_ = (
            (self.nu * self.E / ((1 - self.nu * self.nu)))
            if dim == 2
            else self.nu * self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        )
        mu = self.E / (1 + self.nu) if dim == 2 else self.E / (2 * (1 + self.nu))
        self.equation = {
            "EnergyEquation": EnergyEquation(
                param_dict={"lambda_": lambda_, "mu": mu}, dim=self.dim
            ),
        }

        self.disp_net = None
        self.density_net = None

    def get_volume(self):
        if self.dim == 2:
            return self.geo_dim[0] * self.geo_dim[1]
        elif self.dim == 3:
            return self.geo_dim[0] * self.geo_dim[1] * self.geo_dim[2]

    # transforms
    def transform_in(self, _in):
        x, y = _in["x"], _in["y"]
        x_scaled = 2.0 / self.geo_dim[0] * x + (
            -1.0 - 2.0 * self.geo_origin[0] / self.geo_dim[0]
        )
        y_scaled = 2.0 / self.geo_dim[1] * y + (
            -1.0 - 2.0 * self.geo_origin[1] / self.geo_dim[1]
        )

        sin_x_scaled, sin_y_scaled = paddle.sin(x_scaled), paddle.sin(y_scaled)

        in_trans = {
            "x_scaled": x_scaled,
            "y_scaled": y_scaled,
            "sin_x_scaled": sin_x_scaled,
            "sin_y_scaled": sin_y_scaled,
        }

        if self.dim == 3:
            z = _in["z"]
            z_scaled = 2.0 / self.geo_dim[2] * z + (
                -1.0 - 2.0 * self.geo_origin[2] / self.geo_dim[2]
            )
            sin_z_scaled = paddle.sin(z_scaled)
            in_trans["z_scaled"] = z_scaled
            in_trans["sin_z_scaled"] = sin_z_scaled

        return in_trans

    def transform_out_disp(self, _in, _out):
        "Different for each problems because of different boundary constraints."
        return _out

    def transform_out_density(self, _in, _out):
        density = _out["density"]
        offset = np.log(self.volume_ratio / (1.0 - self.volume_ratio))
        densities = F.sigmoid(self.alpha * density + offset)
        return {"densities": densities}

    # functions
    def compute_energy(self, densities, energy, revers_grad=False):
        if revers_grad:
            densities = GradientReversalLayer.apply(densities)
        energy_densities = paddle.pow(densities, self.exponent) * energy
        return self.volume * paddle.mean(energy_densities, keepdim=True)

    def compute_force(self):
        "Different for each problems because of different force."
        return 0.0

    def compute_penalty(self, densities):
        target_volume = self.volume_ratio * self.volume
        volume_estimate = self.volume * paddle.mean(densities, keepdim=True)
        return (
            self.vol_penalty_strength
            * (volume_estimate - target_volume)
            * (volume_estimate - target_volume)
            / target_volume
        )

    # oc
    def compute_oc_multi_batch(self, densities_list, sensitivities_list):
        target_volume = self.volume_ratio * self.volume
        logger.info(f"target_volume: {target_volume}")
        lagrange_lower_estimate = 0
        lagrange_upper_estimate = 1e9
        conv_threshold = 1e-3

        total_samples = len(densities_list) * densities_list[0].shape[0]
        dv = self.volume / total_samples

        density_lower_bound = [
            paddle.maximum(paddle.to_tensor(0.0), od - self.max_move)
            for od in densities_list
        ]
        density_upper_bound = [
            paddle.minimum(paddle.to_tensor(1.0), od + self.max_move)
            for od in densities_list
        ]

        while (lagrange_upper_estimate - lagrange_lower_estimate) / (
            lagrange_lower_estimate + lagrange_upper_estimate
        ) > conv_threshold:
            lagrange_current = 0.5 * (lagrange_upper_estimate + lagrange_lower_estimate)

            target_densities = [
                (
                    paddle.multiply(
                        densities_list[i],
                        paddle.pow(
                            paddle.divide(
                                sensitivities_list[i],
                                paddle.to_tensor(-dv * lagrange_current),
                            ),
                            self.damping_parameter,
                        ),
                    )
                )
                .detach()
                .clone()
                for i in range(len(densities_list))
            ]

            target_densities = [
                paddle.maximum(
                    density_lower_bound[i],
                    paddle.minimum(density_upper_bound[i], target_densities[i]),
                )
                for i in range(len(densities_list))
            ]

            new_volume = self.volume * np.mean(
                [paddle.mean(di) for di in target_densities]
            )
            # print("new_volume", new_volume)

            if new_volume > target_volume:
                lagrange_lower_estimate = lagrange_current
            else:
                lagrange_upper_estimate = lagrange_current
            if (lagrange_lower_estimate + lagrange_upper_estimate) < conv_threshold:
                break
        logger.info(f"new_volume: {new_volume}")
        return target_densities

    def compute_target_densities_gradient_descent(
        self, densities_list, sensitivities_list
    ):
        projected_sensitivities = [
            (
                paddle.maximum(
                    paddle.to_tensor(0.0),
                    paddle.minimum(
                        paddle.to_tensor(1.0), densities_list[i] - sensitivities_list[i]
                    ),
                )
                - densities_list[i]
            )
            for i in range(len(densities_list))
        ]

        step_size = 0.05 / paddle.mean(
            paddle.to_tensor([paddle.abs(si) for si in projected_sensitivities]),
            keepdim=True,
        )
        return [
            densities_list[i] - step_size * sensitivities_list[i]
            for i in range(len(densities_list))
        ]

    def compute_target_densities(self, densities_list, sensitivities_list):
        if self.use_oc:
            return self.compute_oc_multi_batch(densities_list, sensitivities_list)
        else:
            return self.compute_target_densities_gradient_descent(
                densities_list, sensitivities_list
            )

    # fliter
    def apply_sensitivity_filter(self, sample_positions, old_densities, sensitivities):
        if self.dim == 2:
            return self.apply_sensitivity_filter_2d(
                sample_positions, old_densities, sensitivities
            )

        raise Exception(f"{self.dim} dims is unsupported now")

    def apply_sensitivity_filter_2d(
        self, sample_positions, old_densities, sensitivities
    ):
        gamma = 1e-3

        cell_width = (self.geo_dim[1] - self.geo_dim[0]) / self.batch_size[0]
        grads = sensitivities

        radius_space = self.filter_radius * cell_width
        filter_size = 2 * int(round(self.filter_radius)) + 1
        h_pad = self.batch_size[1] + 2 * (filter_size // 2)
        w_pad = self.batch_size[0] + 2 * (filter_size // 2)
        out_h = h_pad - filter_size + 1
        out_w = w_pad - filter_size + 1

        density_patches = paddle.reshape(
            old_densities, [1, 1, self.batch_size[1], self.batch_size[0]]
        )
        density_patches = self.pad_border(density_patches, filter_size)
        density_patches = F.unfold(
            density_patches,
            kernel_sizes=[filter_size, filter_size],
            strides=1,
            paddings=0,
        ).reshape(
            [1, filter_size * filter_size, out_h, out_w]
        )  # [1, filter_size^2, H, W]
        density_patches = paddle.transpose(density_patches, [0, 2, 3, 1])

        sensitivity_patches = paddle.reshape(
            sensitivities, [1, 1, self.batch_size[1], self.batch_size[0]]
        )
        sensitivity_patches = self.pad_border(sensitivity_patches, filter_size)
        sensitivity_patches = F.unfold(
            sensitivity_patches,
            kernel_sizes=[filter_size, filter_size],
            strides=1,
            paddings=0,
        ).reshape([1, filter_size * filter_size, out_h, out_w])
        sensitivity_patches = paddle.transpose(sensitivity_patches, [0, 2, 3, 1])

        sample_positions = paddle.reshape(
            sample_positions, [1, self.batch_size[1], self.batch_size[0], self.dim]
        )
        sample_patches = self.pad_positions_constant(
            sample_positions, filter_size
        ).transpose([0, 3, 1, 2])
        sample_patches = F.unfold(
            sample_patches,
            kernel_sizes=[filter_size, filter_size],
            strides=1,
            paddings=0,
        )
        sample_patches = paddle.reshape(
            sample_patches,
            [
                1,
                self.batch_size[1],
                self.batch_size[0],
                filter_size * filter_size,
                self.dim,
            ],
        )

        pos_centers = paddle.reshape(
            sample_positions, [1, self.batch_size[1], self.batch_size[0], 1, self.dim]
        )
        diff = sample_patches - pos_centers
        # dists = paddle.sqrt(paddle.sum(diff * diff, axis=-1))
        diff_sq = paddle.sum(diff * diff, axis=-1)
        diff_sq = paddle.clip(diff_sq, min=0.0)
        dists = paddle.sqrt(diff_sq)

        alpha = 10.0
        Hei = paddle.nn.functional.sigmoid(alpha * (radius_space - dists))

        # Hei = paddle.maximum(paddle.to_tensor(0.0), radius_space - dists)
        Heixic = Hei * density_patches * sensitivity_patches
        sum_Heixic = paddle.sum(Heixic, axis=-1)
        sum_Hei = paddle.sum(Hei, axis=-1)

        old_densities_r = paddle.reshape(
            old_densities, [1, self.batch_size[1], self.batch_size[0]]
        )
        div = paddle.maximum(paddle.to_tensor(gamma), old_densities_r) * sum_Hei + 1e-8
        grads = sum_Heixic / div

        return paddle.reshape(grads, [-1, 1])

    def pad_border(self, x, filter_size):
        pad_size = filter_size // 2
        return F.pad(x, [pad_size] * 4, mode="constant", value=0)

    def pad_positions_constant(self, x, filter_size):
        pad_size = filter_size // 2
        return F.pad(
            x, [0, 0, pad_size, pad_size, pad_size, pad_size, 0, 0], mode="constant"
        )

    # loss functions
    def disp_loss_func(
        self,
        output_dict,
        label_dict=None,
        weight_dict={},
        input_dict=None,
    ):
        densities = self.density_net(input_dict)["densities"]
        densities = densities.detach().clone()
        energy = (
            output_dict["energy_xy"] if self.dim == 2 else output_dict["energy_xyz"]
        )
        loss_energy = self.compute_energy(densities, energy)
        loss_force = self.compute_force()
        # logger.info(f"loss_energy: {float(loss_energy)}")
        # logger.info(f"loss_force: {float(loss_force)}")
        return loss_energy + loss_force

    def density_loss_func(
        self,
        output_dicts_list,
        label_dicts_list=None,
        weight_dicts_list=None,
        input_dicts_list=None,
    ):
        if not isinstance(output_dicts_list, list):
            output_dicts_list = [output_dicts_list]
            input_dicts_list = [input_dicts_list]

        loss_list = []
        densities_list = []
        sensitivities_list = []
        for i, output_dict in enumerate(output_dicts_list):
            input_dict = input_dicts_list[i]
            if isinstance(output_dict, list):
                output_dict = output_dict[0]
                input_dict = input_dict[0]
            densities = output_dict["densities"]
            energy_xy = self.equation["EnergyEquation"].equations["energy_xy"](
                {**self.disp_net(input_dict), **input_dict}
            )
            energy_xy = energy_xy.clone()

            loss = self.compute_energy(densities, energy_xy, revers_grad=True)
            if not self.use_oc:
                loss += self.compute_penalty(densities)
            loss_list.append(loss)

            sensitivities = paddle.grad(loss, densities)[0]
            # add fliter
            if self.filter == "sensitivity":
                sample_positions = paddle.concat(
                    [v for k, v in input_dict.items() if k != "sdf"], axis=-1
                )
                sensitivities = self.apply_sensitivity_filter(
                    sample_positions,
                    densities,
                    sensitivities,
                )

            densities_list.append(densities)
            sensitivities_list.append(sensitivities)
            ppsci.autodiff.clear()

        if not self.use_mmse:
            return loss_list
        else:
            target_densities_list = self.compute_target_densities(
                densities_list, sensitivities_list
            )
            logger.info(
                f"use_mmse: {np.mean([td.numpy().mean() for td in target_densities_list])}"
            )
            return [
                F.mse_loss(densities_list[i], target_densities_list[i], "mean")
                for i in range(len(target_densities_list))
            ]

    # eval metric functions
    def density_metric_func(self, output_dict, *args):
        density = output_dict["densities"]
        logger.info(f"mean: {float(paddle.mean(density))}")
        logger.info(f"max: {float(paddle.max(density))}")
        logger.info(f"min: {float(paddle.min(density))}")
        metric_dict = {"densities": density.mean() - self.volume_ratio}
        return metric_dict


class Beam2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.5, 0.5)
        super().__init__(2, geo_origin, geo_dim, **cfg)

        beam = ppsci.geometry.Rectangle((0.0, 0.0), (1.5, 0.5))
        self.geom = {"geo": beam}
        self.force = -0.0025
        self.batch_size = (150, 50)

    # bc
    def transform_out_disp(self, _in, _out):
        x_scaled = _in["x_scaled"]
        x = self.geo_dim[0] / 2 * (1 + x_scaled) + self.geo_origin[0]
        u, v = x * _out["u"], x * _out["v"]
        return {"u": u, "v": v}

    # force
    def compute_force(self):
        input_pos = {
            "x": paddle.to_tensor([[1.5]], dtype=paddle.get_default_dtype()),
            "y": paddle.to_tensor([[0.0]], dtype=paddle.get_default_dtype()),
        }
        output_pos = self.disp_net(input_pos)
        v = output_pos["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class Distributed2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.5, 0.5)
        super().__init__(2, geo_origin, geo_dim, **cfg)

        beam = ppsci.geometry.Rectangle((0.0, 0.0), (1.5, 0.5))
        self.geom = {"geo": beam}
        self.force = -0.0025
        self.batch_size = (150, 50)

    # bc
    def transform_out_disp(self, _in, _out):
        x_scaled = _in["x_scaled"]
        x = self.geo_dim[0] / 2 * (1 + x_scaled) + self.geo_origin[0]
        u, v = x * _out["u"], x * _out["v"]
        return {"u": u, "v": v}

    # force
    def get_force_pos(self):
        sample_num = 400
        input_pos_np = self.geom["geo"].sample_boundary(
            n=sample_num,
            criteria=lambda x, y: y >= self.geo_dim[1] - 1e-3,
        )
        return {
            "x": paddle.to_tensor(input_pos_np["x"], dtype=paddle.get_default_dtype()),
            "y": paddle.full((sample_num, 1), 0.5, dtype=paddle.get_default_dtype()),
        }

    def compute_force(self):
        input_pos = self.get_force_pos()
        output_pos = self.disp_net(input_pos)
        v = output_pos["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class LongBeam2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.0, 0.5)  # (2.0, 0.5)
        super().__init__(2, geo_origin, geo_dim, **cfg)

        long_beam = ppsci.geometry.Rectangle(geo_origin, geo_dim)
        self.geom = {"geo": long_beam}
        self.force = -0.0025
        self.batch_size = (122, 61)  # 50 * 100  # 50 * 200

    # bc
    def transform_out_disp(self, _in, _out):
        x_scaled = _in["x_scaled"]
        x = self.geo_dim[0] / 2 * (1 + x_scaled) + self.geo_origin[0]
        u, v = (
            x * (x - 1) * _out["u"],
            x * _out["v"],
        )  # x * (x - 1) * (x - 2) * _out["u"], x * (x - 2) * _out["v"]
        return {"u": u, "v": v}

    # force
    def compute_force(self):
        input_pos = {
            "x": paddle.to_tensor([[1.0]], dtype=paddle.get_default_dtype()),
            "y": paddle.to_tensor([[0.0]], dtype=paddle.get_default_dtype()),
        }
        output_pos = self.disp_net(input_pos)
        v = output_pos["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class Bridge2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.0, 0.5)  # (2.0, 0.5)
        super().__init__(2, geo_origin, geo_dim, **cfg)

        long_beam = ppsci.geometry.Rectangle(geo_origin, geo_dim)
        self.geom = {"geo": long_beam}
        self.force = -0.0025
        self.batch_size = (122, 61)  # 50 * 100  # 50 * 200

    # bc
    def transform_out_disp(self, _in, _out):
        x_scaled = _in["x_scaled"]
        x = self.geo_dim[0] / 2 * (1 + x_scaled) + self.geo_origin[0]
        u, v = x * (x - 1) * _out["u"], x * _out["v"]
        return {"u": u, "v": v}

    # force
    def get_force_pos(self):
        sample_num = 400
        input_pos_np = self.geom["geo"].sample_boundary(
            n=sample_num,
            criteria=lambda x, y: y <= self.geo_origin[1] + 1e-3,
        )
        return {
            "x": paddle.to_tensor(input_pos_np["x"], dtype=paddle.get_default_dtype()),
            "y": paddle.full((sample_num, 1), 0.0, dtype=paddle.get_default_dtype()),
        }

    def compute_force(self):
        input_pos = self.get_force_pos()
        output_pos = self.disp_net(input_pos)
        v = output_pos["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class Triangle2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (2.0, 3**0.5)
        super().__init__(2, geo_origin, geo_dim, **cfg)

        triangle = ppsci.geometry.Triangle((0.0, 0.0), (2.0, 0.0), (1.0, 3**0.5))
        self.geom = {"geo": triangle}
        force = 0.0025
        self.force = [
            [-(3**0.5) * 0.5 * force, -0.5 * force],
            [(3**0.5) * 0.5 * force, -0.5 * force],
            [0.0, 1 * force],
        ]
        self.batch_size = (int((3**0.5) * 10000), int((3**0.5) * 10000))
        self.volume = 3**0.5

    # # bc
    # def transform_out_disp(self, _in, _out):
    #     x_scaled, y_scaled = _in["x_scaled"], _in["y_scaled"]
    #     x = self.geo_dim[0] / 2 * (1 + x_scaled) + self.geo_origin[0]
    #     y = self.geo_dim[1] / 2 * (1 + y_scaled) + self.geo_origin[1]
    #     constraint = (x - 1) ** 2 + (y - 1 / 3**0.5) ** 2
    #     u, v = constraint * _out["u"], constraint * _out["v"]
    #     return {"u": u, "v": v}

    # force
    def compute_force(self):
        input_pos = {
            "x": paddle.to_tensor(
                [[0.0], [2.0], [1.0]], dtype=paddle.get_default_dtype()
            ),
            "y": paddle.to_tensor(
                [[0.0], [0.0], [3**0.5]], dtype=paddle.get_default_dtype()
            ),
        }
        output_pos = self.disp_net(input_pos)
        u, v = output_pos["u"], output_pos["v"]
        force = paddle.to_tensor(self.force)
        return -paddle.mean(
            paddle.multiply(force[:, 0], u[:, 0])
            + paddle.multiply(force[:, 1], v[:, 0]),
            keepdim=True,
        )


class TriangleVariants2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (2.0, 3**0.5)
        super().__init__(2, geo_origin, geo_dim, **cfg)

        triangle = ppsci.geometry.Triangle((0.0, 0.0), (2.0, 0.0), (1.0, 3**0.5))
        disk = ppsci.geometry.Disk((1.0, 1 / 3**0.5), 0.1)
        self.geom = {"geo": triangle - disk}
        force = 0.0025
        self.force = [
            [-(3**0.5) * 0.5 * force, -0.5 * force],
            [(3**0.5) * 0.5 * force, -0.5 * force],
            [0.0, 1 * force],
        ]
        self.batch_size = int((3**0.5) * 10000)
        self.volume = 3**0.5 - np.pi * 0.01

    # bc
    def transform_out_disp(self, _in, _out):
        x_scaled, y_scaled = _in["x_scaled"], _in["y_scaled"]
        x = self.geo_dim[0] / 2 * (1 + x_scaled) + self.geo_origin[0]
        y = self.geo_dim[1] / 2 * (1 + y_scaled) + self.geo_origin[1]
        constraint = (x - 1) ** 2 + (y - 1 / 3**0.5) ** 2 - 0.01
        u, v = constraint * _out["u"], constraint * _out["v"]
        return {"u": u, "v": v}

    # force
    def compute_force(self):
        input_pos = {
            "x": paddle.to_tensor(
                [[0.0], [2.0], [1.0]], dtype=paddle.get_default_dtype()
            ),
            "y": paddle.to_tensor(
                [[0.0], [0.0], [3**0.5]], dtype=paddle.get_default_dtype()
            ),
        }
        output_pos = self.disp_net(input_pos)
        u, v = output_pos["u"], output_pos["v"]
        force = paddle.to_tensor(self.force)
        return -paddle.mean(
            paddle.multiply(force[:, 0], u[:, 0])
            + paddle.multiply(force[:, 1], v[:, 0]),
            keepdim=True,
        )


class Custom2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.5, 1.5)
        super().__init__(2, geo_origin, geo_dim, **cfg)

        rec_1 = ppsci.geometry.Rectangle((0.0, 0.0), (1.5, 1.5))
        rec_2 = ppsci.geometry.Rectangle((0.5, 0.5), (1.5, 1.5))
        custom_geo = rec_1 - rec_2
        self.geom = {"geo": custom_geo}

        self.force = -0.0025
        self.batch_size = 12500  # = 150 * 150 - 100 * 100
        self.volume = 1.25

    # bc
    def transform_out_disp(self, _in, _out):
        y_scaled = _in["y_scaled"]
        y = self.geo_dim[0] / 2 * (1 + y_scaled) + self.geo_origin[0]
        u, v = (y - 1.5) * _out["u"], (y - 1.5) * _out["v"]
        return {"u": u, "v": v}

    # force
    def compute_force(self):
        input_pos = {
            "x": paddle.to_tensor([[1.5]], dtype=paddle.get_default_dtype()),
            "y": paddle.to_tensor([[0.5]], dtype=paddle.get_default_dtype()),
        }
        output_pos = self.disp_net(input_pos)
        v = output_pos["v"]
        return -paddle.mean(v * self.force, keepdim=True)
