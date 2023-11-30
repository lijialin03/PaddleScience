from typing import Dict

import numpy as np
import paddle
import paddle.nn.functional as F
from skimage.filters import gaussian

import ppsci
from ppsci.utils import logger


class Problems:
    def __init__(self, cfg, geo_origin, geo_dim):
        self.cfg = cfg
        self.geo_origin = geo_origin
        self.geo_dim = geo_dim
        self.dim = len(geo_dim)
        self.volume = self.comput_volume()

        self.lambda_ = (
            (cfg.NU * cfg.E / ((1 - cfg.NU * cfg.NU)))
            if self.dim == 2
            else cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))
        )
        self.mu = cfg.E / (1 + cfg.NU) if self.dim == 2 else cfg.E / (2 * (1 + cfg.NU))
        self.equation = {
            "EnergyEquation": ppsci.equation.EnergyEquation(
                lambda_=self.lambda_, mu=self.mu, dim=self.dim
            ),
        }

        self.disp_net = None
        self.density_net = None

        self.volume_ratio = cfg.VOLUME_RATIO
        self.alpha = cfg.SIGMOID_ALPHA
        self.exponent = cfg.ENERGY_EXP
        self.vol_penalty_strength = cfg.PENALTY
        self.use_mmse = cfg.USE_MMSE
        self.use_oc = cfg.USE_OC
        self.max_move = cfg.MAX_MOVE
        self.damping_parameter = cfg.DAMPING
        self.filter = cfg.FILTER
        self.sigma = cfg.FILTER_SIGMA

    def comput_volume(self):
        return np.prod(self.geo_dim)

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
        logger.info("In default out transform of disp net")
        return _out

    def transform_out_density(self, _in, _out):
        density = _out["density"]
        offset = np.log(self.volume_ratio / (1.0 - self.volume_ratio))
        densities = F.sigmoid(self.alpha * density + offset)
        return {"densities": densities}

    # functions
    def compute_energy(self, densities, energy):
        energy_densities = paddle.pow(densities, self.exponent) * energy
        return self.volume * paddle.mean(energy_densities, keepdim=True)

    def compute_force(self):
        "Different for each problems because of different force."
        logger.info("In default force compute function")
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
    def compute_target_densities(self, densities_list, sensitivities_list):
        if self.use_oc:
            return self.compute_oc_multi_batch(densities_list, sensitivities_list)
        else:
            return self.compute_target_densities_gradient_descent(
                densities_list, sensitivities_list
            )

    def compute_oc_multi_batch(self, old_densities, sensitivities):
        target_volume = self.volume_ratio * self.volume
        logger.info(f"target_volume: {target_volume}")

        lagrange_lower_estimate = 0
        lagrange_upper_estimate = 1e9
        conv_threshold = 1e-3
        total_samples = len(old_densities) * old_densities[0].shape[0]
        dv = self.volume / total_samples

        density_lower_bound = [
            paddle.maximum(paddle.to_tensor(0.0), od - self.max_move)
            for od in old_densities
        ]
        density_upper_bound = [
            paddle.minimum(paddle.to_tensor(1.0), od + self.max_move)
            for od in old_densities
        ]

        while (lagrange_upper_estimate - lagrange_lower_estimate) / (
            lagrange_lower_estimate + lagrange_upper_estimate
        ) > conv_threshold:
            lagrange_current = 0.5 * (lagrange_upper_estimate + lagrange_lower_estimate)

            target_densities_div = [
                paddle.divide(
                    di,
                    paddle.to_tensor(
                        -dv * lagrange_current, dtype=paddle.get_default_dtype()
                    ),
                )
                for di in sensitivities
            ]
            di_div_max = float(paddle.max(paddle.concat(target_densities_div)))
            # di_div_mean = float(paddle.mean(paddle.concat(target_densities_div)))
            if np.isinf(di_div_max):
                logger.info("Warning! target_densities_div is nan")
                exit()
            # else:
            #     print("### di_div_mean", di_div_mean)

            target_densities_mul = [
                paddle.multiply(
                    old_densities[i],
                    paddle.pow(target_densities_div[i], self.damping_parameter),
                )
                for i in range(len(old_densities))
            ]
            # di_mul_mean = float(paddle.mean(paddle.concat(target_densities_mul)))
            # print("di_mul_mean", di_mul_mean)

            target_densities_clip = [
                paddle.maximum(
                    density_lower_bound[i],
                    paddle.minimum(density_upper_bound[i], target_densities_mul[i]),
                )
                for i in range(len(old_densities))
            ]
            # di_clip_mean = float(paddle.mean(paddle.concat(target_densities_clip)))
            # print("di_clip_mean", di_clip_mean)

            new_volume = self.volume * np.mean(
                [paddle.mean(di) for di in target_densities_clip]
            )

            if new_volume > target_volume:
                lagrange_lower_estimate = lagrange_current
            else:
                lagrange_upper_estimate = lagrange_current
            # if (lagrange_lower_estimate + lagrange_upper_estimate) < 1e-9:
            #     break
        logger.info(f"new_volume: {new_volume}")
        return target_densities_clip

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

    # filter
    def gaussian_filter(self, sensitivities):
        sensitivities = paddle.reshape(sensitivities, self.batch_size)
        sensitivities_blur = gaussian(sensitivities.numpy(), self.sigma).reshape(
            [-1, 1]
        )
        return paddle.to_tensor(sensitivities_blur, dtype=paddle.get_default_dtype())

    # loss functions
    def disp_loss_func(self, output_dict, label_dict=None, weight_dict={}):
        input_dict = {"x": output_dict["x"], "y": output_dict["y"]}
        if self.dim == 3:
            input_dict["z"] = output_dict["z"]
        densities = self.density_net(input_dict)["densities"].detach().clone()
        energy = output_dict["energy"]
        loss_energy = self.compute_energy(densities, energy)
        loss_force = self.compute_force()
        return loss_energy + loss_force

    def density_loss_func(
        self,
        output_dicts_list,
        label_dicts_list=None,
        weight_dicts_list=None,
    ):
        loss_list = []
        densities_list = []
        sensitivities_list = []
        input_dicts_list = []
        if isinstance(output_dicts_list, Dict):
            output_dicts_list = [[output_dicts_list]]
        for output_dict in output_dicts_list:
            densities = output_dict[0]["densities"]
            energy = output_dict[0]["energy"]
            ppsci.autodiff.clear()

            loss_energy = self.compute_energy(densities, energy)
            loss = -loss_energy
            if not self.use_oc:
                loss_penalty = self.compute_penalty(densities)
                loss += loss_penalty
            loss_list.append(loss)

            sensitivities = paddle.grad(loss, densities)[0].detach().clone()
            densities = densities.detach().clone()

            # filter
            if self.filter == "Gaussian":
                sensitivities = self.gaussian_filter(sensitivities)

            densities_list.append(densities)
            sensitivities_list.append(sensitivities)
            ppsci.autodiff.clear()

            if self.use_mmse:
                input_dict = {"x": output_dict[0]["x"], "y": output_dict[0]["y"]}
                if self.dim == 3:
                    input_dict["z"] = output_dict[0]["z"]
                input_dicts_list.append(input_dict)

        if not self.use_mmse:
            return loss_list[0]
        else:
            target_densities_list = self.compute_target_densities(
                densities_list, sensitivities_list
            )
            logger.info(
                f"target density: {np.mean([td.numpy().mean() for td in target_densities_list])}"
            )

            def oc_loss_func(model, i):
                densities_i = model(input_dicts_list[i])["densities"]
                loss = F.mse_loss(densities_i, target_densities_list[i], "mean")
                return loss

            return oc_loss_func

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
        super().__init__(cfg, geo_origin, geo_dim)

        beam = ppsci.geometry.Rectangle((0.0, 0.0), (1.5, 0.5))
        self.geom = {"geo": beam}
        self.force = -0.0025
        self.mirror = None
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
        output_force = self.disp_net(input_pos)
        v = output_force["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class Distributed2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.5, 0.5)
        super().__init__(cfg, geo_origin, geo_dim)

        beam = ppsci.geometry.Rectangle((0.0, 0.0), (1.5, 0.5))
        self.geom = {"geo": beam}
        self.force = -0.0025
        self.mirror = None
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
        output_force = self.disp_net(input_pos)
        v = output_force["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class LongBeam2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.0, 0.5)  # (2.0, 0.5)
        super().__init__(cfg, geo_origin, geo_dim)

        long_beam = ppsci.geometry.Rectangle(geo_origin, geo_dim)
        self.geom = {"geo": long_beam}
        self.force = -0.0025
        self.mirror = [True, False]
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
        output_force = self.disp_net(input_pos)
        v = output_force["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class Bridge2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.0, 0.5)  # (2.0, 0.5)
        super().__init__(cfg, geo_origin, geo_dim)

        long_beam = ppsci.geometry.Rectangle(geo_origin, geo_dim)
        self.geom = {"geo": long_beam}
        self.force = -0.0025
        self.mirror = [True, False]
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
        output_force = self.disp_net(input_pos)
        v = output_force["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class Triangle2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (2.0, 3**0.5)
        super().__init__(cfg, geo_origin, geo_dim)

        triangle = ppsci.geometry.Triangle((0.0, 0.0), (2.0, 0.0), (1.0, 3**0.5))
        self.geom = {"geo": triangle}
        force = 0.0025
        self.mirror = None
        self.force = [
            [-(3**0.5) * 0.5 * force, -0.5 * force],
            [(3**0.5) * 0.5 * force, -0.5 * force],
            [0.0, 1 * force],
        ]
        self.batch_size = int((3**0.5) * 10000)
        self.volume = 3**0.5

    # bc
    def transform_out_disp(self, _in, _out):
        x_scaled, y_scaled = _in["x_scaled"], _in["y_scaled"]
        x = self.geo_dim[0] / 2 * (1 + x_scaled) + self.geo_origin[0]
        y = self.geo_dim[1] / 2 * (1 + y_scaled) + self.geo_origin[1]
        constraint = (x - 1) * (y - 1 / 3**0.5)
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
        output_force = self.disp_net(input_pos)
        u, v = output_force["u"], output_force["v"]
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
        super().__init__(cfg, geo_origin, geo_dim)

        triangle = ppsci.geometry.Triangle((0.0, 0.0), (2.0, 0.0), (1.0, 3**0.5))
        disk = ppsci.geometry.Disk((1.0, 1 / 3**0.5), 0.1)
        self.geom = {"geo": triangle - disk}
        force = 0.0025
        self.mirror = None
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
        output_force = self.disp_net(input_pos)
        u, v = output_force["u"], output_force["v"]
        force = paddle.to_tensor(self.force)
        return -paddle.mean(
            paddle.multiply(force[:, 0], u[:, 0])
            + paddle.multiply(force[:, 1], v[:, 0]),
            keepdim=True,
        )


class LShape2D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.5, 1.5)
        super().__init__(cfg, geo_origin, geo_dim)

        rec_1 = ppsci.geometry.Rectangle((0.0, 0.0), (1.5, 1.5))
        rec_2 = ppsci.geometry.Rectangle((0.5, 0.5), (1.5, 1.5))
        custom_geo = rec_1 - rec_2
        self.geom = {"geo": custom_geo}

        self.force = -0.0025
        self.mirror = None
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
        output_force = self.disp_net(input_pos)
        v = output_force["v"]
        return -paddle.mean(v * self.force, keepdim=True)


class Beam3D(Problems):
    def __init__(self, cfg):
        geo_origin = (0.0, 0.0, 0.0)
        geo_dim = (1.0, 0.5, 0.25)
        super().__init__(cfg, geo_origin, geo_dim)

        beam = ppsci.geometry.Cuboid(geo_origin, geo_dim)
        self.geom = {"geo": beam}
        self.force = (0.0, -0.0005, 0.0)
        self.mirror = [False, False, True]
        self.batch_size = (40, 20, 10)
        self.input_force = self.get_input_force(sample_num=400)

    # bc
    def transform_out_disp(self, _in, _out):
        x_scaled, z_scaled = _in["x_scaled"], _in["z_scaled"]
        x = self.geo_dim[0] / 2 * (1 + x_scaled) + self.geo_origin[0]
        z = self.geo_dim[2] / 2 * (1 + z_scaled) + self.geo_origin[2]
        u, v, w = x * _out["u"], x * _out["v"], x * (z - self.geo_dim[2]) * _out["w"]
        return {"u": u, "v": v, "w": w}

    # force
    def get_input_force(self, sample_num):
        input_z_np = np.linspace(
            start=self.geo_origin[2],
            stop=self.geo_dim[2],
            num=sample_num,
            endpoint=True,
        ).reshape(sample_num, 1)
        input_force = {
            "x": paddle.full([sample_num, 1], 1.0, dtype=paddle.get_default_dtype()),
            "y": paddle.full([sample_num, 1], 0.0, dtype=paddle.get_default_dtype()),
            "z": paddle.to_tensor(input_z_np, dtype=paddle.get_default_dtype()),
        }
        return input_force

    def compute_force(self):
        force_volume = self.geo_dim[2] - self.geo_origin[2]
        output_force = self.disp_net(self.input_force)
        u, v, w = output_force["u"], output_force["v"], output_force["w"]
        return -force_volume * paddle.mean(
            (u * self.force[0] + v * self.force[1] + w * self.force[2]), keepdim=True
        )
