import numpy as np
import paddle
import paddle.nn.functional as F

import ppsci


class Problems:
    def __init__(self, dim, geo_origin, geo_dim):
        self.dim = dim
        self.geo_origin = geo_origin
        self.geo_dim = geo_dim
        self.volume = self.get_volume()

        self.volume_ratio = 0.5
        self.alpha = 5.0
        self.exponent = 3.0  # TODO: trainable parameter
        self.vol_penalty_strength = 10.0

        self.nu = 0.3
        self.E = 1.0
        lambda_ = (
            (self.nu * self.E / ((1 - self.nu * self.nu)))
            if dim == 2
            else self.nu * self.E / ((1 + self.nu) * (1 - 2 * self.nu))
        )
        mu = self.E / (1 + self.nu) if dim == 2 else self.E / (2 * (1 + self.nu))
        self.equation = {
            "EnergyEquation": ppsci.equation.EnergyEquation(
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
    def compute_energy(self, densities, energy):
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

    # loss functions
    def disp_loss_func(
        self, output_dict, label_dict=None, weight_dict={}, input_dict=None
    ):
        densities = self.density_net(input_dict)["densities"]
        densities = densities.detach().clone()
        energy = (
            output_dict["energy_xy"] if self.dim == 2 else output_dict["energy_xyz"]
        )
        loss_energy = self.compute_energy(densities, energy)
        loss_force = self.compute_force()
        # print("loss_energy:", "%.3e" % float(loss_energy))
        # print("loss_force:", "%.3e" % float(loss_force))
        return loss_energy + loss_force

    def density_loss_func(
        self, output_dict, label_dict=None, weight_dict={}, input_dict=None
    ):
        densities = output_dict["densities"]
        energy = self.equation["EnergyEquation"].equations["energy_xy"](
            {**self.disp_net(input_dict), **input_dict}
        )
        energy = energy.detach().clone()

        loss_energy = self.compute_energy(densities, energy)
        loss_penalty = self.compute_penalty(densities)
        # print("### loss_energy", "%.3e" % loss_energy)
        # print("### loss_penalty", "%.3e" % loss_penalty)
        return loss_energy + loss_penalty

    def density_metric_func(self, output_dict, *args):
        density = output_dict["densities"]
        print("mean:", float(paddle.mean(density)))
        print("max:", float(paddle.max(density)))
        print("min:", float(paddle.min(density)))
        metric_dict = {"densities": density.mean() - self.volume_ratio}
        return metric_dict


class Beam2D(Problems):
    def __init__(self):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.5, 0.5)
        super().__init__(2, geo_origin, geo_dim)

        beam = ppsci.geometry.Rectangle((0.0, 0.0), (1.5, 0.5))
        self.geom = {"geo": beam}
        self.force = -0.0025
        self.batch_size = 50 * 150

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
    def __init__(self):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.5, 0.5)
        super().__init__(2, geo_origin, geo_dim)

        beam = ppsci.geometry.Rectangle((0.0, 0.0), (1.5, 0.5))
        self.geom = {"geo": beam}
        self.force = -0.0025
        self.batch_size = 50 * 150

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
    def __init__(self):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.0, 0.5)  # (2.0, 0.5)
        super().__init__(2, geo_origin, geo_dim)

        long_beam = ppsci.geometry.Rectangle(geo_origin, geo_dim)
        self.geom = {"geo": long_beam}
        self.force = -0.0025
        self.batch_size = 61 * 122  # 50 * 100  # 50 * 200

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
    def __init__(self):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.0, 0.5)  # (2.0, 0.5)
        super().__init__(2, geo_origin, geo_dim)

        long_beam = ppsci.geometry.Rectangle(geo_origin, geo_dim)
        self.geom = {"geo": long_beam}
        self.force = -0.0025
        self.batch_size = 61 * 122  # 50 * 100  # 50 * 200

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
    def __init__(self):
        geo_origin = (0.0, 0.0)
        geo_dim = (2.0, 3**0.5)
        super().__init__(2, geo_origin, geo_dim)

        triangle = ppsci.geometry.Triangle((0.0, 0.0), (2.0, 0.0), (1.0, 3**0.5))
        self.geom = {"geo": triangle}
        force = 0.0025
        self.force = [
            [-(3**0.5) * 0.5 * force, -0.5 * force],
            [(3**0.5) * 0.5 * force, -0.5 * force],
            [0.0, 1 * force],
        ]
        self.batch_size = int((3**0.5) * 10000)
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
    def __init__(self):
        geo_origin = (0.0, 0.0)
        geo_dim = (2.0, 3**0.5)
        super().__init__(2, geo_origin, geo_dim)

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
