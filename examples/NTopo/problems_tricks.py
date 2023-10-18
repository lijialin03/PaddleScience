import numpy as np
import paddle
import paddle.nn.functional as F

import ppsci
from ppsci.utils import logger


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

        self.use_mmse = True
        self.use_oc = True
        self.max_move = 0.2
        self.damping_parameter = 0.5  # 阻尼参数
        self.filter = "sensitivity"
        self.filter_radius = 2.0

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
            # print("### ", lagrange_lower_estimate + lagrange_upper_estimate)
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

    # # fliter
    # def apply_sensitivity_filter(
    #     self, sample_positions, densities, sensitivities, n_samples, domain, radius
    # ):
    #     dim = 2
    #     gamma = 1e-3

    #     cell_width = (self.geo_dim[0] - self.geo_origin[0]) / n_samples[0]
    #     grads = sensitivities

    #     radius_space = radius * cell_width
    #     filter_size = 2 * round(radius) + 1
    #     density_patches = tf.reshape(densities, [1, n_samples[1], n_samples[0], 1])
    #     density_patches = pad_border(density_patches, filter_size)
    #     density_patches = tf.image.extract_patches(
    #         density_patches,
    #         sizes=[1, filter_size, filter_size, 1],
    #         strides=[1, 1, 1, 1],
    #         rates=[1, 1, 1, 1],
    #         padding="VALID",
    #     )

    #     sensitivity_patches = tf.reshape(
    #         sensitivities, [1, n_samples[1], n_samples[0], 1]
    #     )
    #     sensitivity_patches = pad_border(sensitivity_patches, filter_size)
    #     sensitivity_patches = tf.image.extract_patches(
    #         sensitivity_patches,
    #         sizes=[1, filter_size, filter_size, 1],
    #         strides=[1, 1, 1, 1],
    #         rates=[1, 1, 1, 1],
    #         padding="VALID",
    #     )

    #     sample_positions = tf.reshape(
    #         sample_positions, [1, n_samples[1], n_samples[0], dim]
    #     )
    #     # we pad such that influence is basically 0
    #     sample_patches = pad_positions_constant(sample_positions, filter_size)
    #     sample_patches = tf.image.extract_patches(
    #         sample_patches,
    #         sizes=[1, filter_size, filter_size, 1],
    #         strides=[1, 1, 1, 1],
    #         rates=[1, 1, 1, 1],
    #         padding="VALID",
    #     )
    #     # sample_patches.shape is now [1, rows, cols, filter_size ** 2 * * dim]

    #     diff = tf.reshape(
    #         sample_patches,
    #         [1, n_samples[1], n_samples[0], filter_size * filter_size, dim],
    #     ) - tf.reshape(sample_positions, [1, n_samples[1], n_samples[0], 1, dim])
    #     # [1, n_samples[1], n_samples[0], filter_size ** 2]
    #     dists = tf.math.sqrt(tf.math.reduce_sum(diff * diff, axis=4))

    #     # [1, n_samples[1], n_samples[0], filter_size ** 2]
    #     Hei = tf.math.maximum(0.0, radius_space - dists)
    #     # [1, n_samples[1], n_samples[0], filter_size ** 2]
    #     Heixic = Hei * density_patches * sensitivity_patches
    #     sum_Heixic = tf.math.reduce_sum(Heixic, axis=3)
    #     sum_Hei = tf.math.reduce_sum(Hei, axis=3)
    #     old_densities_r = tf.reshape(old_densities, [1, n_samples[1], n_samples[0]])
    #     assert len(sum_Hei.shape) == len(old_densities_r.shape)
    #     div = tf.math.maximum(gamma, old_densities_r) * sum_Hei
    #     grads = sum_Heixic / div

    #     grads = tf.reshape(grads, (-1, 1))
    #     return grads

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
        self,
        output_dicts_list,
        label_dicts_list=None,
        weight_dicts_list=None,
        input_dicts_list=None,
    ):
        loss_list = []
        densities_list = []
        sensitivities_list = []
        for i, output_dict in enumerate(output_dicts_list):
            densities = output_dict[0]["densities"]
            energy_xy = self.equation["EnergyEquation"].equations["energy_xy"](
                {**self.disp_net(input_dicts_list[i][0]), **input_dicts_list[i][0]}
            )
            energy_xy = energy_xy.detach().clone()

            loss_energy = self.compute_energy(densities, energy_xy)
            # print("### loss_energy", "%.3e" % loss_energy)
            loss = -loss_energy
            if not self.use_oc:
                loss_penalty = self.compute_penalty(densities)
                # print("### loss_penalty", "%.3e" % loss_penalty)
                loss += loss_penalty
            loss_list.append(loss)

            sensitivities = paddle.grad(loss, densities)[0]
            # # add fliter
            # if filter == "sensitivity":
            #     sensitivities_i = self.apply_sensitivity_filter(
            #         input_samples,
            #         densities,
            #         sensitivities_i,
            #         n_samples=n_opt_samples,
            #         domain=problem.domain,
            #         dim=problem.dim,
            #         radius=filter_radius,
            #     )

            densities_list.append(densities)
            sensitivities_list.append(sensitivities)
            ppsci.autodiff.clear()

        if not self.use_mmse:
            return loss_list
        else:
            target_densities_list = self.compute_target_densities(
                densities_list, sensitivities_list
            )
            print(
                "use_mmse", np.mean([td.numpy().mean() for td in target_densities_list])
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


class Custom2D_1(Problems):
    def __init__(self):
        geo_origin = (0.0, 0.0)
        geo_dim = (1.5, 1.5)
        super().__init__(2, geo_origin, geo_dim)

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
