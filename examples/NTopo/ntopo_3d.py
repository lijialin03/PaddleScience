import os

import cv2
import model as model_module
import numpy as np
import paddle
import paddle.nn.functional as F

import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def get_force_pos(geo):
    sample_num = 400
    input_pos_np = geo.sample_boundary(
        n=sample_num,
        criteria=lambda x, y, z: np.logical_and(
            x >= BEAM_ORIGIN[0] + BEAM_DIM[0] - 1e-3,
            y <= BEAM_ORIGIN[1] + 1e-3,
        ),
    )
    pos = {
        "x": paddle.full(
            input_pos_np["x"].shape, 1.0, dtype=paddle.get_default_dtype()
        ),
        "y": paddle.full(
            input_pos_np["y"].shape, 0.0, dtype=paddle.get_default_dtype()
        ),
        "z": paddle.to_tensor(input_pos_np["z"], dtype=paddle.get_default_dtype()),
    }
    return pos


def comput_energy(densities, energy_xyz):
    exponent = 3.0  # TODO: trainable parameter
    energy_densities = paddle.pow(densities, exponent) * energy_xyz
    return volume * paddle.mean(energy_densities, keepdim=True)


def comput_force():
    force = [0.0, -0.0005, 0.0]
    force_volume = 0.25
    # input_pos = {
    #     "x": paddle.to_tensor([[0], [0.5], [1.0]], dtype=paddle.get_default_dtype()),
    #     "y": paddle.to_tensor([[0], [0.25], [0.5]], dtype=paddle.get_default_dtype()),
    #     "z": paddle.to_tensor([[0], [0.125], [0.25]], dtype=paddle.get_default_dtype()),
    # }
    output_pos = disp_net(input_pos)
    u, v, w = output_pos["u"], output_pos["v"], output_pos["w"]
    # print((u * force[0],v * force[1], w * force[2]))
    return -force_volume * paddle.mean(
        (u * force[0] + v * force[1] + w * force[2]), keepdim=True
    )


def comput_penalty(densities):
    vol_penalty_strength = 10.0
    target_volume = volume_ratio * volume
    volume_estimate = volume * paddle.mean(densities, keepdim=True)
    return (
        vol_penalty_strength
        * (volume_estimate - target_volume)
        * (volume_estimate - target_volume)
        / target_volume
    )


# def apply_flitter(loss, densities):
#     sensitivities_i = paddle.grad(loss, densities)
#     if filter == "sensitivity":
#         sensitivities_i = apply_sensitivity_filter()

#     old_densities.append(densities)
#     sensitivities.append(sensitivities_i)


def compute_target_densities(old_densities, sensitivities, use_oc):
    # print("####", type(old_densities), type(sensitivities))
    # if use_oc:
    #     target_densities = compute_oc_multi_batch()
    # else:
    projected_sensitivities = (
        paddle.maximum(
            paddle.to_tensor(0.0),
            paddle.minimum(paddle.to_tensor(1.0), old_densities - sensitivities),
        )
        - old_densities
    )

    step_size = 0.05 / paddle.mean(paddle.abs(projected_sensitivities), keepdim=True)
    return old_densities - step_size * sensitivities


def disp_loss_func(output_dict, label_dict=None, weight_dict={}, input_dict=None):
    densities = density_net(input_dict)["densities"]
    densities.stop_gradient = True
    # densities = output_dict["densities"]
    energy_xyz = output_dict["energy_xyz"]
    loss_energy = comput_energy(densities, energy_xyz)
    loss_force = comput_force()
    # print("loss_energy:", "%.3e" % float(loss_energy))
    # print("loss_force:", "%.3e" % float(loss_force))
    return loss_energy + loss_force


def density_loss_func(output_dict, label_dict=None, weight_dict={}, input_dict=None):
    densities = output_dict["densities"]
    # energy_xyz = output_dict["energy_xyz"]
    energy_xyz = equation["EnergyEquation"].equations["energy_xyz"](
        {**disp_net(input_dict), **input_dict}
    )
    energy_xyz.stop_gradient = True

    loss_energy = comput_energy(densities, energy_xyz)
    # print("### loss_energy", "%.3e" % loss_energy)
    losses = -loss_energy
    if not use_oc:
        loss_penalty = comput_penalty(densities)
        # print("### loss_penalty", "%.3e" % loss_penalty)
        losses += loss_penalty
    if not use_mmse:
        return losses
    else:
        sensitivities = paddle.grad(losses, densities)[0]
        target_densities = compute_target_densities(densities, sensitivities, use_oc)
        return F.mse_loss(densities, target_densities, "mean")


def density_metric_func(output_dict, *args):
    density = output_dict["densities"]
    print("mean:", float(paddle.mean(density)))
    print("max:", float(paddle.max(density)))
    print("min:", float(paddle.min(density)))
    metric_dict = {"densities": density.mean() - volume_ratio}
    return metric_dict


def save_img(pred_input_dict, epoch_id):
    os.makedirs(f"{OUTPUT_DIR}visual", exist_ok=True)

    pred_dict = solver_density.predict(
        pred_input_dict,
        {"densities": lambda out: out["densities"]},
        batch_size=BATCH_SIZE_PRED,
        no_grad=False,
    )

    pred_img = np.ones((50, 150, 1)) * 255
    for i in range(BATCH_SIZE_PRED):
        idx_x, idx_y = (
            float(pred_input_dict["x"][i]) * 150,
            float(pred_input_dict["y"][i]) * 150,
        )
        idx_x, idx_y = np.clip(int(idx_x), 0, 149), np.clip(int(idx_y), 0, 49)
        pred_img[idx_y, idx_x, 0] = (1 - float(pred_dict["densities"][i])) * 255

    cv2.imwrite(
        f"{OUTPUT_DIR}visual/density_{str(epoch_id)}.png",
        pred_img,
    )


# def density_custom_gradients(loss):
#     for name, tensor in solver.model.model_list[1].named_parameters():
#         # print(name, tensor.grad[0][0])
#         tensor.grad *= -1
#         # print(name, tensor.grad[0][0])
#         # break


if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = "./output_ntopo_3d/" if args.output_dir is None else args.output_dir

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # Specify parameters
    NU = 0.3  # 泊松比
    E = 1.0  # 弹性模量
    LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))  # lambda 拉梅常数之一
    MU = E / (2 * (1 + NU))  # mu 拉梅常数之一
    # T = -0.0005  # 牵引力大小

    # set equation
    # equation = {
    #     "LinearElasticity": ppsci.equation.LinearElasticity(
    #         E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=3
    #     )
    # }
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity_v2(
            param_dict={"lambda_": LAMBDA_, "mu": MU},
            dim=3,
        ),
        "EnergyEquation": ppsci.equation.EnergyEquation(
            param_dict={"lambda_": LAMBDA_, "mu": MU},
            dim=3,
        ),
    }

    # set geometry
    BEAM_ORIGIN = (0.0, 0.0, 0.0)
    BEAM_DIM = (1.0, 0.5, 0.25)
    beam = ppsci.geometry.Cuboid(BEAM_ORIGIN, BEAM_DIM)
    # geometry bool operation
    geo = beam
    geom = {"geo": geo}

    def comput_volume(dim):
        return dim[0] * dim[1] * dim[2]

    volume = comput_volume(BEAM_DIM)
    volume_ratio = 0.5

    # set training hyper-parameters
    # MARK
    loss_str = "mean"
    plt_name = "vis"
    use_para = False
    LR = 3e-4

    use_mmse = True
    filter = "sensitivity"
    # oc setting
    use_oc = False
    max_move = 0.2
    damping_parameter = 0.5  # 阻尼参数

    EPOCHS = 1  # times for for-loop
    EPOCHS_DISP = EPOCHS_DENSITY = 1
    ITERS_PER_EPOCH = 1
    ITERS_PER_EPOCH_DISP = 1
    ITERS_PER_EPOCH_DENSITY = 1  # times for n_opt_batch
    N_OPT_BATCHES = 10

    # set model
    input_keys = (
        "x_scaled",
        "y_scaled",
        "z_scaled",
        "sin_x_scaled",
        "sin_y_scaled",
        "sin_z_scaled",
    )
    disp_net = model_module.DenseSIRENModel(input_keys, ("u", "v", "w"), 6, 180, 0.001)
    density_net = model_module.DenseSIRENModel(input_keys, ("density",), 6, 180, 0.001)

    # input transform
    def transform_in(_in):
        x, y, z = _in["x"], _in["y"], _in["z"]
        x_scaled = 2.0 / BEAM_DIM[0] * x + (-1.0 - 2.0 * BEAM_ORIGIN[0] / BEAM_DIM[0])
        y_scaled = 2.0 / BEAM_DIM[1] * y + (-1.0 - 2.0 * BEAM_ORIGIN[1] / BEAM_DIM[1])
        z_scaled = 2.0 / BEAM_DIM[2] * z + (-1.0 - 2.0 * BEAM_ORIGIN[2] / BEAM_DIM[2])

        sin_x_scaled, sin_y_scaled, sin_z_scaled = (
            paddle.sin(x_scaled),
            paddle.sin(y_scaled),
            paddle.sin(z_scaled),
        )
        return {
            "x_scaled": x_scaled,
            "y_scaled": y_scaled,
            "z_scaled": z_scaled,
            "sin_x_scaled": sin_x_scaled,
            "sin_y_scaled": sin_y_scaled,
            "sin_z_scaled": sin_z_scaled,
        }

    def transform_out_disp(_in, _out):
        x_scaled, z_scaled = _in["x_scaled"], _in["z_scaled"]
        x = BEAM_DIM[0] / 2 * (1 + x_scaled) + BEAM_ORIGIN[0]
        z = BEAM_DIM[2] / 2 * (1 + z_scaled) + BEAM_ORIGIN[2]
        # print("### bc_output[0]", x)
        # print("### bc_output[2]", x * (z - BEAM_DIM[2]))
        u, v, w = x * _out["u"], x * _out["v"], x * (z - BEAM_DIM[2]) * _out["w"]
        return {"u": u, "v": v, "w": w}

    def transform_out_density(_in, _out):
        density = _out["density"]
        # volume_ratio = 0.5
        alpha = 5.0
        offset = np.log(volume_ratio / (1.0 - volume_ratio))
        densities = F.sigmoid(alpha * density + offset)
        return {"densities": densities}

    disp_net.register_input_transform(transform_in)
    disp_net.register_output_transform(transform_out_disp)

    density_net.register_input_transform(transform_in)
    density_net.register_output_transform(transform_out_density)

    # wrap to a model_list
    # model = ppsci.arch.ModelList((disp_net, density_net))

    # set optimizer
    optimizer_disp = ppsci.optimizer.Adam(learning_rate=LR, beta2=0.99, epsilon=1e-7)(
        (disp_net,)
    )
    optimizer_density = ppsci.optimizer.Adam(
        learning_rate=LR, beta1=0.8, beta2=0.9, epsilon=1e-7
    )((density_net,))

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "num_workers": 1,
    }
    batch_size = {"bs": 80 * 40 * 20}

    # debug
    pt_sup = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "x": np.array(
                        [[0], [0.5], [1.0]], dtype=paddle.get_default_dtype()
                    ),
                    "y": np.array(
                        [[0], [0.25], [0.5]], dtype=paddle.get_default_dtype()
                    ),
                    "z": np.array(
                        [[0], [0.125], [0.25]], dtype=paddle.get_default_dtype()
                    ),
                },
                "label": {
                    "u": np.array([[0], [0], [0]], dtype=paddle.get_default_dtype()),
                    "v": np.array([[0], [0], [0]], dtype=paddle.get_default_dtype()),
                    "w": np.array([[0], [0], [0]], dtype=paddle.get_default_dtype()),
                    "energy_xyz": np.array(
                        [[0], [0], [0]], dtype=paddle.get_default_dtype()
                    ),
                },
            },
            "batch_size": 3,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": True,
                "shuffle": False,
            },
            "num_workers": 1,
        },
        # ppsci.loss.MSELoss("mean"),
        ppsci.loss.FunctionalLoss(disp_loss_func),
        # ppsci.loss.FunctionalLoss(density_loss_func),
        equation["EnergyEquation"].equations,
        name="debug",
    )
    # force_sup = ppsci.constraint.SupervisedConstraint(
    #     {
    #         "dataset": {
    #             "name": "NamedArrayDataset",
    #             "input": {
    #                 "x": np.array([[1.5]], dtype=paddle.get_default_dtype()),
    #                 "y": np.array([[0]], dtype=paddle.get_default_dtype()),
    #             },
    #             "label": {
    #                 "u": np.array([[0]], dtype=paddle.get_default_dtype()),
    #                 "v": np.array([[0]], dtype=paddle.get_default_dtype()),
    #             },
    #         },
    #         "batch_size": 1,
    #         "sampler": {
    #             "name": "BatchSampler",
    #             "drop_last": True,
    #             "shuffle": True,
    #         },
    #         "num_workers": 1,
    #     },
    #     ppsci.loss.FunctionalLoss(disp_force_loss_func),
    #     name="debug",
    # )
    pt_sup_2 = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {
                "name": "NamedArrayDataset",
                "input": {
                    "x": np.array(
                        [[0], [0.5], [1.0]], dtype=paddle.get_default_dtype()
                    ),
                    "y": np.array(
                        [[0], [0.25], [0.5]], dtype=paddle.get_default_dtype()
                    ),
                    "z": np.array(
                        [[0], [0.125], [0.25]], dtype=paddle.get_default_dtype()
                    ),
                },
                "label": {
                    "density": np.array(
                        [[0], [0], [0]], dtype=paddle.get_default_dtype()
                    ),
                },
            },
            "batch_size": 3,
            "sampler": {
                "name": "BatchSampler",
                "drop_last": True,
                "shuffle": False,
            },
            "num_workers": 1,
        },
        ppsci.loss.FunctionalLoss(density_loss_func),
        equation["EnergyEquation"].equations,
        name="debug",
    )

    constraint_test = {pt_sup_2.name: pt_sup_2}

    # set constraint
    # bc_left_disp = ppsci.constraint.BoundaryConstraint(
    #     {"u": lambda out: out["u"], "v": lambda out: out["v"]},
    #     {"u": 0, "v": 0},
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": batch_size["bs"]},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y: x == BEAM_ORIGIN[0],
    #     # weight_dict={"u": 1, "v": 1},
    #     name="BC_LEFT_DISP",
    # )
    # bc_right_corner_disp = ppsci.constraint.BoundaryConstraint(
    #     equation["EnergyEquation"].equations,
    #     {"energy_xy": 0},
    #     # {"traction_x": 0, "traction_y": T},
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": batch_size["bs"]},
    #     # ppsci.loss.MSELoss(loss_str),
    #     ppsci.loss.FunctionalLoss(disp_force_loss_func),
    #     criteria=lambda x, y: np.logical_and(
    #         x == BEAM_ORIGIN[1], y > BEAM_ORIGIN[0], y < BEAM_ORIGIN[0] + 1e-3
    #     ),
    #     name="BC_RIGHT_CORNER_DISP",
    # )
    interior_disp = ppsci.constraint.InteriorConstraint(
        equation["EnergyEquation"].equations,
        {"energy_xyz": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["bs"]},
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(disp_loss_func),
        name="INTERIOR_DISP",
    )

    interior_density = ppsci.constraint.InteriorConstraint(
        {
            "densities": lambda out: out["densities"],
        },
        {"densities": 0},
        geom["geo"],
        {
            **train_dataloader_cfg,
            "batch_size": batch_size["bs"],
            "iters_per_epoch": ITERS_PER_EPOCH_DENSITY,
        },
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(density_loss_func),
        name="INTERIOR_DENSITY",
    )

    # re-assign to ITERS_PER_EPOCH
    # if use_para:
    # ITERS_PER_EPOCH_DISP = len(bc_left_disp.data_loader)
    # ITERS_PER_EPOCH_DENSITY = len(interior_density.data_loader)

    # wrap constraints together
    constraint_disp = {
        # bc_left_disp.name: bc_left_disp,
        # bc_right_corner_disp.name: bc_right_corner_disp,
        interior_disp.name: interior_disp,
    }
    constraint_density = {interior_density.name: interior_density}

    # set visualizer(optional)
    # add inferencer data
    BATCH_SIZE_PRED = batch_size["bs"]
    pred_input_dict = geom["geo"].sample_interior(BATCH_SIZE_PRED)
    pred_keys = list(pred_input_dict.keys())
    for key in pred_keys:
        if key not in ("x", "y", "z"):
            pred_input_dict.pop(key)

    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": pred_input_dict,
            "label": {
                "densities": np.zeros_like(
                    pred_input_dict["x"], dtype=paddle.get_default_dtype()
                ),
            },
        },
        "batch_size": batch_size["bs"],
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
        "num_workers": 1,
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        ppsci.loss.MSELoss("mean"),
        metric={"mean": ppsci.metric.FunctionalMetric(density_metric_func)},
        name="eval",
    )
    validator_density = {sup_validator.name: sup_validator}

    visualizer_disp = {
        "vis": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
            },
            prefix=plt_name + "_disp",
        ),
    }
    visualizer_density = {
        "vis": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "density": lambda out: out["densities"],
            },
            prefix=plt_name + "_density",
        ),
    }

    # initialize solver
    solver_disp = ppsci.solver.Solver(
        model=disp_net,
        constraint=constraint_disp,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer_disp,
        epochs=EPOCHS_DISP,
        iters_per_epoch=ITERS_PER_EPOCH_DISP,
        save_freq=10,
        log_freq=500,
        eval_during_train=False,
        eval_freq=500,
        seed=SEED,
        equation=equation,
        geom=geom,
        # validator=validator,
        visualizer=visualizer_disp,
        # eval_with_no_grad=True,
        pretrained_model_path="./init_params_3d/paddle_init_only_disp",
    )

    solver_density = ppsci.solver.Solver(
        model=density_net,
        constraint=constraint_density,
        output_dir=OUTPUT_DIR,
        optimizer=optimizer_density,
        epochs=EPOCHS_DENSITY,
        iters_per_epoch=ITERS_PER_EPOCH_DENSITY,
        save_freq=100,
        log_freq=500,
        eval_during_train=False,
        eval_freq=10,
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator_density,
        visualizer=visualizer_density,
        eval_with_no_grad=True,
        pretrained_model_path="./init_params_3d/paddle_init_only_density",
        # custom_gradients = density_custom_gradients,
    )

    # for name, tensor in density_net.named_parameters():
    #     if name == "linears.5.linear.bias":
    #         print(name, tensor.grad)

    # pre-processing
    input_pos = get_force_pos(geom["geo"])
    solver_disp.train()
    # solver_density.train()

    # solver_disp.visualize()
    # solver_density.visualize()

    # for name, tensor in density_net.named_parameters():
    #     if name == "linears.5.linear.bias":
    #         print(name, tensor.grad)

    PRED_INTERVAL = 10
    for i in range(1, EPOCHS + 1):
        ppsci.utils.logger.info(f"\nEpoch: {i}\n")

        input_pos = get_force_pos(geom["geo"])
        solver_disp.train()
        # solver_density.train()
        for j in range(1, N_OPT_BATCHES + 1):
            solver_density.train()

        # plotting during training
        if i == 1 or i % PRED_INTERVAL == 0 or i == EPOCHS:
            solver_density.eval()
            # save_img(pred_input_dict, i)

            visualizer_density["vis"].prefix = plt_name + f"_density_e{i}"
            solver_density.visualize()

            visualizer_disp["vis"].prefix = plt_name + f"_disp_e{i}"
            solver_disp.visualize()

    # plot losses
    solver_disp.plot_losses(by_epoch=True, smooth_step=1)
    solver_density.plot_losses(by_epoch=False, smooth_step=10)
