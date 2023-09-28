import model as model_module
import numpy as np
import paddle
import paddle.nn.functional as F

import ppsci
from ppsci.autodiff import jacobian
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


def gen_energy_equation_param(out_, in_, dim):
    param_dict = {"lambda_": LAMBDA_, "mu": MU}
    x, y, z = in_["x"], in_["y"], in_["z"]
    u, v, w = out_["u"], out_["v"], out_["w"]
    param_dict["u__x"] = jacobian(u, x).detach().clone()
    param_dict["u__y"] = jacobian(u, y).detach().clone()
    param_dict["v__x"] = jacobian(v, x).detach().clone()
    param_dict["v__y"] = jacobian(v, y).detach().clone()
    # print("####", paddle.device.cuda.memory_allocated())
    ppsci.autodiff.clear()
    if dim == 3:
        param_dict["u__z"] = jacobian(u, z).detach().clone()
        param_dict["v__z"] = jacobian(v, z).detach().clone()
        param_dict["w__x"] = jacobian(w, x).detach().clone()
        param_dict["w__y"] = jacobian(w, y).detach().clone()
        param_dict["w__z"] = jacobian(w, z).detach().clone()
        ppsci.autodiff.clear()
    return ppsci.equation.EnergyEquation(param_dict, dim)


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


def compute_oc_multi_batch(old_densities, sensitivities):
    target_volume = volume_ratio * volume
    # print("target_volume",target_volume)
    lagrange_lower_estimate = 0
    lagrange_upper_estimate = 1e9
    conv_threshold = 1e-3

    total_samples = len(old_densities) * old_densities[0].shape[0]
    dv = volume / total_samples

    density_lower_bound = [
        paddle.maximum(paddle.to_tensor(0.0), od - max_move) for od in old_densities
    ]
    density_upper_bound = [
        paddle.minimum(paddle.to_tensor(1.0), od + max_move) for od in old_densities
    ]

    while (lagrange_upper_estimate - lagrange_lower_estimate) / (
        lagrange_lower_estimate + lagrange_upper_estimate
    ) > conv_threshold:
        lagrange_current = 0.5 * (lagrange_upper_estimate + lagrange_lower_estimate)
        # print("lagrange_current",lagrange_current)

        target_densities = [
            paddle.maximum(
                density_lower_bound[i],
                paddle.minimum(
                    density_upper_bound[i],
                    old_densities[i]
                    * paddle.pow(
                        sensitivities[i] / (-dv * lagrange_current), damping_parameter
                    ),
                ),
            )
            .detach()
            .clone()
            for i in range(len(old_densities))
        ]
        # print("target_densities", target_densities)

        # new_volume = volume * paddle.mean(
        #     paddle.to_tensor([paddle.mean(di) for di in target_densities])
        # )
        new_volume = volume * np.mean([paddle.mean(di) for di in target_densities])
        # print("new_volume", new_volume)

        if new_volume > target_volume:
            lagrange_lower_estimate = lagrange_current
        else:
            lagrange_upper_estimate = lagrange_current
        # exit()
    # print("target_densities", target_densities)
    return target_densities


def compute_target_densities_gradient_descent(old_densities, sensitivities):
    projected_sensitivities = [
        (
            paddle.maximum(
                paddle.to_tensor(0.0),
                paddle.minimum(
                    paddle.to_tensor(1.0), old_densities[i] - sensitivities[i]
                ),
            )
            - old_densities[i]
        )
        for i in range(len(old_densities))
    ]

    step_size = 0.05 / paddle.mean(
        paddle.to_tensor([paddle.abs(si) for si in projected_sensitivities]),
        keepdim=True,
    )
    return [
        old_densities[i] - step_size * sensitivities[i]
        for i in range(len(old_densities))
    ]


def compute_target_densities(old_densities, sensitivities, use_oc):
    # print("####", type(old_densities), type(sensitivities))
    if use_oc:
        return compute_oc_multi_batch(old_densities, sensitivities)
    else:
        return compute_target_densities_gradient_descent(old_densities, sensitivities)


def disp_loss_func(output_dict, label_dict=None, weight_dict={}, input_dict=None):
    densities = density_net(input_dict)["densities"].detach().clone()
    # densities = output_dict["densities"]
    energy_xyz = output_dict["energy_xyz"]
    loss_energy = comput_energy(densities, energy_xyz)
    loss_force = comput_force()
    # print("loss_energy:", "%.3e" % float(loss_energy))
    # print("loss_force:", "%.3e" % float(loss_force))
    return loss_energy + loss_force


def density_loss_func(
    output_dicts_list,
    label_dicts_list=None,
    weight_dicts_list=None,
    input_dicts_list=None,
):
    loss_list = []
    densities_list = []
    sensitivities_list = []
    for i, output_dict in enumerate(output_dicts_list):
        # print("####", i, paddle.device.cuda.memory_allocated())
        densities = output_dict[0]["densities"]
        output_dict_disp = disp_net(input_dicts_list[i][0])
        energy_no_backward = gen_energy_equation_param(
            output_dict_disp, input_dicts_list[i][0], dim=3
        )
        energy_xyz = energy_no_backward.equations["energy_xyz"]({})
        # energy_xyz = equation["EnergyEquation"].equations(
        #     {**disp_net(input_dicts_list[i][0]), **input_dicts_list[i][0]}
        # )
        # energy_xyz = energy_xyz.detach().clone()

        loss_energy = comput_energy(densities, energy_xyz)
        # print("### loss_energy", "%.3e" % loss_energy)
        loss = -loss_energy
        if not use_oc:
            loss_penalty = comput_penalty(densities)
            # print("### loss_penalty", "%.3e" % loss_penalty)
            loss += loss_penalty
        loss_list.append(loss)

        sensitivities = paddle.grad(loss, densities)[0]
        densities_list.append(densities)
        sensitivities_list.append(sensitivities)
        ppsci.autodiff.clear()

    if not use_mmse:
        return loss_list
    else:
        #     print("densities_list",np.mean([td.numpy().mean() for td in densities_list]))
        #     print("sensitivities_list",np.mean([td.numpy().mean() for td in sensitivities_list]))
        target_densities_list = compute_target_densities(
            densities_list, sensitivities_list, use_oc
        )
        print("use_mmse", np.mean([td.numpy().mean() for td in target_densities_list]))
        return [
            F.mse_loss(densities_list[i], target_densities_list[i], "mean")
            for i in range(len(target_densities_list))
        ]


def density_metric_func(output_dict, *args):
    density = output_dict["densities"]
    print("mean:", float(paddle.mean(density)))
    print("max:", float(paddle.max(density)))
    print("min:", float(paddle.min(density)))
    metric_dict = {"densities": density.mean() - volume_ratio}
    return metric_dict


if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = (
        "./output_ntopo_3d_density/" if args.output_dir is None else args.output_dir
    )
    OUTPUT_DIR_DISP = "./output_ntopo_3d_disp/"

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
    use_oc = True
    max_move = 0.2
    damping_parameter = 0.5  # 阻尼参数

    EPOCHS = 900  # times for for-loop
    EPOCHS_DISP = EPOCHS_DENSITY = 1
    ITERS_PER_EPOCH = 1
    ITERS_PER_EPOCH_DISP = 1000
    ITERS_PER_EPOCH_DENSITY = 20  # times for n_opt_batch
    # N_OPT_BATCHES = 10

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
    # pt_sup = ppsci.constraint.SupervisedConstraint(
    #     {
    #         "dataset": {
    #             "name": "NamedArrayDataset",
    #             "input": {
    #                 "x": np.array(
    #                     [[0], [0.5], [1.0]], dtype=paddle.get_default_dtype()
    #                 ),
    #                 "y": np.array(
    #                     [[0], [0.25], [0.5]], dtype=paddle.get_default_dtype()
    #                 ),
    #                 "z": np.array(
    #                     [[0], [0.125], [0.25]], dtype=paddle.get_default_dtype()
    #                 ),
    #             },
    #             "label": {
    #                 "u": np.array([[0], [0], [0]], dtype=paddle.get_default_dtype()),
    #                 "v": np.array([[0], [0], [0]], dtype=paddle.get_default_dtype()),
    #                 "w": np.array([[0], [0], [0]], dtype=paddle.get_default_dtype()),
    #                 "energy_xyz": np.array(
    #                     [[0], [0], [0]], dtype=paddle.get_default_dtype()
    #                 ),
    #             },
    #         },
    #         "batch_size": 3,
    #         "sampler": {
    #             "name": "BatchSampler",
    #             "drop_last": True,
    #             "shuffle": False,
    #         },
    #         "num_workers": 1,
    #     },
    #     # ppsci.loss.MSELoss("mean"),
    #     ppsci.loss.FunctionalLoss(disp_loss_func),
    #     # ppsci.loss.FunctionalLoss(density_loss_func),
    #     equation["EnergyEquation"].equations,
    #     name="debug",
    # )
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
    #     ppsci.loss.FunctionalLoss(disp_loss_func),
    #     name="debug",
    # )
    # pt_sup_2 = ppsci.constraint.SupervisedConstraint(
    #     {
    #         "dataset": {
    #             "name": "NamedArrayDataset",
    #             "input": {
    #                 "x": np.array(
    #                     [[0], [0.5], [1.0]], dtype=paddle.get_default_dtype()
    #                 ),
    #                 "y": np.array(
    #                     [[0], [0.25], [0.5]], dtype=paddle.get_default_dtype()
    #                 ),
    #                 "z": np.array(
    #                     [[0], [0.125], [0.25]], dtype=paddle.get_default_dtype()
    #                 ),
    #             },
    #             "label": {
    #                 "density": np.array(
    #                     [[0], [0], [0]], dtype=paddle.get_default_dtype()
    #                 ),
    #             },
    #         },
    #         "batch_size": 3,
    #         "sampler": {
    #             "name": "BatchSampler",
    #             "drop_last": True,
    #             "shuffle": False,
    #         },
    #         "num_workers": 1,
    #     },
    #     ppsci.loss.FunctionalLoss(density_loss_func),
    #     equation["EnergyEquation"].equations,
    #     name="debug",
    # )

    # constraint_test = {pt_sup_2.name: pt_sup_2}

    # set constraint
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
        ppsci.loss.FunctionalLossBatch(density_loss_func),
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
        output_dir=OUTPUT_DIR_DISP,
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
        # pretrained_model_path="./init_params_3d/paddle_init_only_disp",
        pretrained_model_path=f"{OUTPUT_DIR_DISP}checkpoints/latest",
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
        # pretrained_model_path="./init_params_3d/paddle_init_only_density",
        pretrained_model_path=f"{OUTPUT_DIR}checkpoints/latest",
    )

    # pre-processing
    # input_pos = get_force_pos(geom["geo"])
    # solver_disp.train()
    # solver_density.train()

    PRED_INTERVAL = 5
    for i in range(1, EPOCHS + 1):
        ppsci.utils.logger.info(f"\nEpoch: {i}\n")
        # print("####", i, paddle.device.cuda.memory_allocated())

        input_pos = get_force_pos(geom["geo"])
        solver_disp.train()
        solver_density.train_batch()
        # for j in range(1, N_OPT_BATCHES + 1):
        #     solver_density.train()

        # plotting during training
        if i == 1 or i % PRED_INTERVAL == 0 or i == EPOCHS:
            solver_density.eval()
            # save_img(pred_input_dict, i)

            visualizer_density["vis"].prefix = plt_name + f"_density_e{i}"
            solver_density.visualize()

            visualizer_disp["vis"].prefix = plt_name + f"_disp_e{i}"
            solver_disp.visualize()
        # print("####", i, paddle.device.cuda.memory_allocated())

    # # # plot losses
    # # solver_disp.plot_losses(by_epoch=True, smooth_step=1)
    # # solver_density.plot_losses(by_epoch=False, smooth_step=10)
