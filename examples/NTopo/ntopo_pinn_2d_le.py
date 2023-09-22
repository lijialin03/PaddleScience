import model as model_module
import numpy as np
import paddle
import paddle.nn.functional as F

import ppsci
from ppsci.utils import config
from ppsci.utils import logger


def comput_energy(densities, energy_xy):
    exponent = 3.0  # TODO: trainable parameter
    energy_densities = paddle.pow(densities, exponent) * energy_xy
    return volume * paddle.mean(energy_densities, keepdim=True)


def comput_penalty(densities):
    vol_penalty_strength = 10.0
    target_volume = 0.5 * volume
    volume_estimate = volume * paddle.mean(densities, keepdim=True)
    return (
        vol_penalty_strength
        * (volume_estimate - target_volume)
        * (volume_estimate - target_volume)
        / target_volume
    )


def apply_oc(old_densities, sensitivities):
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


def density_loss_func(output_dict, label_dict=None, weight_dict={}, input_dict=None):
    densities = output_dict["densities"]
    # energy_xy = output_dict["energy_xy"]
    energy_xy = equation["EnergyEquation"].equations["energy_xy"](
        {**disp_net(input_dict), **input_dict}
    )
    energy_xy.stop_gradient = True

    loss_energy = comput_energy(densities, energy_xy)
    loss_penalty = comput_penalty(densities)
    # print("### loss_energy", "%.3e" % loss_energy)
    # print("### loss_penalty", "%.3e" % loss_penalty)
    if use_mmse:
        losses = loss_energy if use_oc else loss_energy + loss_penalty
        sensitivities = paddle.grad(losses, densities)[0]
        target_densities = apply_oc(densities, sensitivities)
        return F.mse_loss(densities, target_densities, "mean")
    else:
        return -loss_energy + loss_penalty


def density_metric_func(output_dict, *args):
    density = output_dict["densities"]
    print("mean:", float(paddle.mean(density)))
    print("max:", float(paddle.max(density)))
    print("min:", float(paddle.min(density)))
    metric_dict = {"densities": density.mean() - 0.5}
    return metric_dict


def reduce_loss_func(output_dict, label_dict=None, weight_dict={}, input_dict=None):
    losses = 0.0
    for key in label_dict:
        losses += paddle.mean(
            (output_dict[key] - label_dict[key]) * weight_dict[key], keepdim=True
        )
    return losses


if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = (
        "./output_ntopo_pinn_2d_le/" if args.output_dir is None else args.output_dir
    )

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # Specify parameters
    NU = 0.3  # 泊松比
    E = 1.0  # 弹性模量
    LAMBDA_ = NU * E / ((1 - NU * NU))  # lambda 拉梅常数之一 但不太一样
    MU = E / (1 + NU)  # mu 拉梅常数之一 但不太一样
    # T = -0.0025  # 牵引力大小
    T = -25

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=2
        ),
        "EnergyEquation": ppsci.equation.EnergyEquation(
            param_dict={"lambda_": LAMBDA_, "mu": MU},
            dim=2,
        ),
    }

    # set geometry
    BEAM_ORIGIN = (0.0, 0.0)
    BEAM_DIM = (1.5, 0.5)
    beam = ppsci.geometry.Rectangle(
        BEAM_ORIGIN, (BEAM_ORIGIN[0] + BEAM_DIM[0], BEAM_ORIGIN[1] + BEAM_DIM[1])
    )
    # geometry bool operation
    geo = beam
    geom = {"geo": geo}

    def comput_volume(dim):
        return dim[0] * dim[1]

    volume = comput_volume(BEAM_DIM)

    # set training hyper-parameters
    # MARK
    loss_str = "mean"
    plt_name = "vis"
    use_para = False
    LR = 1e-4

    filter = "sensitivity"
    use_oc = False
    use_mmse = False

    EPOCHS = 30  # times for for-loop
    EPOCHS_DISP = EPOCHS_DENSITY = 1
    ITERS_PER_EPOCH = 1000
    ITERS_PER_EPOCH_DISP = 1000
    ITERS_PER_EPOCH_DENSITY = 50  # times for n_opt_batch
    # N_OPT_BATCHES = 10

    # set model
    input_keys = ("x_scaled", "y_scaled", "sin_x_scaled", "sin_y_scaled")
    # stress_net = model_module.DenseSIRENModel(
    #     input_keys, ("sigma_xx", "sigma_yy", "sigma_xy"), 6, 60, 0.001
    # )
    # disp_net = model_module.DenseSIRENModel(input_keys, ("u", "v"), 6, 60, 0.001)
    disp_net = ppsci.arch.MLP(input_keys, ("u", "v"), 6, 512, "silu", weight_norm=True)
    stress_net = ppsci.arch.MLP(
        input_keys,
        ("sigma_xx", "sigma_yy", "sigma_xy"),
        6,
        512,
        "silu",
        weight_norm=True,
    )
    density_net = model_module.DenseSIRENModel(input_keys, ("density",), 6, 60, 0.001)

    # input transform
    def transform_in(_in):
        x, y = _in["x"], _in["y"]
        x_scaled = 2.0 / BEAM_DIM[0] * x + (-1.0 - 2.0 * BEAM_ORIGIN[0] / BEAM_DIM[0])
        y_scaled = 2.0 / BEAM_DIM[1] * y + (-1.0 - 2.0 * BEAM_ORIGIN[1] / BEAM_DIM[1])

        sin_x_scaled, sin_y_scaled = paddle.sin(x_scaled), paddle.sin(y_scaled)
        return {
            "x_scaled": x_scaled,
            "y_scaled": y_scaled,
            "sin_x_scaled": sin_x_scaled,
            "sin_y_scaled": sin_y_scaled,
        }

    def transform_out_disp(_in, _out):
        x_scaled = _in["x_scaled"]
        x = BEAM_DIM[0] / 2 * (1 + x_scaled) + BEAM_ORIGIN[0]
        out_trans = {}
        for key in _out:
            out_trans[key] = x * _out[key]
        return out_trans

    def transform_out_density(_in, _out):
        density = _out["density"]
        volume_ratio = 0.5
        alpha = 5.0
        offset = np.log(volume_ratio / (1.0 - volume_ratio))
        densities = F.sigmoid(alpha * density + offset)
        return {"densities": densities}

    disp_net.register_input_transform(transform_in)
    disp_net.register_output_transform(transform_out_disp)

    stress_net.register_input_transform(transform_in)
    stress_net.register_output_transform(transform_out_disp)

    density_net.register_input_transform(transform_in)
    density_net.register_output_transform(transform_out_density)

    # wrap to a model_list
    model = ppsci.arch.ModelList((disp_net, stress_net))

    # set optimizer
    optimizer_disp = ppsci.optimizer.Adam(learning_rate=LR, beta2=0.99)((model,))
    optimizer_density = ppsci.optimizer.Adam(learning_rate=LR, beta1=0.8, beta2=0.9)(
        (density_net,)
    )

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
    batch_size = {"bs": 50 * 150, "bc": 50, "force": 1, "interior": 50 * 150}

    # set constraint
    bc_left_disp = ppsci.constraint.BoundaryConstraint(
        {"u": lambda out: out["u"], "v": lambda out: out["v"]},
        {"u": 0, "v": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["bc"]},
        ppsci.loss.MSELoss(loss_str),
        # ppsci.loss.FunctionalLoss(reduce_loss_func),
        criteria=lambda x, y: x == BEAM_ORIGIN[0],
        weight_dict={"u": 10, "v": 10},
        name="BC_LEFT_DISP",
    )
    bc_right_corner_disp = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": T},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["force"]},
        ppsci.loss.MSELoss(loss_str),
        # ppsci.loss.FunctionalLoss(reduce_loss_func),
        criteria=lambda x, y: np.logical_and(
            x > BEAM_ORIGIN[0] + BEAM_DIM[0] - 1e-3,
            y < BEAM_ORIGIN[1] + 1e-3,
        ),
        # criteria=lambda x, y: x == BEAM_DIM[0],
        name="BC_RIGHT_CORNER_DISP",
    )
    interior_disp = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_xy": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["interior"]},
        ppsci.loss.MSELoss(loss_str),
        # ppsci.loss.FunctionalLoss(reduce_loss_func),
        weight_dict={
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_xy": "sdf",
        },
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
    #     ITERS_PER_EPOCH_DISP = len(bc_left_disp.data_loader)
    #     ITERS_PER_EPOCH_DENSITY = len(interior_density.data_loader)

    # wrap constraints together
    constraint_disp = {
        bc_left_disp.name: bc_left_disp,
        bc_right_corner_disp.name: bc_right_corner_disp,
        interior_disp.name: interior_disp,
    }
    constraint_density = {interior_density.name: interior_density}

    # set visualizer(optional)
    # add inferencer data
    BATCH_SIZE_PRED = 50 * 150
    pred_input_dict = geom["geo"].sample_interior(BATCH_SIZE_PRED)
    pred_keys = list(pred_input_dict.keys())
    for key in pred_keys:
        if key not in ("x", "y"):
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
        "batch_size": 50 * 150,
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
                "sigma_xx": lambda out: out["sigma_xx"],
                "sigma_yy": lambda out: out["sigma_yy"],
                "sigma_xy": lambda out: out["sigma_xy"],
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
        model=model,
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
        # pretrained_model_path="./init_params/paddle_init_only_disp",
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
        # pretrained_model_path="./init_params/paddle_init_only_density",
        # custom_gradients = density_custom_gradients,
    )

    # for name, tensor in density_net.named_parameters():
    #     if name == "linears.5.linear.bias":
    #         print(name, tensor.grad)

    # pre-processing
    # solver_disp.train()
    # solver_density.train()

    # solver_disp.visualize()
    # solver_density.visualize()

    # for name, tensor in density_net.named_parameters():
    #     if name == "linears.5.linear.bias":
    #         print(name, tensor.grad)

    PRED_INTERVAL = 10
    for i in range(1, EPOCHS + 1):
        ppsci.utils.logger.info(f"\nEpoch: {i}\n")
        # plotting during training
        if i == 1 or i % PRED_INTERVAL == 0 or i == EPOCHS:
            # solver_density.eval()

            # visualizer_density["vis"].prefix = plt_name + f"_density_e{i}"
            # solver_density.visualize()

            visualizer_disp["vis"].prefix = plt_name + f"_disp_e{i}"
            solver_disp.visualize()

        solver_disp.train()
        # solver_density.train()
        # for j in range(1, N_OPT_BATCHES + 1):
        #     solver_density.train()

    # # plot losses
    # solver_disp.plot_losses(by_epoch=True, smooth_step=1)
    # solver_density.plot_losses(by_epoch=False, smooth_step=10)
