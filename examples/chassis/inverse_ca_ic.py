import numpy as np
import paddle

import ppsci
from ppsci.autodiff import jacobian
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import save_load


def gen_input_data(input_sample):
    disp_net = ppsci.arch.MLP(
        ("x", "y", "z"), ("u", "v", "w"), DEEP, WIDTH, "silu", weight_norm=True
    )
    stress_net = ppsci.arch.MLP(
        ("x", "y", "z"),
        ("sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"),
        DEEP,
        WIDTH,
        "silu",
        weight_norm=True,
    )

    # wrap to a model_list
    model_forward = ppsci.arch.ModelList((disp_net, stress_net))
    save_load.load_checkpoint(
        path="./saved_model/control_arm_3_4/epoch_2000", model=model_forward
    )

    for key in ("x", "y", "z"):
        input_sample[key] = paddle.to_tensor(input_sample[key])
        input_sample[key].stop_gradient = False

    input = {"x": input_sample["x"], "y": input_sample["y"], "z": input_sample["z"]}
    output = model_forward(input)

    x, y, z = input["x"], input["y"], input["z"]
    u, v, w = output["u"], output["v"], output["w"]
    sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz = (
        output["sigma_xx"],
        output["sigma_yy"],
        output["sigma_zz"],
        output["sigma_xy"],
        output["sigma_xz"],
        output["sigma_yz"],
    )

    return {
        "u__x": jacobian(u, x).numpy(),
        "u__y": jacobian(u, y).numpy(),
        "u__z": jacobian(u, z).numpy(),
        "v__x": jacobian(v, x).numpy(),
        "v__y": jacobian(v, y).numpy(),
        "v__z": jacobian(v, z).numpy(),
        "w__x": jacobian(w, x).numpy(),
        "w__y": jacobian(w, y).numpy(),
        "w__z": jacobian(w, z).numpy(),
        "sigma_xx": sigma_xx.numpy(),
        "sigma_yy": sigma_yy.numpy(),
        "sigma_zz": sigma_zz.numpy(),
        "sigma_xy": sigma_xy.numpy(),
        "sigma_xz": sigma_xz.numpy(),
        "sigma_yz": sigma_yz.numpy(),
        "sigma_xx__x": jacobian(sigma_xx, x).numpy(),
        "sigma_xy__x": jacobian(sigma_xy, x).numpy(),
        "sigma_xz__x": jacobian(sigma_xz, x).numpy(),
        "sigma_xy__y": jacobian(sigma_xy, y).numpy(),
        "sigma_yy__y": jacobian(sigma_yy, y).numpy(),
        "sigma_yz__y": jacobian(sigma_yz, y).numpy(),
        "sigma_xz__z": jacobian(sigma_xz, z).numpy(),
        "sigma_yz__z": jacobian(sigma_yz, z).numpy(),
        "sigma_zz__z": jacobian(sigma_zz, z).numpy(),
    }


if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = (
        "./output_inverse_ca_2/" if args.output_dir is None else args.output_dir
    )

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set geometry
    control_arm = ppsci.geometry.Mesh("./datasets/stl/control_arm.stl")
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}

    CIRCLE_LEFT_CENTER_XY = (-4.4, 0)
    CIRCLE_LEFT_RADIUS = 1.65
    CIRCLE_RIGHT_CENTER_XZ = (15.8, 0)
    CIRCLE_RIGHT_RADIUS = 2.21

    # set bounds
    # print(control_arm.bounds)
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

    # set training hyper-parameters
    # MARK
    loss_str = "mean"
    LR = 1e-3
    ITERS_PER_EPOCH = 100
    EPOCHS = 20
    DEEP = 6
    WIDTH = 512
    plt_name = "vis"
    use_para = False
    BATCH_SIZE_PRED = 1000

    # set model
    inverse_net = ppsci.arch.MLP(
        (
            "u__x",
            "u__y",
            "u__z",
            "v__x",
            "v__y",
            "v__z",
            "w__x",
            "w__y",
            "w__z",
            "sigma_xx",
            "sigma_yy",
            "sigma_zz",
            "sigma_xy",
            "sigma_xz",
            "sigma_yz",
            "sigma_xx__x",
            "sigma_xy__x",
            "sigma_xz__x",
            "sigma_xy__y",
            "sigma_yy__y",
            "sigma_yz__y",
            "sigma_xz__z",
            "sigma_yz__z",
            "sigma_zz__z",
            # "normal_x",
            # "normal_y",
            # "normal_z",
        ),
        ("lambda_", "mu"),
        DEEP,
        WIDTH,
        "silu",
        weight_norm=True,
    )
    # wrap to a model_list
    model = ppsci.arch.ModelList((inverse_net,))

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        EPOCHS,
        ITERS_PER_EPOCH,
        LR,
        0.95,
        15000,
        by_epoch=False,
    )()

    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

    # add input data
    input_sample = geom["geo"].sample_interior(
        BATCH_SIZE_PRED,
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )

    # get data from loaded model
    input_dict = gen_input_data(input_sample)

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH,
        "batch_size": BATCH_SIZE_PRED,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    # arm_left_constraint = ppsci.constraint.BoundaryConstraint(
    #     equation["LinearElasticity"].equations,
    #     {"traction_x": 0, "traction_y": 0, "traction_z": T},
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": 128},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: np.sqrt(
    #         np.square(x - CIRCLE_LEFT_CENTER_XY[0])
    #         + np.square(y - CIRCLE_LEFT_CENTER_XY[1])
    #     )
    #     <= CIRCLE_LEFT_RADIUS + 1e-1,
    #     name="BC_LEFT",
    #     from_dict = input_sample,
    # )
    # arm_right_constraint = ppsci.constraint.BoundaryConstraint(
    #     {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
    #     {"u": 0, "v": 0, "w": 0},
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": 256},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: np.sqrt(
    #         np.square(x - CIRCLE_RIGHT_CENTER_XZ[0])
    #         + np.square(z - CIRCLE_RIGHT_CENTER_XZ[1])
    #     )
    #     <= CIRCLE_RIGHT_RADIUS + 1e-1,
    #     weight_dict={"u": 100, "v": 100, "w": 100},
    #     name="BC_RIGHT",
    #     from_dict = input_sample,
    # )
    # arm_surface_constraint = ppsci.constraint.BoundaryConstraint(
    #     equation["LinearElasticity"].equations,
    #     {"traction_x": 0, "traction_y": 0, "traction_z": 0},
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": 4096},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: np.sqrt(
    #         np.square(x - CIRCLE_LEFT_CENTER_XY[0])
    #         + np.square(y - CIRCLE_LEFT_CENTER_XY[1])
    #     )
    #     > CIRCLE_LEFT_RADIUS + 1e-1,
    #     name="BC_SURFACE",
    #     from_dict = input_sample,
    # )
    # arm_interior_constraint = ppsci.constraint.InteriorConstraint(
    #     equation["LinearElasticity"].equations,
    #     {
    #         "equilibrium_x": 0,
    #         "equilibrium_y": 0,
    #         "equilibrium_z": 0,
    #         "stress_disp_xx": 0,
    #         "stress_disp_yy": 0,
    #         "stress_disp_zz": 0,
    #         "stress_disp_xy": 0,
    #         "stress_disp_xz": 0,
    #         "stress_disp_yz": 0,
    #     },
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": BATCH_SIZE_PRED},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: (
    #         (BOUNDS_X[0] < x)
    #         & (x < BOUNDS_X[1])
    #         & (BOUNDS_Y[0] < y)
    #         & (y < BOUNDS_Y[1])
    #         & (BOUNDS_Z[0] < z)
    #         & (z < BOUNDS_Z[1])
    #     ),
    #     # weight_dict={
    #     #     "equilibrium_x": "sdf",
    #     #     "equilibrium_y": "sdf",
    #     #     "equilibrium_z": "sdf",
    #     #     "stress_disp_xx": "sdf",
    #     #     "stress_disp_yy": "sdf",
    #     #     "stress_disp_zz": "sdf",
    #     #     "stress_disp_xy": "sdf",
    #     #     "stress_disp_xz": "sdf",
    #     #     "stress_disp_yz": "sdf",
    #     # },
    #     name="INTERIOR",
    #     from_dict = input_sample,
    # )
    #
    # # re-assign to ITERS_PER_EPOCH
    # if use_para:
    #     ITERS_PER_EPOCH = len(arm_interior_constraint.data_loader)
    # wrap constraints together
    # constraint = {
    #     # arm_left_constraint.name: arm_left_constraint,
    #     # arm_right_constraint.name: arm_right_constraint,
    #     # arm_surface_constraint.name: arm_surface_constraint,
    #     arm_interior_constraint.name: arm_interior_constraint,
    # }

    equation = {
        "LinearElasticity_v2": ppsci.equation.LinearElasticity_v2(param_dict={}, dim=3)
    }

    test_ic = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity_v2"].equations,
        {
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "equilibrium_z": 0,
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_zz": 0,
            "stress_disp_xy": 0,
            "stress_disp_xz": 0,
            "stress_disp_yz": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": BATCH_SIZE_PRED},
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.L1Loss(loss_str),
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
        weight_dict={
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "equilibrium_z": "sdf",
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_zz": "sdf",
            "stress_disp_xy": "sdf",
            "stress_disp_xz": "sdf",
            "stress_disp_yz": "sdf",
        },
        name="INTERIOR",
        from_dict={**input_dict, "sdf": input_sample["sdf"]},
    )
    constraint = {test_ic.name: test_ic}

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": {
                "lambda_": np.full(
                    (BATCH_SIZE_PRED, 1), 1.5, dtype=paddle.get_default_dtype()
                ),
                "mu": np.full(
                    (BATCH_SIZE_PRED, 1), 1.0, dtype=paddle.get_default_dtype()
                ),
            },
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": 128},
        ppsci.loss.L1Loss(loss_str),
        {
            "lambda_": lambda out: out["lambda_"],
            "mu": lambda out: out["mu"],
        },
        metric={"MAE": ppsci.metric.MAE()},
        name="mse_lambda_mu",
    )
    validator = {sup_validator.name: sup_validator}

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        OUTPUT_DIR,
        optimizer,
        lr_scheduler,
        EPOCHS,
        ITERS_PER_EPOCH,
        save_freq=500,
        log_freq=500,
        eval_during_train=True,
        eval_freq=20,
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator,
        # visualizer=visualizer,
        eval_with_no_grad=True,
        # pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )

    # train model
    solver.train()

    # plot losses
    solver.plot_losses(by_epoch=True, smooth_step=1)

    # evaluate after finished training
    solver.eval()

    # # visualize prediction after finished training
    # solver.visualize()

    pred_dict = solver.predict(input_dict)
    print("lambda:", pred_dict["lambda_"].numpy().mean())
    print("mu:", pred_dict["mu"].numpy().mean())
