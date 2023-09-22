import numpy as np

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = (
        "./output_control_arm_interior/" if args.output_dir is None else args.output_dir
    )

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # Specify parameters
    NU = 0.3  # 泊松比
    E = 1  # 弹性模量
    LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))  # lambda 拉梅常数之一
    MU = E / (2 * (1 + NU))  # mu 拉梅常数之一
    T = -0.0025  # 牵引力大小

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=3
        )
    }

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
    LR = 4e-3
    ITERS_PER_EPOCH = 1000
    EPOCHS = 2000
    DEEP = 6
    WIDTH = 512
    plt_name = "vis"
    use_para = True

    # set model
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
    model = ppsci.arch.ModelList((disp_net, stress_net))

    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        EPOCHS,
        ITERS_PER_EPOCH,
        LR,
        0.95,
        15000,
        by_epoch=False,
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))

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

    # set constraint
    arm_left_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": T, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - CIRCLE_LEFT_CENTER_XY[0])
            + np.square(y - CIRCLE_LEFT_CENTER_XY[1])
        )
        <= CIRCLE_LEFT_RADIUS + 1e-1,
        name="BC_LEFT",
    )
    arm_right_constraint = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 256},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - CIRCLE_RIGHT_CENTER_XZ[0])
            + np.square(z - CIRCLE_RIGHT_CENTER_XZ[1])
        )
        <= CIRCLE_RIGHT_RADIUS + 1e-1,
        weight_dict={"u": 10, "v": 10, "w": 10},
        name="BC_RIGHT",
    )
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
    # )
    arm_interior_constraint = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
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
        {**train_dataloader_cfg, "batch_size": 2048},
        ppsci.loss.MSELoss(loss_str),
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
    )

    # BOUNDS_X_HR = 0.0
    # arm_interior_constraint_lr = ppsci.constraint.InteriorConstraint(
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
    #     {**train_dataloader_cfg, "batch_size": 2048},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: (
    #         (BOUNDS_X[0] < x)
    #         & (x < BOUNDS_X_HR)
    #         & (BOUNDS_Y[0] < y)
    #         & (y < BOUNDS_Y[1])
    #         & (BOUNDS_Z[0] < z)
    #         & (z < BOUNDS_Z[1])
    #     ),
    #     weight_dict={
    #         "equilibrium_x": "sdf",
    #         "equilibrium_y": "sdf",
    #         "equilibrium_z": "sdf",
    #         "stress_disp_xx": "sdf",
    #         "stress_disp_yy": "sdf",
    #         "stress_disp_zz": "sdf",
    #         "stress_disp_xy": "sdf",
    #         "stress_disp_xz": "sdf",
    #         "stress_disp_yz": "sdf",
    #     },
    #     name="INTERIOR_LR",
    # )
    # arm_interior_constraint_hr = ppsci.constraint.InteriorConstraint(
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
    #     {**train_dataloader_cfg, "batch_size": 4096},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: (
    #         (BOUNDS_X_HR < x)
    #         & (x < BOUNDS_X[1])
    #         & (BOUNDS_Y[0] < y)
    #         & (y < BOUNDS_Y[1])
    #         & (BOUNDS_Z[0] < z)
    #         & (z < BOUNDS_Z[1])
    #     ),
    #     weight_dict={
    #         "equilibrium_x": "sdf",
    #         "equilibrium_y": "sdf",
    #         "equilibrium_z": "sdf",
    #         "stress_disp_xx": "sdf",
    #         "stress_disp_yy": "sdf",
    #         "stress_disp_zz": "sdf",
    #         "stress_disp_xy": "sdf",
    #         "stress_disp_xz": "sdf",
    #         "stress_disp_yz": "sdf",
    #     },
    #     name="INTERIOR_HR",
    # )

    # re-assign to ITERS_PER_EPOCH
    if use_para:
        ITERS_PER_EPOCH = len(arm_left_constraint.data_loader)
    # wrap constraints together
    constraint = {
        arm_left_constraint.name: arm_left_constraint,
        arm_right_constraint.name: arm_right_constraint,
        # arm_surface_constraint.name: arm_surface_constraint,
        arm_interior_constraint.name: arm_interior_constraint,
        # arm_interior_constraint_lr.name: arm_interior_constraint_lr,
        # arm_interior_constraint_hr.name: arm_interior_constraint_hr,
    }

    # # set validator
    # ref_xyz = ppsci.utils.reader.load_csv_file(
    #     "./datasets/data/chassis_input.txt",
    #     ("x", "y", "z"),
    #     {
    #         "x": "X Location (m)",
    #         "y": "Y Location (m)",
    #         "z": "Z Location (m)",
    #     },
    #     "\t",
    # )

    # input_dict = {
    #     "x": ref_xyz["x"],
    #     "y": ref_xyz["y"],
    #     "z": ref_xyz["z"],
    # }
    # label_dict = {
    #     "u": ref_xyz["x"],
    #     "v": ref_xyz["x"],
    #     "w": ref_xyz["x"],
    #     "sigma_xx": ref_xyz["x"],
    #     "sigma_yy": ref_xyz["x"],
    #     "sigma_zz": ref_xyz["x"],
    #     "sigma_xy": ref_xyz["x"],
    #     "sigma_xz": ref_xyz["x"],
    #     "sigma_yz": ref_xyz["x"],
    # }
    # eval_dataloader_cfg = {
    #     "dataset": {
    #         "name": "NamedArrayDataset",
    #         "input": input_dict,
    #         "label": label_dict,
    #         "weight": {k: np.ones_like(v) for k, v in label_dict.items()},
    #     },
    #     "sampler": {
    #         "name": "BatchSampler",
    #         "drop_last": False,
    #         "shuffle": False,
    #     },
    # }
    # sup_validator = ppsci.validate.SupervisedValidator(
    #     {**eval_dataloader_cfg, "batch_size": 128},
    #     ppsci.loss.MSELoss(loss_str),
    #     {
    #         "u": lambda out: out["u"],
    #         "v": lambda out: out["v"],
    #         "w": lambda out: out["w"],
    #         "sigma_xx": lambda out: out["sigma_xx"],
    #         "sigma_yy": lambda out: out["sigma_yy"],
    #         "sigma_zz": lambda out: out["sigma_zz"],
    #         "sigma_xy": lambda out: out["sigma_xy"],
    #         "sigma_xz": lambda out: out["sigma_xz"],
    #         "sigma_yz": lambda out: out["sigma_yz"],
    #     },
    #     metric={"MSE": ppsci.metric.MSE()},
    #     name="commercial_ref_u_v_w_sigmas",
    # )
    # validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    # add inferencer data
    BATCH_SIZE_PRED = 100000
    pred_input_dict = geom["geo"].sample_interior(
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
    pred_keys = list(pred_input_dict.keys())
    for key in pred_keys:
        if key not in ("x", "y", "z"):
            del pred_input_dict[key]

    visualizer = {
        "visulzie_u_v_w_sigmas": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
                "sigma_xx": lambda out: out["sigma_xx"],
                "sigma_yy": lambda out: out["sigma_yy"],
                "sigma_zz": lambda out: out["sigma_zz"],
                "sigma_xy": lambda out: out["sigma_xy"],
                "sigma_xz": lambda out: out["sigma_xz"],
                "sigma_yz": lambda out: out["sigma_yz"],
            },
            prefix=plt_name,
        )
    }

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
        eval_during_train=False,
        eval_freq=500,
        seed=SEED,
        equation=equation,
        geom=geom,
        # validator=validator,
        visualizer=visualizer,
        # eval_with_no_grad=True,
        # pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
        # checkpoint_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )

    # train model
    solver.train()

    # plot losses
    solver.plot_losses(by_epoch=True, smooth_step=1)

    # evaluate after finished training
    # solver.eval()

    # visualize prediction after finished training
    solver.visualize()

    # # directly evaluate pretrained model(optional)
    # logger.init_logger("ppsci", f"{OUTPUT_DIR}/eval.log", "info")
    # solver = ppsci.solver.Solver(
    #     model,
    #     constraint,
    #     OUTPUT_DIR,
    #     equation=equation,
    #     geom=geom,
    #     validator=validator,
    #     visualizer=visualizer,
    # pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    # )
    # solver.eval()
    # # visualize prediction for pretrained model(optional)
    # solver.visualize()
