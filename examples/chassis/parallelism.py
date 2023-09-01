import time

import numpy as np

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = (
        "./output_parallelism_ori/" if args.output_dir is None else args.output_dir
    )

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # Specify parameters
    NU = 0.3  # 泊松比
    E = 100.0e9  # 弹性模量
    LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))  # lambda 拉梅常数之一
    MU = E / (2 * (1 + NU))  # mu 拉梅常数之一
    # MU_C = 0.01 * MU  # 给定条件：无量纲剪切模量
    # LAMBDA_ = LAMBDA_ / MU_C  # 无量纲化，即去掉单位
    # MU = MU / MU_C  # 无量纲化，即去掉单位
    T = -100  # 牵引力大小

    LAMBDA_ = LAMBDA_ / MU
    MU = 1.0
    # print("LAMBDA_:", LAMBDA_, ", MU:", MU)

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

    # CHARACTERISTIC_LENGTH = CIRCLE_RADIUS * 2  # 给定条件：特征长度
    # CHARACTERISTIC_DISPLACEMENT = 1.0e-4  # 给定条件：特征位移
    # SIGMA_NORMALIZATION = CHARACTERISTIC_LENGTH / (CHARACTERISTIC_DISPLACEMENT * MU_C)
    # T = T * SIGMA_NORMALIZATION  # 牵引力大小
    # print("T", T)

    # set training hyper-parameters
    # MARK
    loss_str = "mean"
    LR = 1e-3
    ITERS_PER_EPOCH = 100
    EPOCHS = 200
    DEEP = 6
    WIDTH = 512
    plt_name = "vis"
    use_para = False

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
        {"traction_x": 0, "traction_y": 0, "traction_z": T},
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
        weight_dict={"u": 100, "v": 100, "w": 100},
        name="BC_RIGHT",
    )
    arm_surface_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 4096},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - CIRCLE_LEFT_CENTER_XY[0])
            + np.square(y - CIRCLE_LEFT_CENTER_XY[1])
        )
        > CIRCLE_LEFT_RADIUS + 1e-1,
        name="BC_SURFACE",
    )
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

    # re-assign to ITERS_PER_EPOCH
    if use_para:
        ITERS_PER_EPOCH = len(arm_left_constraint.data_loader)

    # wrap constraints togetherg
    constraint = {
        arm_left_constraint.name: arm_left_constraint,
        arm_right_constraint.name: arm_right_constraint,
        arm_surface_constraint.name: arm_surface_constraint,
        arm_interior_constraint.name: arm_interior_constraint,
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
        seed=SEED,
        equation=equation,
        geom=geom,
    )

    start = time.perf_counter()
    # train model
    solver.train()
    end = time.perf_counter()

    print("training time:", end - start)
