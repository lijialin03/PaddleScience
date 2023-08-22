import numpy as np

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = "./output_extend/" if args.output_dir is None else args.output_dir

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # params for test
    loss_str = "mean"
    LR = 1e-3
    ITERS_PER_EPOCH = 100
    EPOCHS = 200
    DEEP = 6
    WIDTH = 512
    plt_name = "vis1_0"
    pretrained_model_path = "./saved_model/control_arm_2_7/epoch_2000"

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

    # Specify parameters
    NU = 0.3  # 泊松比
    E = 100.0e9  # 弹性模量
    LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))  # lambda 拉梅常数之一
    MU = E / (2 * (1 + NU))  # mu 拉梅常数之一
    # MU_C = 0.01 * MU  # 给定条件：无量纲剪切模量
    # LAMBDA_ = LAMBDA_ / MU_C  # 无量纲化，即去掉单位
    # MU = MU / MU_C  # 无量纲化，即去掉单位
    # CHARACTERISTIC_LENGTH = 1  # 给定条件：特征长度
    # CHARACTERISTIC_DISPLACEMENT = 1.0e-4  # 给定条件：特征位移
    # SIGMA_NORMALIZATION = CHARACTERISTIC_LENGTH / (CHARACTERISTIC_DISPLACEMENT * MU_C)
    # T = -4.0e4 * SIGMA_NORMALIZATION  # 牵引力大小
    T = -100  # 牵引力大小

    LAMBDA_ = LAMBDA_ / MU
    MU = 1.0

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=3
        )
    }

    # set geometry
    chassis = ppsci.geometry.Mesh("./datasets/stl/chassis.stl")
    # geometry bool operation
    geo = chassis
    geom = {"geo": geo}

    # set dataloader config
    # ITERS_PER_EPOCH = 100
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
    CHASSIS_ORIGIN = (-5, -0.5, -0.5)
    CHASSIS_DIM = (10, 1, 1)
    CHASSIS_X = (-5, 5)
    CHASSIS_Y = (-0.5, 0.5)
    CHASSIS_Z = (-0.5, 0.5)

    # force in x_max
    bc_back = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y, z: x == CHASSIS_ORIGIN[0],
        weight_dict={"u": 100, "v": 100, "w": 100},
        name="BC_BACK",
    )
    bc_front = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": T},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y, z: x == CHASSIS_ORIGIN[0] + CHASSIS_DIM[0],
        name="BC_FRONT",
    )
    bc_surface = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 2048},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y, z: np.logical_and(
            x > CHASSIS_ORIGIN[0] + 1e-7, x < CHASSIS_ORIGIN[0] + CHASSIS_DIM[0] - 1e-7
        ),
        name="BC_SURFACE",
    )

    # # force in y_max
    # bc_back = ppsci.constraint.BoundaryConstraint(
    #     {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
    #     {"u": 0, "v": 0, "w": 0},
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": 1024},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: y == CHASSIS_ORIGIN[1],
    #     weight_dict={"u": 10, "v": 10, "w": 10},
    #     name="BC_BACK",
    # )
    # bc_front = ppsci.constraint.BoundaryConstraint(
    #     equation["LinearElasticity"].equations,
    #     {"traction_x": 0, "traction_y": 0, "traction_z": T},
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": 128},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: y == CHASSIS_ORIGIN[1] + CHASSIS_DIM[1],
    #     name="BC_FRONT",
    # )
    # bc_surface = ppsci.constraint.BoundaryConstraint(
    #     equation["LinearElasticity"].equations,
    #     {"traction_x": 0, "traction_y": 0, "traction_z": 0},
    #     geom["geo"],
    #     {**train_dataloader_cfg, "batch_size": 4096},
    #     ppsci.loss.MSELoss(loss_str),
    #     criteria=lambda x, y, z: np.logical_and(
    #         y > CHASSIS_ORIGIN[1] + 1e-7, y < CHASSIS_ORIGIN[1] + CHASSIS_DIM[1] - 1e-7
    #     ),
    #     name="BC_SURFACE",
    # )

    chassis_interior_constraint = ppsci.constraint.InteriorConstraint(
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
            (CHASSIS_X[0] < x)
            & (x < CHASSIS_X[1])
            & (CHASSIS_Y[0] < y)
            & (y < CHASSIS_Y[1])
            & (CHASSIS_Z[0] < z)
            & (z < CHASSIS_Z[1])
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
        name="chassis_interior",
    )

    # wrap constraints together
    constraint = {
        bc_back.name: bc_back,
        bc_front.name: bc_front,
        bc_surface.name: bc_surface,
        chassis_interior_constraint.name: chassis_interior_constraint,
    }

    # set training hyper-parameters
    # EPOCHS = 200 if not args.epochs else args.epochs
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

    # set validator
    ref_xyz = ppsci.utils.reader.load_csv_file(
        "./datasets/data/chassis_input.txt",
        ("x", "y", "z"),
        {
            "x": "X Location (m)",
            "y": "Y Location (m)",
            "z": "Z Location (m)",
        },
        "\t",
    )

    input_dict = {
        "x": ref_xyz["x"],
        "y": ref_xyz["y"],
        "z": ref_xyz["z"],
    }
    label_dict = {
        "u": ref_xyz["x"],
        "v": ref_xyz["x"],
        "w": ref_xyz["x"],
        "sigma_xx": ref_xyz["x"],
        "sigma_yy": ref_xyz["x"],
        "sigma_zz": ref_xyz["x"],
        "sigma_xy": ref_xyz["x"],
        "sigma_xz": ref_xyz["x"],
        "sigma_yz": ref_xyz["x"],
    }
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
            "weight": {k: np.ones_like(v) for k, v in label_dict.items()},
        },
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": False,
        },
    }
    sup_validator = ppsci.validate.SupervisedValidator(
        {**eval_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss(loss_str),
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
        metric={"MSE": ppsci.metric.MSE()},
        name="commercial_ref_u_v_w_sigmas",
    )
    validator = {sup_validator.name: sup_validator}

    # set visualizer(optional)
    visualizer = {
        "visulzie_u_v_w_sigmas": ppsci.visualize.VisualizerVtu(
            input_dict,
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
        validator=validator,
        visualizer=visualizer,
        eval_with_no_grad=True,
        # pretrained_model_path=pretrained_model_path,
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
    #     pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    # )
    # solver.eval()
    # # visualize prediction for pretrained model(optional)
    # solver.visualize()
