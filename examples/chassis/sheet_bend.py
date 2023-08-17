import numpy as np

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = "./output_sheet/" if args.output_dir is None else args.output_dir

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # params for test
    LR = 1e-2
    ITERS_PER_EPOCH = 1
    EPOCHS = 2
    DEEP = 4
    WIDTH = 512
    plt_name = "sheet_test"

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
    MU_C = 0.01 * MU  # 给定条件：无量纲剪切模量
    LAMBDA_ = LAMBDA_ / MU_C  # 无量纲化，即去掉单位
    MU = MU / MU_C  # 无量纲化，即去掉单位
    CHARACTERISTIC_LENGTH = 1  # 给定条件：特征长度
    CHARACTERISTIC_DISPLACEMENT = 1e-4  # 给定条件：特征位移
    SIGMA_NORMALIZATION = CHARACTERISTIC_LENGTH / (CHARACTERISTIC_DISPLACEMENT * MU_C)
    T = -4.0e5 * SIGMA_NORMALIZATION  # 牵引力大小

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=3
        )
    }

    # set geometry
    sheet = ppsci.geometry.Mesh("./datasets/stl/sheet.stl")
    # geometry bool operation
    geo = sheet
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
    SHEET_ORIGIN = (-1, -0.5, -5e-4)
    SHEET_DIM = (2, 1, 1e-3)
    SHEET_X = (-1, 1)
    SHEET_Y = (-0.5, 0.5)
    SHEET_Z = (-5e-4, 5e-4)

    # force in x_max
    bc_side = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"], "v": lambda d: d["v"], "w": lambda d: d["w"]},
        {"u": 0, "v": 0, "w": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.logical_or(
            y == SHEET_ORIGIN[1], y == SHEET_ORIGIN[1] + SHEET_DIM[1]
        ),
        weight_dict={"u": 10, "v": 10, "w": 10},
        name="BC_SIDE",
    )
    bc_mid = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": T},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 128},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.logical_and(x > -1e-1, x < 1e-1),
        name="BC_MID",
    )
    bc_surface = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 2048},
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: np.logical_and(
            x > SHEET_ORIGIN[0] + 1e-7, x < SHEET_ORIGIN[0] + SHEET_DIM[0] - 1e-7
        ),
        name="BC_SURFACE",
    )

    sheet_interior_constraint = ppsci.constraint.InteriorConstraint(
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
        ppsci.loss.MSELoss("sum"),
        criteria=lambda x, y, z: (
            (SHEET_X[0] < x)
            & (x < SHEET_X[1])
            & (SHEET_Y[0] < y)
            & (y < SHEET_Y[1])
            & (SHEET_Z[0] < z)
            & (z < SHEET_Z[1])
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

    # wrap constraints together
    constraint = {
        bc_side.name: bc_side,
        bc_mid.name: bc_mid,
        bc_surface.name: bc_surface,
        sheet_interior_constraint.name: sheet_interior_constraint,
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
        "./datasets/data/sheet_input.txt",
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
        ppsci.loss.MSELoss("sum"),
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
        save_freq=20,
        log_freq=20,
        eval_during_train=False,
        eval_freq=20,
        seed=SEED,
        equation=equation,
        geom=geom,
        # validator=validator,
        visualizer=visualizer,
        eval_with_no_grad=True,
    )
    # train model
    solver.train()
    # plot losses
    # solver.plot_losses(by_epoch=True)
    # evaluate after finished training
    # solver.eval()
    # visualize prediction after finished training
    # solver.visualize()

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
