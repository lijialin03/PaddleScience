import numpy as np
import paddle

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

# for test
loss_str = "mean"


def loss_func_panel_top(output_dict, label_dict, weight_dict, input_dict):
    label_dict["traction_y"] = input_dict["sigma_hoop"]
    # print(input_dict["sigma_hoop"].mean(), output_dict["traction_y"].mean())
    loss_fn = ppsci.loss.MSELoss("mean")
    losses = loss_fn(output_dict, label_dict, weight_dict)
    return losses


if __name__ == "__main__":
    args = config.parse_args()
    # set random seed for reproducibility
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    np.random.seed(SEED)
    # set output directory
    OUTPUT_DIR = (
        "./output_fuselage_panel_test" if not args.output_dir else args.output_dir
    )
    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # Specify parameters
    NU = 0.33
    E = 73.0e9
    LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))
    MU_REAL = E / (2 * (1 + NU))
    LAMBDA_ = LAMBDA_ / MU_REAL
    MU = 1.0

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=2
        )
    }

    # set geometry
    PANEL_ORIGIN = (-0.5, -0.9)
    PANEL_DIM = (1, 1.8)  # Panel width is the characteristic length.
    WINDOW_ORIGIN = (-0.125, -0.2)
    WINDOW_DIM = (0.25, 0.4)
    PANEL_AUX1_ORIGIN = (-0.075, -0.2)
    PANEL_AUX1_DIM = (0.15, 0.4)
    PANEL_AUX2_ORIGIN = (-0.125, -0.15)
    PANEL_AUX2_DIM = (0.25, 0.3)
    HR_ZONE_ORIGIN = (-0.2, -0.4)
    HR_ZONE_DIM = (0.4, 0.8)
    CIRCLE_NW_CENTER = (-0.075, 0.15)
    CIRCLE_NE_CENTER = (0.075, 0.15)
    CIRCLE_SE_CENTER = (0.075, -0.15)
    CIRCLE_SW_CENTER = (-0.075, -0.15)
    CIRCLE_RADIUS = 0.05

    panel = ppsci.geometry.Rectangle(
        PANEL_ORIGIN, (PANEL_ORIGIN[0] + PANEL_DIM[0], PANEL_ORIGIN[1] + PANEL_DIM[1])
    )
    window = ppsci.geometry.Rectangle(
        WINDOW_ORIGIN,
        (WINDOW_ORIGIN[0] + WINDOW_DIM[0], WINDOW_ORIGIN[1] + WINDOW_DIM[1]),
    )
    panel_aux1 = ppsci.geometry.Rectangle(
        PANEL_AUX1_ORIGIN,
        (
            PANEL_AUX1_ORIGIN[0] + PANEL_AUX1_DIM[0],
            PANEL_AUX1_ORIGIN[1] + PANEL_AUX1_DIM[1],
        ),
    )
    panel_aux2 = ppsci.geometry.Rectangle(
        PANEL_AUX2_ORIGIN,
        (
            PANEL_AUX2_ORIGIN[0] + PANEL_AUX2_DIM[0],
            PANEL_AUX2_ORIGIN[1] + PANEL_AUX2_DIM[1],
        ),
    )
    hr_zone = ppsci.geometry.Rectangle(
        HR_ZONE_ORIGIN,
        (HR_ZONE_ORIGIN[0] + HR_ZONE_DIM[0], HR_ZONE_ORIGIN[1] + HR_ZONE_DIM[1]),
    )
    circle_nw = ppsci.geometry.Disk(CIRCLE_NW_CENTER, CIRCLE_RADIUS)
    circle_ne = ppsci.geometry.Disk(CIRCLE_NE_CENTER, CIRCLE_RADIUS)
    circle_se = ppsci.geometry.Disk(CIRCLE_SE_CENTER, CIRCLE_RADIUS)
    circle_sw = ppsci.geometry.Disk(CIRCLE_SW_CENTER, CIRCLE_RADIUS)

    # geometry bool operation
    corners = (
        window - panel_aux1 - panel_aux2 - circle_nw - circle_ne - circle_se - circle_sw
    )
    window = window - corners
    geo = panel - window
    hr_geo = geo & hr_zone
    geom = {"geo": geo}

    # Parameterization
    CHARACTERISTIC_LENGTH = PANEL_DIM[0]
    CHARACTERISTIC_DISPLACEMENT = 0.001 * WINDOW_DIM[0]
    SIGMA_NORMALIZATION = CHARACTERISTIC_LENGTH / (
        MU_REAL * CHARACTERISTIC_DISPLACEMENT
    )
    SIGMA_HOOP_LOWER = 46 * 10**6 * SIGMA_NORMALIZATION
    SIGMA_HOOP_UPPER = 56.5 * 10**6 * SIGMA_NORMALIZATION
    SIGMA_HOOP_RANGE = (SIGMA_HOOP_LOWER, SIGMA_HOOP_UPPER)
    PARAM_RANGES = SIGMA_HOOP_RANGE
    INFERENCE_PARAM_RANGES = 46 * 10**6 * SIGMA_NORMALIZATION

    # bounds
    BOUNDS_X = (PANEL_ORIGIN[0], PANEL_ORIGIN[0] + PANEL_DIM[0])
    BOUNDS_Y = (PANEL_ORIGIN[1], PANEL_ORIGIN[1] + PANEL_DIM[1])
    HR_BOUNDS_X = (HR_ZONE_ORIGIN[0], HR_ZONE_ORIGIN[0] + HR_ZONE_DIM[0])
    HR_BOUNDS_Y = (HR_ZONE_ORIGIN[1], HR_ZONE_ORIGIN[1] + HR_ZONE_DIM[1])

    # set model
    elasticity_net = ppsci.arch.MLP(
        ("x", "y", "sigma_hoop"),
        ("u", "v", "sigma_xx", "sigma_yy", "sigma_xy"),
        6,
        512,
        "silu",
        weight_norm=True,
    )

    def transtorm_in(_in):
        # test: sigma_hoop is a fix value
        # sigma_hoop = paddle.full_like(
        #     _in["x"],
        #     INFERENCE_PARAM_RANGES,
        #     dtype=paddle.get_default_dtype(),
        # )

        sigma_hoop = (
            paddle.rand(_in["x"].shape, dtype=paddle.get_default_dtype())
            * (PARAM_RANGES[1] - PARAM_RANGES[0])
            + PARAM_RANGES[0]
        )
        sigma_hoop.stop_gradient = False
        _in.update({"sigma_hoop": sigma_hoop})
        return _in

    elasticity_net.register_input_transform(transtorm_in)
    model = elasticity_net

    # set training hyper-parameters
    ITERS_PER_EPOCH = 10
    EPOCHS = 200 if not args.epochs else args.epochs
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        EPOCHS,
        ITERS_PER_EPOCH,
        0.001,
        0.95,
        15000,
        by_epoch=False,
    )()

    # set optimizer
    optimizer = ppsci.optimizer.Adam(lr_scheduler)(model)

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
    panel_left_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 250},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y: x == PANEL_ORIGIN[0],
        name="PANEL_LEFT",
    )
    panel_right_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 250},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y: x == PANEL_ORIGIN[0] + PANEL_DIM[0],
        name="PANEL_RIGHT",
    )
    panel_bottom_constraint = ppsci.constraint.BoundaryConstraint(
        {"v": lambda d: d["v"]},
        {"v": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 150},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y: y == PANEL_ORIGIN[1],
        name="PANEL_BOTTOM",
    )
    panel_corner_constraint = ppsci.constraint.BoundaryConstraint(
        {"u": lambda d: d["u"]},
        {"u": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 5},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y: np.logical_and(
            x == PANEL_ORIGIN[0], y > PANEL_ORIGIN[1], y < PANEL_ORIGIN[1] + 1e-3
        ),
        name="PANEL_CORNER",
    )
    panel_top_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0},  # "traction_y" is a fake value
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 150},
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(loss_func_panel_top),
        criteria=lambda x, y: y == PANEL_ORIGIN[1] + PANEL_DIM[1],
        name="PANEL_TOP",
    )

    panel_window_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0},
        window,
        {**train_dataloader_cfg, "batch_size": 3500},
        ppsci.loss.MSELoss(loss_str),
        name="PANEL_WINDOW",
    )

    # low-resolution interior
    low_res_interior_constraint = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_xy": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": 7000},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
        ),
        weight_dict={
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_xy": "sdf",
        },
        name="LOW_RES_INTERIOR",
    )

    # high-resolution interior
    high_res_interior_constraint = ppsci.constraint.InteriorConstraint(
        equation["LinearElasticity"].equations,
        {
            "equilibrium_x": 0,
            "equilibrium_y": 0,
            "stress_disp_xx": 0,
            "stress_disp_yy": 0,
            "stress_disp_xy": 0,
        },
        hr_geo,
        {**train_dataloader_cfg, "batch_size": 4000},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y: (
            (HR_BOUNDS_X[0] < x)
            & (x < HR_BOUNDS_X[1])
            & (HR_BOUNDS_Y[0] < y)
            & (y < HR_BOUNDS_Y[1])
        ),
        weight_dict={
            "equilibrium_x": "sdf",
            "equilibrium_y": "sdf",
            "stress_disp_xx": "sdf",
            "stress_disp_yy": "sdf",
            "stress_disp_xy": "sdf",
        },
        name="HIGH_RES_INTERIOR",
    )

    # wrap constraints together
    constraint = {
        panel_left_constraint.name: panel_left_constraint,
        panel_right_constraint.name: panel_right_constraint,
        panel_bottom_constraint.name: panel_bottom_constraint,
        panel_corner_constraint.name: panel_corner_constraint,
        panel_top_constraint.name: panel_top_constraint,
        panel_window_constraint.name: panel_window_constraint,
        low_res_interior_constraint.name: low_res_interior_constraint,
        high_res_interior_constraint.name: high_res_interior_constraint,
    }

    # add inferencer data
    BATCH_SIZE_PRED = 100000
    pred_input_dict = geom["geo"].sample_interior(
        BATCH_SIZE_PRED,
        criteria=lambda x, y: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
        ),
    )

    # set visualizer(optional)
    visualizer = {
        "visulzie_u_v_sigmas": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "sigma_xx": lambda out: out["sigma_xx"],
                "sigma_yy": lambda out: out["sigma_yy"],
                "sigma_xy": lambda out: out["sigma_xy"],
            },
            prefix="vis_fuselage_panel",
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
        save_freq=100,
        log_freq=500,
        eval_during_train=False,
        eval_freq=100,
        seed=SEED,
        equation=equation,
        geom=geom,
        # validator=validator,
        visualizer=visualizer,
        # eval_with_no_grad=True,
        pretrained_model_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )
    # train model
    # solver.train()
    # plot losses
    # solver.plot_losses(by_epoch=True, smooth_step=10)

    # evaluate after finished training
    # solver.eval()
    # visualize prediction after finished training
    solver.visualize()

    # # read inf data
    # inf_data = ppsci.utils.reader.load_vtp("./datasets/data_fp/inf_data.vtp")
    # print(type(inf_data))

    # predict
    def transtorm_in_pred(_in):
        sigma_hoop = paddle.full_like(
            _in["x"],
            INFERENCE_PARAM_RANGES,
            dtype=paddle.get_default_dtype(),
        )
        _in.update({"sigma_hoop": sigma_hoop})
        return _in

    elasticity_net.register_input_transform(transtorm_in_pred)

    pred_dict = solver.predict(
        input_dict=pred_input_dict,
        expr_dict={
            "u": lambda out: out["u"],
            "v": lambda out: out["v"],
            "sigma_xx": lambda out: out["sigma_xx"],
            "sigma_yy": lambda out: out["sigma_yy"],
            "sigma_xy": lambda out: out["sigma_xy"],
        },
        batch_size=BATCH_SIZE_PRED,
    )
    for key in pred_dict:
        print(key, ":", pred_dict[key].numpy().mean())
