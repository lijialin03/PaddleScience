import numpy as np
import paddle.nn.functional as F

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

# def sdf_func(self, points: np.ndarray) -> np.ndarray:
#     """Compute signed distance field.

#     Args:
#         points (np.ndarray): The coordinate points used to calculate the SDF value,
#             the shape is [N, 3]

#     Returns:
#         np.ndarray: Unsquared SDF values of input points, the shape is [N, 1].

#     NOTE: This function usually returns ndarray with negative values, because
#     according to the definition of SDF, the SDF value of the coordinate point inside
#     the object(interior points) is negative, the outside is positive, and the edge
#     is 0. Therefore, when used for weighting, a negative sign is often added before
#     the result of this function.
#     """
#     if not checker.dynamic_import_to_globals(["pymesh"]):
#         raise ImportError(
#             "Could not import pymesh python package."
#             "Please install it as https://pymesh.readthedocs.io/en/latest/installation.html."
#         )
#     import pymesh

#     sdf, _, _, _ = pymesh.signed_distance_to_mesh(self.py_mesh, points)
#     sdf = sdf[..., np.newaxis]
#     return sdf


def rest_loss_func(output_dict, label_dict=None, weight_dict={}, input_dict=None):
    mask = F.sigmoid(output_dict["mask"])
    losses = 0.0
    for key in label_dict:
        weight_dict[key] *= mask
        loss = (
            F.mse_loss(output_dict[key], label_dict[key], "none") * weight_dict[key]
        ).mean()
        losses += loss
    return losses


def hole_loss_func(output_dict, label_dict=None, weight_dict={}, input_dict=None):
    hole = 1 - F.sigmoid(output_dict["mask"])
    losses = 0.0
    for key in label_dict:
        weight_dict[key] = hole
        loss = (
            F.mse_loss(output_dict[key], label_dict[key], "none") * weight_dict[key]
        ).mean()
        losses += loss
    return losses


def metrial_loss_func(output_dict, label_dict=None, weight_dict={}, input_dict=None):
    losses = 0.0
    for key in label_dict:
        if key == "mask":
            loss = (
                F.mse_loss(output_dict[key], label_dict[key], "none") * weight_dict[key]
            ).mean()
        else:
            loss = (output_dict[key] * weight_dict[key]).mean()
        losses += loss
    return losses


if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = "./output_tp_ca/" if args.output_dir is None else args.output_dir

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # Specify parameters
    NU = 0.3  # 泊松比
    E = 100.0e9  # 弹性模量
    LAMBDA_ = NU * E / ((1 + NU) * (1 - 2 * NU))  # lambda 拉梅常数之一
    MU = E / (2 * (1 + NU))  # mu 拉梅常数之一
    LAMBDA_ = LAMBDA_ / MU
    MU = 1.0
    T = -100  # 牵引力大小

    # set equation
    equation = {
        "LinearElasticity": ppsci.equation.LinearElasticity(
            E=None, nu=None, lambda_=LAMBDA_, mu=MU, dim=3
        )
    }
    # equation = {
    #     "LinearElasticity": ppsci.equation.LinearElasticity_v2(
    #         param_dict={"lambda_": LAMBDA_, "mu": MU},
    #         dim=3,
    #     )
    # }

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
    EPOCHS = 1000
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
    mask_net = ppsci.arch.MLP(
        ("x", "y", "z"), ("mask",), DEEP, WIDTH, "sigmoid", weight_norm=True
    )
    # wrap to a model_list
    model = ppsci.arch.ModelList((disp_net, stress_net, mask_net))

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
    batch_size = {
        "left": 128,
        "right": 256,
        "surface": 4096,
        "interior": 2048,
        "metirials": 4096,
    }

    # set constraint
    arm_left_constraint = ppsci.constraint.BoundaryConstraint(
        {**equation["LinearElasticity"].equations, "mask": lambda d: d["mask"]},
        {"traction_x": T, "traction_y": 0, "traction_z": 0, "mask": 1},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["left"]},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - CIRCLE_LEFT_CENTER_XY[0])
            + np.square(y - CIRCLE_LEFT_CENTER_XY[1])
        )
        <= CIRCLE_LEFT_RADIUS + 1e-1,
        weight_dict={"mask": 100},
        name="BC_LEFT",
    )
    arm_right_constraint = ppsci.constraint.BoundaryConstraint(
        {
            "u": lambda d: d["u"],
            "v": lambda d: d["v"],
            "w": lambda d: d["w"],
            "mask": lambda d: d["mask"],
        },
        {"u": 0, "v": 0, "w": 0, "mask": 1},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["right"]},
        ppsci.loss.MSELoss(loss_str),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - CIRCLE_RIGHT_CENTER_XZ[0])
            + np.square(z - CIRCLE_RIGHT_CENTER_XZ[1])
        )
        <= CIRCLE_RIGHT_RADIUS + 1e-1,
        weight_dict={"u": 100, "v": 100, "w": 100, "mask": 100},
        name="BC_RIGHT",
    )
    arm_surface_constraint = ppsci.constraint.BoundaryConstraint(
        equation["LinearElasticity"].equations,
        {"traction_x": 0, "traction_y": 0, "traction_z": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["surface"]},
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(rest_loss_func),
        criteria=lambda x, y, z: np.sqrt(
            np.square(x - CIRCLE_LEFT_CENTER_XY[0])
            + np.square(y - CIRCLE_LEFT_CENTER_XY[1])
        )
        > CIRCLE_LEFT_RADIUS + 1e-1,
        name="BC_SURFACE",
    )
    arm_hole_constraint = ppsci.constraint.BoundaryConstraint(
        {
            **equation["LinearElasticity"].equations,
            "u": lambda d: d["u"],
            "v": lambda d: d["v"],
            "w": lambda d: d["w"],
        },
        {
            "u": 0,
            "v": 0,
            "w": 0,
            "traction_x": 0,
            "traction_y": 0,
            "traction_z": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["surface"]},
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(hole_loss_func),
        criteria=lambda x, y, z: np.logical_and(
            (
                np.sqrt(
                    np.square(x - CIRCLE_LEFT_CENTER_XY[0])
                    + np.square(y - CIRCLE_LEFT_CENTER_XY[1])
                )
                > CIRCLE_LEFT_RADIUS + 1e-1
            ),
            (
                np.sqrt(
                    np.square(x - CIRCLE_RIGHT_CENTER_XZ[0])
                    + np.square(z - CIRCLE_RIGHT_CENTER_XZ[1])
                )
                > CIRCLE_RIGHT_RADIUS + 1e-1
            ),
        ),
        weight_dict={
            "u": 100,
            "v": 100,
            "w": 100,
            "traction_x": 100,
            "traction_y": 100,
            "traction_z": 100,
        },
        name="BC_HOLE",
    )
    arm_hole_interior_constraint = ppsci.constraint.InteriorConstraint(
        {
            "u": lambda d: d["u"],
            "v": lambda d: d["v"],
            "w": lambda d: d["w"],
        },
        {
            "u": 0,
            "v": 0,
            "w": 0,
        },
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["interior"]},
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(hole_loss_func),
        criteria=lambda x, y, z: np.logical_and(
            (
                np.sqrt(
                    np.square(x - CIRCLE_LEFT_CENTER_XY[0])
                    + np.square(y - CIRCLE_LEFT_CENTER_XY[1])
                )
                > CIRCLE_LEFT_RADIUS + 1e-1
            ),
            (
                np.sqrt(
                    np.square(x - CIRCLE_RIGHT_CENTER_XZ[0])
                    + np.square(z - CIRCLE_RIGHT_CENTER_XZ[1])
                )
                > CIRCLE_RIGHT_RADIUS + 1e-1
            ),
        ),
        weight_dict={"u": 100, "v": 100, "w": 100},
        name="HOLE_INTERIOR",
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
        {**train_dataloader_cfg, "batch_size": batch_size["interior"]},
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(rest_loss_func),
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

    arm_metrials_constraint = ppsci.constraint.InteriorConstraint(
        {
            "u": lambda d: d["u"],
            "v": lambda d: d["v"],
            "w": lambda d: d["w"],
            "sigma_xx": lambda d: d["sigma_xx"],
            "sigma_yy": lambda d: d["sigma_yy"],
            "sigma_zz": lambda d: d["sigma_zz"],
            "sigma_xy": lambda d: d["sigma_xy"],
            "sigma_xz": lambda d: d["sigma_xz"],
            "sigma_yz": lambda d: d["sigma_yz"],
            "mask": lambda d: F.sigmoid(d["mask"]),
        },
        {"mask": 0},
        geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["metirials"]},
        ppsci.loss.FunctionalLoss(metrial_loss_func),
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
        weight_dict={
            "u": 1,
            "v": 1,
            "w": 1,
            "sigma_xx": 1,
            "sigma_yy": 1,
            "sigma_zz": 1,
            "sigma_xy": 1,
            "sigma_xz": 1,
            "sigma_yz": 1,
            "mask": 1,
        },
        name="M_INTERIOR",
    )

    # re-assign to ITERS_PER_EPOCH
    if use_para:
        ITERS_PER_EPOCH = len(arm_left_constraint.data_loader)
    # wrap constraints together
    constraint = {
        arm_left_constraint.name: arm_left_constraint,
        arm_right_constraint.name: arm_right_constraint,
        arm_surface_constraint.name: arm_surface_constraint,
        arm_hole_constraint.name: arm_hole_constraint,
        arm_interior_constraint.name: arm_interior_constraint,
        arm_hole_interior_constraint.name: arm_hole_interior_constraint,
        arm_metrials_constraint.name: arm_metrials_constraint,
    }

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
            pred_input_dict.pop(key)

    visualizer = {
        "visulzie_u_v_w_sigmas_mask": ppsci.visualize.VisualizerVtu(
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
                "mask": lambda out: out["mask"],
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
        checkpoint_path=f"{OUTPUT_DIR}/checkpoints/latest",
    )

    # train model
    solver.train()

    # plot losses
    solver.plot_losses(by_epoch=True, smooth_step=1)

    # evaluate after finished training
    # solver.eval()

    # visualize prediction after finished training
    solver.visualize()
