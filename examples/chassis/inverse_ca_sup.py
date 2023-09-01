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
    save_load.load_pretrain(
        model=model_forward, path="./saved_model/control_arm_3_4/epoch_2000"
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
        "./output_inverse_ca_1/" if args.output_dir is None else args.output_dir
    )

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set geometry
    control_arm = ppsci.geometry.Mesh("./datasets/stl/control_arm.stl")
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}
    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

    # set training hyper-parameters
    # MARK
    DEEP = 6
    WIDTH = 512
    LR = 4 * 1e-3
    DECAY_STEPS = 100

    ITERS_PER_EPOCH = 1
    EPOCHS = 1
    MAX_ITER = 50000
    BATCH_SIZE_TRAIN = 12000
    BATCH_SIZE_CONSTRAINT = 1024
    BATCH_SIZE_PRED = 1000
    plt_name = "vis"
    use_para = False
    loss_str = "mean"
    loss_fn = ppsci.loss.L1Loss("mean")

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
        1,
        LR,
        0.95,
        DECAY_STEPS,
        by_epoch=False,
    )()

    # optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))
    optimizer = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)((model,))

    # add input data
    input_sample = geom["geo"].sample_interior(
        BATCH_SIZE_TRAIN,
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

    train_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": {
                "equilibrium_x": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
                "equilibrium_y": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
                "equilibrium_z": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
                "stress_disp_xx": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
                "stress_disp_yy": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
                "stress_disp_zz": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
                "stress_disp_xy": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
                "stress_disp_xz": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
                "stress_disp_yz": np.full(
                    (BATCH_SIZE_TRAIN, 1), 0, dtype=paddle.get_default_dtype()
                ),
            },
            "weight": {
                "equilibrium_x": input_sample["sdf"],
                "equilibrium_y": input_sample["sdf"],
                "equilibrium_z": input_sample["sdf"],
                "stress_disp_xx": input_sample["sdf"],
                "stress_disp_yy": input_sample["sdf"],
                "stress_disp_zz": input_sample["sdf"],
                "stress_disp_xy": input_sample["sdf"],
                "stress_disp_xz": input_sample["sdf"],
                "stress_disp_yz": input_sample["sdf"],
            },
        },
        "batch_size": BATCH_SIZE_TRAIN,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    equation = {
        "LinearElasticity_v2": ppsci.equation.LinearElasticity_v2(param_dict={}, dim=3)
    }

    sup_constraint = ppsci.constraint.SupervisedConstraint(
        {**train_dataloader_cfg, "batch_size": BATCH_SIZE_CONSTRAINT},
        # ppsci.loss.L1Loss(loss_str),
        loss_fn,
        equation["LinearElasticity_v2"].equations,
        name="sup_constraint",
    )

    # re-assign to ITERS_PER_EPOCH
    if use_para:
        ITERS_PER_EPOCH = len(sup_constraint.data_loader)

    constraint = {sup_constraint.name: sup_constraint}

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": {
                "lambda_": np.full(
                    (BATCH_SIZE_TRAIN, 1), 1.5, dtype=paddle.get_default_dtype()
                ),
                "mu": np.full(
                    (BATCH_SIZE_TRAIN, 1), 1.0, dtype=paddle.get_default_dtype()
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
        {**eval_dataloader_cfg, "batch_size": BATCH_SIZE_CONSTRAINT},
        ppsci.loss.L1Loss(loss_str),
        # loss_fn,
        {
            "lambda_": lambda out: out["lambda_"],
            "mu": lambda out: out["mu"],
        },
        metric={"MAE": ppsci.metric.MAE()},
        name="sup_eval",
    )
    validator = {sup_validator.name: sup_validator}

    # add inferencer data
    pred_sample = geom["geo"].sample_interior(
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

    # set visualizer(optional)
    vis_input_dict = {
        "x": input_sample["x"],
        "y": input_sample["y"],
        "z": input_sample["z"],
    }

    visualizer = {
        "visulzie_lambda_mu": ppsci.visualize.VisualizerVtu(
            vis_input_dict,
            {
                "lambda": lambda out: out["lambda_"],
                "mu": lambda out: out["mu"],
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
        eval_during_train=True,
        eval_freq=min(20, EPOCHS),
        seed=SEED,
        equation=equation,
        geom=geom,
        validator=validator,
        visualizer=visualizer,
        eval_with_no_grad=True,
        # checkpoint_path=f"{OUTPUT_DIR}/checkpoints/latest",
        # checkpoint_path=f"{OUTPUT_DIR}/checkpoints/epoch_500",
    )

    # train model
    solver.train()

    # plot losses
    solver.plot_losses(by_epoch=True, smooth_step=10)

    # evaluate after finished training
    solver.eval()

    # # visualize prediction after finished training
    # solver.visualize()

    # # prediction
    # pred_sample = geom["geo"].sample_interior(
    #     BATCH_SIZE_PRED,
    #     criteria=lambda x, y, z: (
    #         (BOUNDS_X[0] < x)
    #         & (x < BOUNDS_X[1])
    #         & (BOUNDS_Y[0] < y)
    #         & (y < BOUNDS_Y[1])
    #         & (BOUNDS_Z[0] < z)
    #         & (z < BOUNDS_Z[1])
    #     ),
    # )

    # get data from loaded model
    pred_input_dict = gen_input_data(pred_sample)

    pred_dict = solver.predict(pred_input_dict)
    print("lambda:", pred_dict["lambda_"].numpy().mean())
    print("mu:", pred_dict["mu"].numpy().mean())
