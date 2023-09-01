import os

import numpy as np
import paddle

import ppsci
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import reader


def load_n_csv_file(dataset_dir, filename, filenum, data_keys):
    data_list = []
    for i in range(filenum):
        csv_path = os.path.join(dataset_dir, f"{filename}_{i}.csv")
        logger.info(f"Loading csv file: {csv_path}")
        data_dict = reader.load_csv_file(csv_path, data_keys)
        data_list.append(data_dict)
    return data_list


def concate_lists_to_dict(data_list, data_keys):
    data_dict = {}
    for key in data_keys:
        value = np.concatenate([dict[key] for dict in data_list], axis=0)
        data_dict.update({key: value})
    return data_dict


def load_data(
    dataset_dir, filename, filenum, input_keys, label_keys, weight_keys, vis_keys
):
    data_list = load_n_csv_file(
        dataset_dir, filename, filenum, input_keys + label_keys + weight_keys + vis_keys
    )

    input_dict = concate_lists_to_dict(data_list, input_keys)
    label_dict = concate_lists_to_dict(data_list, label_keys)
    weight_dict = concate_lists_to_dict(data_list, weight_keys)
    vis_dict = concate_lists_to_dict(data_list, vis_keys)

    return input_dict, label_dict, weight_dict, vis_dict


if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    DATASET_DIR = "./datasets/data_inverse/"
    OUTPUT_DIR = (
        "./output_inverse_ca_2m_test/" if args.output_dir is None else args.output_dir
    )

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set geometry
    control_arm = ppsci.geometry.Mesh("./datasets/stl/control_arm.stl")
    # geometry bool operation
    geo = control_arm
    geom = {"geo": geo}

    # set training hyper-parameters
    # MARK
    DEEP = 6
    WIDTH = 512
    LR = 4 * 1e-4
    DECAY_STEPS = 100

    ITERS_PER_EPOCH = 1
    EPOCHS = 200
    BATCH_SIZE_TRAIN = 200000
    BATCH_SIZE_CONSTRAINT = 8192
    use_para = True
    plt_name = "vis"
    loss_str = "mean"
    loss_fn = ppsci.loss.L1Loss("mean")

    DATASET_NAME = "train_mult_type_10000"
    DATASET_NUM = 20

    # set keys
    input_keys = (
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
    )
    label_keys = (
        "equilibrium_x",
        "equilibrium_y",
        "equilibrium_z",
        "stress_disp_xx",
        "stress_disp_yy",
        "stress_disp_zz",
        "stress_disp_xy",
        "stress_disp_xz",
        "stress_disp_yz",
    )

    # set model
    inverse_net_lambda = ppsci.arch.MLP(
        input_keys,
        ("lambda_",),
        DEEP,
        WIDTH,
        "silu",
        weight_norm=True,
    )
    inverse_net_mu = ppsci.arch.MLP(
        input_keys,
        ("mu",),
        DEEP,
        WIDTH,
        "silu",
        weight_norm=True,
    )
    # wrap to a model_list
    model = ppsci.arch.ModelList((inverse_net_lambda, inverse_net_mu))

    # set optimizer
    lr_scheduler = ppsci.optimizer.lr_scheduler.ExponentialDecay(
        EPOCHS,
        1,
        LR,
        0.95,
        DECAY_STEPS,
        by_epoch=False,
    )()
    optimizer = ppsci.optimizer.Adam(lr_scheduler)((model,))
    # MAX_ITER = 50000
    # optimizer = ppsci.optimizer.LBFGS(max_iter=MAX_ITER)((model,))

    # load train dataset
    input_dict, label_dict, weight_dict, vis_dict = load_data(
        DATASET_DIR,
        DATASET_NAME,
        DATASET_NUM,
        input_keys,
        label_keys,
        ("sdf",),
        ("x", "y", "z"),
    )
    train_dataloader_cfg = {
        "dataset": {
            "name": "NamedArrayDataset",
            "input": input_dict,
            "label": label_dict,
            "weight": {
                "equilibrium_x": weight_dict["sdf"],
                "equilibrium_y": weight_dict["sdf"],
                "equilibrium_z": weight_dict["sdf"],
                "stress_disp_xx": weight_dict["sdf"],
                "stress_disp_yy": weight_dict["sdf"],
                "stress_disp_zz": weight_dict["sdf"],
                "stress_disp_xy": weight_dict["sdf"],
                "stress_disp_xz": weight_dict["sdf"],
                "stress_disp_yz": weight_dict["sdf"],
            },
        },
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

    # load test dataset
    pred_input_dict, _, _, vis_input_dict = load_data(
        DATASET_DIR,
        "test_20000",
        1,
        input_keys,
        label_keys,
        ("sdf",),
        ("x", "y", "z"),
    )

    # set visualizer(optional)
    vis_input_dict.update(pred_input_dict)

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
        save_freq=200,
        log_freq=200,
        eval_during_train=True,
        eval_freq=min(50, EPOCHS),
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
    solver.plot_losses(by_epoch=True, smooth_step=1)

    # evaluate after finished training
    solver.eval()

    # visualize prediction after finished training
    solver.visualize()

    # prediction
    pred_dict = solver.predict(pred_input_dict)
    print("lambda:", pred_dict["lambda_"].numpy().mean())
    print("mu:", pred_dict["mu"].numpy().mean())
