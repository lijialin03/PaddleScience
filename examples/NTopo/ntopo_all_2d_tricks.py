import model as model_module
import numpy as np
import paddle
import problems_tricks as problems_module

import ppsci
from ppsci.utils import config
from ppsci.utils import logger

if __name__ == "__main__":
    args = config.parse_args()
    SEED = 42
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR = (
        "./output_ntopo_all_2d_tricks_longbeam_1000_2/"
        # "./output_ntopo_all_2d_tricks_tv_1000_2/"
        # "./output_ntopo_all_2d_tricks_custom_1/"
        if args.output_dir is None
        else args.output_dir
    )

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR}/train.log", "info")

    # set training hyper-parameters
    # MARK
    plt_name = "vis"
    use_para = False
    LR_DISP = 3e-4
    LR_DENSITY = 5e-5

    EPOCHS = 200  # times for for-loop
    EPOCHS_DISP = EPOCHS_DENSITY = 1
    ITERS_PER_EPOCH_DISP = 1000
    ITERS_PER_EPOCH_DENSITY = 50  # times for n_opt_batch
    PRED_INTERVAL = 10

    # set problem
    problem = problems_module.LongBeam2D()
    # problem = problems_module.TriangleVariants2D()
    # problem = problems_module.Custom2D_1()

    # set model
    input_keys = ("x_scaled", "y_scaled", "sin_x_scaled", "sin_y_scaled")
    problem.disp_net = model_module.DenseSIRENModel(
        input_keys, ("u", "v"), 6, 60, 0.001
    )
    problem.density_net = model_module.DenseSIRENModel(
        input_keys, ("density",), 6, 60, 0.001
    )

    # input transform
    problem.disp_net.register_input_transform(problem.transform_in)
    problem.disp_net.register_output_transform(problem.transform_out_disp)

    problem.density_net.register_input_transform(problem.transform_in)
    problem.density_net.register_output_transform(problem.transform_out_density)

    # set optimizer
    optimizer_disp = ppsci.optimizer.Adam(learning_rate=LR_DISP, beta2=0.99)(
        (problem.disp_net,)
    )
    optimizer_density = ppsci.optimizer.Adam(
        learning_rate=LR_DENSITY, beta1=0.8, beta2=0.9
    )((problem.density_net,))

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "iters_per_epoch": ITERS_PER_EPOCH_DISP,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "num_workers": 1,
    }
    batch_size = {"bs": problem.batch_size}

    # set constraint
    interior_disp = ppsci.constraint.InteriorConstraint(
        problem.equation["EnergyEquation"].equations,
        {"energy_xy": 0},
        problem.geom["geo"],
        {**train_dataloader_cfg, "batch_size": batch_size["bs"]},
        ppsci.loss.FunctionalLoss(problem.disp_loss_func),
        name="INTERIOR_DISP",
    )

    interior_density = ppsci.constraint.InteriorConstraint(
        {
            "densities": lambda out: out["densities"],
        },
        {"densities": 0},
        problem.geom["geo"],
        {
            **train_dataloader_cfg,
            "batch_size": batch_size["bs"],
            "iters_per_epoch": ITERS_PER_EPOCH_DENSITY,
        },
        ppsci.loss.FunctionalLossBatch(problem.density_loss_func),
        name="INTERIOR_DENSITY",
    )

    # re-assign to ITERS_PER_EPOCH
    # if use_para:
    #     ITERS_PER_EPOCH_DISP = len(bc_left_disp.data_loader)
    #     ITERS_PER_EPOCH_DENSITY = len(interior_density.data_loader)

    # wrap constraints together
    constraint_disp = {
        # bc_left_disp.name: bc_left_disp,
        # bc_right_corner_disp.name: bc_right_corner_disp,
        interior_disp.name: interior_disp,
    }
    constraint_density = {interior_density.name: interior_density}

    # set visualizer(optional)
    # add inferencer data
    BATCH_SIZE_PRED = problem.batch_size
    pred_input_dict = problem.geom["geo"].sample_interior(BATCH_SIZE_PRED)
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
        "batch_size": BATCH_SIZE_PRED,
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
        metric={"mean": ppsci.metric.FunctionalMetric(problem.density_metric_func)},
        name="eval",
    )
    validator_density = {sup_validator.name: sup_validator}

    visualizer_disp = {
        "vis": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
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
        model=problem.disp_net,
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
        equation=problem.equation,
        geom=problem.geom,
        # validator=validator,
        visualizer=visualizer_disp,
        # eval_with_no_grad=True,
        # pretrained_model_path="./init_params/paddle_init_only_disp",
    )

    solver_density = ppsci.solver.Solver(
        model=problem.density_net,
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
        equation=problem.equation,
        geom=problem.geom,
        validator=validator_density,
        visualizer=visualizer_density,
        eval_with_no_grad=True,
        # pretrained_model_path="./init_params/paddle_init_only_density",
        # pretrained_model_path=f"{OUTPUT_DIR}checkpoints/latest",
    )

    # pre-processing
    solver_disp.train()
    # solver_disp.visualize()

    # for name, tensor in density_net.named_parameters():
    #     if name == "linears.5.linear.bias":
    #         print(name, tensor.grad)

    # PRED_INTERVAL = 1
    for i in range(1, EPOCHS + 1):
        ppsci.utils.logger.info(f"\nEpoch: {i}\n")
        solver_disp.train()
        solver_density.train_batch()

        # plotting during training
        if i == 1 or i % PRED_INTERVAL == 0 or i == EPOCHS:
            solver_density.eval()
            visualizer_density["vis"].prefix = plt_name + f"_density_e{i}"
            solver_density.visualize()
            visualizer_disp["vis"].prefix = plt_name + f"_disp_e{i}"
            solver_disp.visualize()

    # # plot losses
    # solver_disp.plot_losses(by_epoch=True, smooth_step=1)
    # solver_density.plot_losses(by_epoch=False, smooth_step=10)
