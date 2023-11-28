from os import makedirs
from os import path as osp

import functions as func_module
import hydra
import model as model_module
import numpy as np
import paddle
import problems as problems_module
from omegaconf import DictConfig

import ppsci
from ppsci.utils import logger
from ppsci.utils import save_load


def train(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # make dirs
    makedirs(cfg.output_dir_disp, exist_ok=True)
    makedirs(cfg.output_dir_density, exist_ok=True)

    # set problem
    problem = getattr(problems_module, cfg.PROBLEM)(cfg)

    # set model
    problem.disp_net = model_module.DenseSIRENModel(**cfg.MODEL.disp_net)
    problem.density_net = model_module.DenseSIRENModel(**cfg.MODEL.density_net)

    # set transforms
    problem.disp_net.register_input_transform(problem.transform_in)
    problem.disp_net.register_output_transform(problem.transform_out_disp)
    problem.density_net.register_input_transform(problem.transform_in)
    problem.density_net.register_output_transform(problem.transform_out_density)

    # set optimizer
    optimizer_disp = ppsci.optimizer.Adam(**cfg.TRAIN.disp_net.optimizer)(
        problem.disp_net
    )
    optimizer_density = ppsci.optimizer.Adam(**cfg.TRAIN.density_net.optimizer)(
        problem.density_net
    )

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "sampler": {
            "name": "DistributedBatchSampler"
            if cfg.TRAIN.enable_parallel
            else "BatchSampler",
            "drop_last": True,
            "shuffle": False,
        },
        "num_workers": 0,
    }

    # set constraint
    # interior_disp = func_module.StratifiedInteriorConstraint(
    #     problem.equation["EnergyEquation"].equations,
    #     {"energy_xyz": 0} if problem.dim == 3 else {"energy_xy": 0},
    #     problem.geom["geo"],
    #     {
    #         **train_dataloader_cfg,
    #         "batch_size": cfg.TRAIN.batch_size.constraint,
    #         "iters_per_epoch": cfg.TRAIN.disp_net.iters_per_epoch,
    #     },
    #     # ppsci.loss.MSELoss(loss_str),
    #     ppsci.loss.FunctionalLoss(problem.disp_loss_func),
    #     bounds=(
    #         (problem.geo_origin[0], problem.geo_dim[0]),
    #         (problem.geo_origin[1], problem.geo_dim[1]),
    #     ),
    #     n_samples=problem.batch_size,
    #     name="INTERIOR_DISP",
    # )

    interior_disp = ppsci.constraint.InteriorConstraint(
        problem.equation["EnergyEquation"].equations,
        {"energy_xyz": 0} if problem.dim == 3 else {"energy_xy": 0},
        problem.geom["geo"],
        {
            **train_dataloader_cfg,
            "batch_size": cfg.TRAIN.batch_size.constraint,
            "iters_per_epoch": cfg.TRAIN.disp_net.iters_per_epoch,
        },
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(problem.disp_loss_func),
        name="INTERIOR_DISP",
    )

    # re-assign to ITERS_PER_EPOCH_DISP
    if cfg.TRAIN.enable_parallel:
        cfg.TRAIN.disp_net.iters_per_epoch = len(interior_disp.data_loader)

    # wrap constraints together
    constraint_disp = {interior_disp.name: interior_disp}

    # interior_density = func_module.StratifiedInteriorConstraint(
    #     {
    #         "densities": lambda out: out["densities"],
    #     },
    #     {"densities": 0},
    #     problem.geom["geo"],
    #     {
    #         **train_dataloader_cfg,
    #         "batch_size": cfg.TRAIN.batch_size.constraint,
    #         "iters_per_epoch": cfg.TRAIN.density_net.iters_per_epoch,
    #     },
    #     # ppsci.loss.MSELoss(loss_str),
    #     ppsci.loss.FunctionalLossBatch(problem.density_loss_func),
    #     bounds=(
    #         (problem.geo_origin[0], problem.geo_dim[0]),
    #         (problem.geo_origin[1], problem.geo_dim[1]),
    #     ),
    #     n_samples=problem.batch_size,
    #     name="INTERIOR_DENSITY",
    # )

    sampler = func_module.Sampler(
        problem.geom["geo"],
        bounds=(
            (problem.geo_origin[0], problem.geo_dim[0]),
            (problem.geo_origin[1], problem.geo_dim[1]),
        ),
    )
    input, mask = sampler.sample_interior_stratified(
        n_samples=problem.batch_size, n_iter=cfg.TRAIN.density_net.iters_per_epoch
    )  # TODO: Use mask to filter samples that do not need to participate in loss calculation
    interior_density = ppsci.constraint.SupervisedConstraint(
        {
            "dataset": {"name": "NamedArrayDataset", "input": input},
            "sampler": {
                "name": "DistributedBatchSampler"
                if cfg.TRAIN.enable_parallel
                else "BatchSampler",
                "drop_last": True,
                "shuffle": False,
            },
            "num_workers": 0,
            "batch_size": cfg.TRAIN.batch_size.constraint,
            # "iters_per_epoch": cfg.TRAIN.density_net.iters_per_epoch,
        },
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLossBatch(problem.density_loss_func),
        {
            "densities": lambda out: out["densities"],
        },
        name="INTERIOR_DENSITY",
    )

    # interior_density = ppsci.constraint.InteriorConstraint(
    #     {
    #         "densities": lambda out: out["densities"],
    #     },
    #     {"densities": 0},
    #     problem.geom["geo"],
    #     {
    #         **train_dataloader_cfg,
    #         "batch_size": cfg.TRAIN.batch_size.constraint,
    #         "iters_per_epoch": cfg.TRAIN.density_net.iters_per_epoch,
    #     },
    #     # ppsci.loss.MSELoss(loss_str),
    #     ppsci.loss.FunctionalLossBatch(problem.density_loss_func),
    #     name="INTERIOR_DENSITY",
    # )

    constraint_density = {interior_density.name: interior_density}

    # set visualizer(optional)
    pred_input_keys = ("x", "y")
    if problem.dim == 3:
        pred_input_keys += ("z",)

    # add inferencer data
    samplers = problem.geom["geo"].sample_interior(cfg.TRAIN.batch_size.visualizer)
    pred_input_dict = {}
    for key in pred_input_keys:
        pred_input_dict.update({key: samplers[key]})

    visualizer_disp = {
        "vis_disp": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {key: lambda out, k=key: out[k] for key in cfg.MODEL.disp_net.output_keys},
            prefix="vtu_disp",
        ),
    }
    visualizer_density = {
        "vis_density": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "density": lambda out: out["densities"],
            },
            batch_size=cfg.TRAIN.batch_size.visualizer,
            prefix="vtu_density",
        ),
    }

    # initialize solver
    solver_disp = ppsci.solver.Solver(
        model=problem.disp_net,
        constraint=constraint_disp,
        output_dir=cfg.output_dir_disp,
        optimizer=optimizer_disp,
        epochs=cfg.TRAIN.disp_net.epochs,
        iters_per_epoch=cfg.TRAIN.disp_net.iters_per_epoch,
        seed=cfg.seed,
        equation=problem.equation,
        geom=problem.geom,
        log_freq=cfg.log_freq,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        visualizer=visualizer_disp,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    solver_density = ppsci.solver.Solver(
        model=problem.density_net,
        constraint=constraint_density,
        output_dir=cfg.output_dir_density,
        optimizer=optimizer_density,
        epochs=cfg.TRAIN.density_net.epochs,
        iters_per_epoch=cfg.TRAIN.density_net.iters_per_epoch,
        equation=problem.equation,
        geom=problem.geom,
        log_freq=cfg.log_freq,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        visualizer=visualizer_density,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    # initialize density trainer
    if cfg.USE_MMSE:
        density_trainer = func_module.Trainer(solver_density)

    # training
    solver_disp.train()

    for i in range(1, cfg.TRAIN.epochs + 1):
        ppsci.utils.logger.info(f"\nEpoch: {i}\n")

        solver_disp.train()
        if cfg.USE_MMSE:
            density_trainer.train_batch()
        else:
            solver_density.train()

        # plotting during training
        if i == 1 or i % cfg.log_freq == 0 or i == cfg.TRAIN.epochs:
            visualizer_density["vis_density"].prefix = f"vtu_density_e{i}"
            solver_density.visualize()

            visualizer_disp["vis_disp"].prefix = f"vtu_disp_e{i}"
            solver_disp.visualize()

            save_load.save_checkpoint(
                solver_density.model,
                solver_density.optimizer,
                solver_density.scaler,
                {"metric": "dummy", "epoch": i},
                solver_density.output_dir,
                f"epoch_{i}",
                solver_density.equation,
            )


def evaluate(cfg: DictConfig):
    # set random seed for reproducibility
    ppsci.utils.misc.set_random_seed(cfg.seed)
    # initialize logger
    logger.init_logger("ppsci", osp.join(cfg.output_dir, f"{cfg.mode}.log"), "info")

    # set problem
    problem = getattr(problems_module, cfg.PROBLEM)(cfg)

    # set model
    problem.density_net = model_module.DenseSIRENModel(**cfg.MODEL.density_net)

    # set transforms
    problem.density_net.register_input_transform(problem.transform_in)
    problem.density_net.register_output_transform(problem.transform_out_density)

    if problem.dim == 2:
        # add inferencer data
        samplers = problem.geom["geo"].sample_interior(cfg.EVAL.num_sample)
        pred_input_dict = {}
        if problem.mirror:
            if problem.mirror[0]:
                pred_input_dict["x"] = np.concatenate(
                    [samplers["x"], 2 * problem.geo_dim[0] - samplers["x"]]
                )
                pred_input_dict["y"] = np.concatenate([samplers["y"], samplers["y"]])
            if problem.mirror[1]:
                pred_input_dict["x"] = np.concatenate([samplers["x"], samplers["x"]])
                pred_input_dict["y"] = np.concatenate(
                    [samplers["y"], 2 * problem.geo_dim[1] - samplers["y"]]
                )
        else:
            pred_input_dict["x"] = samplers["x"]
            pred_input_dict["y"] = samplers["y"]

        def compute_mirror_density(problem, out):
            densities = out["densities"][: cfg.EVAL.num_sample]
            if problem.mirror:
                if problem.mirror[0]:
                    densities = paddle.concat([densities, densities])
                if problem.mirror[1]:
                    densities = paddle.concat([densities, densities])
            return densities

        visualizer_density = {
            "vis_density": ppsci.visualize.VisualizerVtu(
                pred_input_dict,
                {"density": lambda out: compute_mirror_density(problem, out)},
                batch_size=pred_input_dict["x"].shape[0],
                prefix="vtu_density",
            ),
        }

        solver_density = ppsci.solver.Solver(
            model=problem.density_net,
            output_dir=cfg.output_dir,
            visualizer=visualizer_density,
            pretrained_model_path=cfg.EVAL.pretrained_model_path,
        )
        solver_density.visualize()
    elif problem.dim == 3:
        # load pretrained model
        save_load.load_pretrain(problem.density_net, cfg.EVAL.pretrained_model_path)
        # plotting
        plot = func_module.Plot(
            osp.join(cfg.output_dir, "density.obj"),
            problem,
            cfg.EVAL.n_cells,
            0.5,
        )
        plot.plot_3d()


@hydra.main(version_base=None, config_path="./conf", config_name="ntopo.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
