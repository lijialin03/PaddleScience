from os import makedirs
from os import path as osp

import functions as func_module
import hydra
import model as model_module
import numpy as np
import paddle
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

    # specify parameters
    LAMBDA_ = cfg.NU * cfg.E / ((1 + cfg.NU) * (1 - 2 * cfg.NU))
    MU = cfg.E / (2 * (1 + cfg.NU))

    # set equation
    equation = {
        "EnergyEquation": ppsci.equation.EnergyEquation(
            param_dict={"lambda_": LAMBDA_, "mu": MU},
            dim=3,
        ),
    }

    # set geometry
    beam = ppsci.geometry.Cuboid(cfg.GEO_ORIGIN, cfg.GEO_DIM)
    # geometry bool operation
    geo = beam
    geom = {"geo": geo}

    # set model
    disp_net = model_module.DenseSIRENModel(**cfg.MODEL.disp_net)
    density_net = model_module.DenseSIRENModel(**cfg.MODEL.density_net)

    # set transforms
    transforms = func_module.Transform(cfg)
    disp_net.register_input_transform(transforms.transform_in)
    disp_net.register_output_transform(transforms.transform_out_disp)

    density_net.register_input_transform(transforms.transform_in)
    density_net.register_output_transform(transforms.transform_out_density)

    # set optimizer
    optimizer_disp = ppsci.optimizer.Adam(**cfg.TRAIN.disp_net.optimizer)(disp_net)
    optimizer_density = ppsci.optimizer.Adam(**cfg.TRAIN.density_net.optimizer)(
        density_net
    )

    # set functions
    functions = func_module.Funcs(cfg, geo, disp_net, density_net)

    # set dataloader config
    train_dataloader_cfg = {
        "dataset": "NamedArrayDataset",
        "sampler": {
            "name": "BatchSampler",
            "drop_last": True,
            "shuffle": True,
        },
        "num_workers": 0,
    }

    # set constraint
    interior_disp = ppsci.constraint.InteriorConstraint(
        equation["EnergyEquation"].equations,
        {"energy_xyz": 0},
        geom["geo"],
        {
            **train_dataloader_cfg,
            "batch_size": cfg.TRAIN.batch_size.constraint,
            "iters_per_epoch": cfg.TRAIN.disp_net.iters_per_epoch,
        },
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLoss(functions.disp_loss_func),
        name="INTERIOR_DISP",
    )

    # re-assign to ITERS_PER_EPOCH_DISP
    if cfg.TRAIN.enable_parallel:
        cfg.TRAIN.disp_net.iters_per_epoch = len(interior_disp.data_loader)

    # wrap constraints together
    constraint_disp = {interior_disp.name: interior_disp}

    interior_density = ppsci.constraint.InteriorConstraint(
        {
            "densities": lambda out: out["densities"],
        },
        {"densities": 0},
        geom["geo"],
        {
            **train_dataloader_cfg,
            "batch_size": cfg.TRAIN.batch_size.constraint,
            "iters_per_epoch": cfg.TRAIN.density_net.iters_per_epoch,
        },
        # ppsci.loss.MSELoss(loss_str),
        ppsci.loss.FunctionalLossBatch(functions.density_loss_func),
        name="INTERIOR_DENSITY",
    )

    constraint_density = {interior_density.name: interior_density}

    # set visualizer(optional)
    # add inferencer data
    pred_input_dict = geom["geo"].sample_interior(cfg.TRAIN.batch_size.visualizer)
    pred_keys = list(pred_input_dict.keys())
    for key in pred_keys:
        if key not in ("x", "y", "z"):
            pred_input_dict.pop(key)
    visualizer_disp = {
        "vis_disp": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "u": lambda out: out["u"],
                "v": lambda out: out["v"],
                "w": lambda out: out["w"],
            },
            prefix="vtu_disp",
        ),
    }
    visualizer_density = {
        "vis_density": ppsci.visualize.VisualizerVtu(
            pred_input_dict,
            {
                "density": lambda out: out["densities"],
            },
            prefix="vtu_density",
        ),
    }

    # initialize solver
    solver_disp = ppsci.solver.Solver(
        model=disp_net,
        constraint=constraint_disp,
        output_dir=cfg.output_dir_disp,
        optimizer=optimizer_disp,
        epochs=cfg.TRAIN.disp_net.epochs,
        iters_per_epoch=cfg.TRAIN.disp_net.iters_per_epoch,
        seed=cfg.seed,
        equation=equation,
        geom=geom,
        log_freq=cfg.log_freq,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        visualizer=visualizer_disp,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    solver_density = ppsci.solver.Solver(
        model=density_net,
        constraint=constraint_density,
        output_dir=cfg.output_dir_density,
        optimizer=optimizer_density,
        epochs=cfg.TRAIN.density_net.epochs,
        iters_per_epoch=cfg.TRAIN.density_net.iters_per_epoch,
        equation=equation,
        geom=geom,
        log_freq=cfg.log_freq,
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        visualizer=visualizer_density,
        pretrained_model_path=cfg.TRAIN.pretrained_model_path,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    # initialize density trainer
    density_trainer = func_module.Trainer(solver_density)

    # training
    solver_disp.train()

    for i in range(1, cfg.TRAIN.epochs + 1):
        ppsci.utils.logger.info(f"\nEpoch: {i}\n")

        solver_disp.train()
        density_trainer.train_batch()

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

    # set model
    disp_net = model_module.DenseSIRENModel(**cfg.MODEL.disp_net)
    density_net = model_module.DenseSIRENModel(**cfg.MODEL.density_net)

    # set transforms
    transforms = func_module.Transform(cfg)
    disp_net.register_input_transform(transforms.transform_in)
    disp_net.register_output_transform(transforms.transform_out_disp)

    density_net.register_input_transform(transforms.transform_in)
    density_net.register_output_transform(transforms.transform_out_density)

    # load pretrained model
    save_load.load_pretrain(
        density_net, "./output_ntopo_3d_density/checkpoints/epoch_100"
    )

    # add inferencer data
    cx = 0.5 * cfg.GEO_DIM[0] / cfg.n_cells[0]
    cy = 0.5 * cfg.GEO_DIM[1] / cfg.n_cells[1]
    cz = 0.5 * cfg.GEO_DIM[2] / cfg.n_cells[2]
    x = np.linspace(
        cfg.GEO_ORIGIN[0] + cx,
        cfg.GEO_DIM[0] - cx,
        num=cfg.n_cells[0],
        dtype=paddle.get_default_dtype(),
    )
    y = np.linspace(
        cfg.GEO_ORIGIN[1] + cy,
        cfg.GEO_DIM[1] - cy,
        num=cfg.n_cells[1],
        dtype=paddle.get_default_dtype(),
    )
    z = np.linspace(
        cfg.GEO_ORIGIN[2] + cz,
        cfg.GEO_DIM[2] - cz,
        num=cfg.n_cells[2],
        dtype=paddle.get_default_dtype(),
    )
    xs, ys, zs = np.meshgrid(x, y, z, indexing="ij")

    input_dict = {}
    input_dict["x"] = paddle.to_tensor(
        xs.reshape(-1, 1), dtype=paddle.get_default_dtype()
    )
    input_dict["y"] = paddle.to_tensor(
        ys.reshape(-1, 1), dtype=paddle.get_default_dtype()
    )
    input_dict["z"] = paddle.to_tensor(
        zs.reshape(-1, 1), dtype=paddle.get_default_dtype()
    )

    densities = density_net(input_dict)["densities"].numpy().reshape(cfg.n_cells)

    # compute_mirrored_densities
    densities = np.concatenate(
        (densities, densities[:, :, range(cfg.n_cells[2] - 1, -1, -1)]), axis=2
    )

    # plotting
    plot = func_module.Plot(osp.join(cfg.output_dir_density, "density.obj"))
    plot.save_solid(densities, cfg.n_cells, threshold=0.99)


@hydra.main(version_base=None, config_path="./conf", config_name="ntopo_3d.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()
