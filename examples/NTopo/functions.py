# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import sympy
from paddle.distributed.fleet.utils import hybrid_parallel_util as hpu
from skimage import measure
from sympy.parsing import sympy_parser as sp_parser

import ppsci
from ppsci import geometry
from ppsci.constraint import base
from ppsci.data import dataset
from ppsci.solver import Solver
from ppsci.utils import logger
from ppsci.utils import misc
from ppsci.utils import save_load

if TYPE_CHECKING:
    from ppsci import loss


class Trainer:
    def __init__(self, solver: "Solver") -> None:
        self.solver = solver

    def train_forward(self):
        input_dicts_list = []
        label_dicts_list = []
        weight_dicts_list = []
        output_dicts_list = []
        for _ in range(1, self.solver.iters_per_epoch + 1):
            loss_dict = misc.Prettydefaultdict(float)
            loss_dict["loss"] = 0.0

            input_dicts = []
            label_dicts = []
            weight_dicts = []
            for _, _constraint in self.solver.constraint.items():
                input_dict, label_dict, weight_dict = next(_constraint.data_iter)
                for v in input_dict.values():
                    v.stop_gradient = False

                # gather each constraint's input, label, weight to a list
                input_dicts.append(input_dict)
                label_dicts.append(label_dict)
                weight_dicts.append(weight_dict)

            output_dicts = self.solver.forward_helper.run_forward(
                (
                    _constraint.output_expr
                    for _constraint in self.solver.constraint.values()
                ),
                input_dicts,
                self.solver.model,
                label_dicts,
            )
            input_dicts_list.append(input_dicts)
            label_dicts_list.append(label_dicts)
            weight_dicts_list.append(weight_dicts)
            output_dicts_list.append(output_dicts)
        return input_dicts_list, label_dicts_list, weight_dicts_list, output_dicts_list

    def train_backward(self, constraint_losses_list):
        for iter_id in range(1, self.solver.iters_per_epoch + 1):
            # compute loss for each constraint according to its' own output, label and weight
            total_loss = 0
            for i, _ in enumerate(self.solver.constraint.values()):
                loss = constraint_losses_list[i]
                if isinstance(loss, Callable):
                    total_loss += loss(self.solver.model, iter_id - 1)
                elif isinstance(loss, List):
                    total_loss += loss[iter_id - 1]

            if self.solver.use_amp:
                total_loss_scaled = self.solver.scaler.scale(total_loss)
                total_loss_scaled.backward()
            else:
                total_loss.backward()

            # update parameters
            if (
                iter_id % self.solver.update_freq == 0
                or iter_id == self.solver.iters_per_epoch
            ):
                if self.solver.world_size > 1:
                    # fuse + allreduce manually before optimization if use DDP + no_sync
                    # details in https://github.com/PaddlePaddle/Paddle/issues/48898#issuecomment-1343838622
                    hpu.fused_allreduce_gradients(
                        list(self.solver.model.parameters()), None
                    )
                if self.solver.use_amp:
                    self.solver.scaler.minimize(
                        self.solver.optimizer, total_loss_scaled
                    )
                else:
                    self.solver.optimizer.step()
                self.solver.optimizer.clear_grad()

            # update learning rate by step
            if (
                self.solver.lr_scheduler is not None
                and not self.solver.lr_scheduler.by_epoch
            ):
                self.solver.lr_scheduler.step()

            # update and log training information
            self.solver.global_step += 1

    def train_batch(self):
        """Training."""
        self.solver.global_step = (
            self.solver.best_metric["epoch"] * self.solver.iters_per_epoch
        )

        for epoch_id in range(
            self.solver.best_metric["epoch"] + 1, self.solver.epochs + 1
        ):
            # forward
            (
                input_dicts_list,
                label_dicts_list,
                weight_dicts_list,
                output_dicts_list,
            ) = self.train_forward()

            # batch loss
            constraint_losses_list = []
            for _, _constraint in enumerate(self.solver.constraint.values()):
                if not isinstance(_constraint.loss, ppsci.loss.FunctionalLossBatch):
                    raise TypeError(
                        "Loss function of constraint should be FunctionalLossBatch when using train_batch"
                    )
                constraint_loss_list = _constraint.loss(
                    output_dicts_list,
                    label_dicts_list,
                    weight_dicts_list,
                    input_dicts_list,
                )
                constraint_losses_list.append(constraint_loss_list)

            # backward
            self.train_backward(constraint_losses_list)

            # log training summation at end of a epoch
            metric_msg = ", ".join(
                [
                    self.solver.train_output_info[key].avg_info
                    for key in self.solver.train_output_info
                ]
            )
            logger.info(
                f"[Train][Epoch {epoch_id}/{self.solver.epochs}][Avg] {metric_msg}"
            )
            self.solver.train_output_info.clear()

            cur_metric = float("inf")
            # evaluate during training
            if (
                self.solver.eval_during_train
                and epoch_id % self.solver.eval_freq == 0
                and epoch_id >= self.solver.start_eval_epoch
            ):
                cur_metric = self.solver.eval(epoch_id)
                if cur_metric < self.solver.best_metric["metric"]:
                    self.solver.best_metric["metric"] = cur_metric
                    self.solver.best_metric["epoch"] = epoch_id
                    save_load.save_checkpoint(
                        self.solver.model,
                        self.solver.optimizer,
                        self.solver.scaler,
                        self.solver.best_metric,
                        self.solver.output_dir,
                        "best_model",
                        self.solver.equation,
                    )
                logger.info(
                    f"[Eval][Epoch {epoch_id}]"
                    f"[best metric: {self.solver.best_metric['metric']}]"
                    f"[best epoch: {self.solver.best_metric['epoch']}]"
                    f"[current metric: {cur_metric}]"
                )
                logger.scaler(
                    "eval_metric", cur_metric, epoch_id, self.solver.vdl_writer
                )

                # visualize after evaluation
                if self.solver.visualizer is not None:
                    self.solver.visualize(epoch_id)

            # update learning rate by epoch
            if (
                self.solver.lr_scheduler is not None
                and self.solver.lr_scheduler.by_epoch
            ):
                self.solver.lr_scheduler.step()

            # save epoch model every save_freq epochs
            if self.solver.save_freq > 0 and epoch_id % self.solver.save_freq == 0:
                save_load.save_checkpoint(
                    self.solver.model,
                    self.solver.optimizer,
                    self.solver.scaler,
                    {"metric": cur_metric, "epoch": epoch_id},
                    self.solver.output_dir,
                    f"epoch_{epoch_id}",
                    self.solver.equation,
                )

            # save the latest model for convenient resume training
            save_load.save_checkpoint(
                self.solver.model,
                self.solver.optimizer,
                self.solver.scaler,
                {"metric": cur_metric, "epoch": epoch_id},
                self.solver.output_dir,
                "latest",
                self.solver.equation,
            )

        # close VisualDL
        if self.solver.vdl_writer is not None:
            self.solver.vdl_writer.close()


class Plot:
    def __init__(self, filename, problem, n_cells, threshold=0.25) -> None:
        self.filename = filename
        self.problem = problem
        self.n_cells = n_cells
        self.threshold = threshold

    def prepare_data(self):
        cx = 0.5 * self.problem.geo_dim[0] / self.n_cells[0]
        cy = 0.5 * self.problem.geo_dim[1] / self.n_cells[1]
        cz = 0.5 * self.problem.geo_dim[2] / self.n_cells[2]

        x = np.linspace(
            self.problem.geo_origin[0] + cx,
            self.problem.geo_dim[0] - cx,
            num=self.n_cells[0],
            dtype=paddle.get_default_dtype(),
        )
        y = np.linspace(
            self.problem.geo_origin[1] + cy,
            self.problem.geo_dim[1] - cy,
            num=self.n_cells[1],
            dtype=paddle.get_default_dtype(),
        )
        z = np.linspace(
            self.problem.geo_origin[2] + cz,
            self.problem.geo_dim[2] - cz,
            num=self.n_cells[2],
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

        self.input_dict = input_dict

    def compute_densities(self):
        self.densities = (
            self.problem.density_net(self.input_dict)["densities"]
            .numpy()
            .reshape(self.n_cells)
        )

    def compute_mirror(self):
        if self.problem.mirror[0]:
            self.densities = np.concatenate(
                (
                    self.densities,
                    self.densities[range(self.n_cells[0] - 1, -1, -1), :, :],
                ),
                axis=0,
            )
        if self.problem.mirror[1]:
            self.densities = np.concatenate(
                (
                    self.densities,
                    self.densities[:, range(self.n_cells[1] - 1, -1, -1), :],
                ),
                axis=1,
            )
        if self.problem.mirror[2]:
            self.densities = np.concatenate(
                (
                    self.densities,
                    self.densities[:, :, range(self.n_cells[2] - 1, -1, -1)],
                ),
                axis=2,
            )

    def pad_with_zeros(self):
        self.density_grid = np.pad(
            self.density_grid, ((1, 1), (1, 1), (1, 1)), "constant", constant_values=0.0
        )

    def save_solid(self):
        if self.problem.mirror:
            for i in range(len(self.problem.mirror)):
                self.n_cells[i] *= 2 if self.problem.mirror[i] else 1
        self.density_grid = np.reshape(
            self.densities, (self.n_cells[0], self.n_cells[1], self.n_cells[2])
        )
        self.pad_with_zeros()

        if (
            np.amax(self.density_grid) < self.threshold
            or np.amin(self.density_grid) > self.threshold
        ):
            print("Warning! Cannot save density grid cause the levelset is empty")
            return

        verts, faces, normals, _ = measure.marching_cubes(
            self.density_grid,
            level=self.threshold,
            spacing=[0.005, 0.005, 0.005],
            gradient_direction="ascent",
            method="lewiner",
        )

        with open(self.filename, "w") as file:
            for item in verts:
                file.write(f"v {item[0]} {item[1]} {item[2]}\n")

            for item in normals:
                file.write(f"vn {item[0]} {item[1]} {item[2]}\n")

            for item in faces:
                idx_0 = item[0] + 1
                idx_1 = item[1] + 1
                idx_2 = item[2] + 1
                file.write(f"f {idx_0}//{idx_0} {idx_1}//{idx_1} {idx_2}//{idx_2}\n")

    def plot_3d(self):
        self.prepare_data()
        self.compute_densities()
        if self.problem.mirror:
            self.compute_mirror()
        self.save_solid()


class Sampler:
    def __init__(
        self,
        geom: geometry.Geometry,
        bounds: Tuple[Tuple[float, float], ...],
        criteria: Optional[Callable] = None,
    ) -> None:
        self.geom = geom
        self.dim = geom.ndim
        self.dim_keys = geom.dim_keys
        self.bounds = bounds
        self.criteria = criteria

    def stratified_random(self, n_samples: Tuple[int, ...]) -> np.ndarray:
        """Stratified random."""
        # Divide the geometry into uniformly sized chunks
        # and randomly sample within each chunk.

        zeros = np.zeros(n_samples)
        grid_points = np.transpose(np.where(zeros == 0)).astype(
            paddle.get_default_dtype()
        )
        all_samples = np.prod(n_samples)
        for i in range(self.dim):
            random = np.random.uniform(0.0, 1.0, (all_samples))
            grid_size = (self.bounds[i][1] - self.bounds[i][0]) / n_samples[i]
            grid_points[:, i] = (grid_points[:, i] + random) * grid_size
        return grid_points

    def sample_interior_stratified(self, n_samples: Tuple[int, ...], n_iter: int):
        """Sample random points in the geometry and return those meet criteria."""
        points = self.stratified_random(n_samples)
        for _ in range(1, n_iter):
            points = np.concatenate((points, self.stratified_random(n_samples)), axis=0)

        if self.criteria is not None:
            criteria_mask = self.criteria(*np.split(points, self.dim, axis=1)).flatten()
            points = points[criteria_mask]
        else:
            criteria_mask = self.geom.is_inside(points)
            points = points[criteria_mask]
        points_dict = {}
        for i in range(self.dim):
            points_dict[self.dim_keys[i]] = points[:, i].reshape([-1, 1])

        return points_dict, criteria_mask


class StratifiedInteriorConstraint(base.Constraint):
    def __init__(
        self,
        output_expr: Dict[str, Callable],
        label_dict: Dict[str, Union[float, Callable]],
        geom: geometry.Geometry,
        dataloader_cfg: Dict[str, Any],
        loss: "loss.Loss",
        bounds: Tuple[Tuple[float, float], ...],
        n_samples: Tuple[int, ...],
        criteria: Optional[Callable] = None,
        weight_dict: Optional[Dict[str, Union[Callable, float]]] = None,
        name: str = "EQ",
    ):
        self.output_expr = output_expr
        for label_name, expr in self.output_expr.items():
            if isinstance(expr, str):
                self.output_expr[label_name] = sp_parser.parse_expr(expr)

        self.label_dict = label_dict
        self.input_keys = geom.dim_keys
        self.output_keys = list(label_dict.keys())

        if isinstance(criteria, str):
            criteria = eval(criteria)

        # prepare input
        if np.prod(n_samples) != dataloader_cfg["batch_size"]:
            raise ValueError(
                f"batch size is {dataloader_cfg['batch_size']} which is not equal to samples {np.prod(n_samples)}"
            )
        sampler = Sampler(geom, bounds)
        input, mask = sampler.sample_interior_stratified(
            n_samples, dataloader_cfg["iters_per_epoch"]
        )  # TODO: Use mask to filter samples that do not need to participate in loss calculation

        # prepare label
        label = {}
        for key, value in label_dict.items():
            if isinstance(value, str):
                value = sp_parser.parse_expr(value)
            if isinstance(value, (int, float)):
                label[key] = np.full_like(next(iter(input.values())), value)
            elif isinstance(value, sympy.Basic):
                func = sympy.lambdify(
                    sympy.symbols(geom.dim_keys),
                    value,
                    [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                )
                label[key] = func(
                    **{k: v for k, v in input.items() if k in geom.dim_keys}
                )
            elif callable(value):
                func = value
                label[key] = func(input)
                if isinstance(label[key], (int, float)):
                    label[key] = np.full_like(next(iter(input.values())), label[key])
            else:
                raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        # prepare weight
        weight = {key: np.ones_like(next(iter(label.values()))) for key in label}
        if weight_dict is not None:
            for key, value in weight_dict.items():
                if isinstance(value, str):
                    if value == "sdf":
                        weight[key] = input["sdf"]
                    else:
                        raise NotImplementedError(f"string {value} is invalid yet.")
                elif isinstance(value, (int, float)):
                    weight[key] = np.full_like(next(iter(label.values())), float(value))
                elif isinstance(value, sympy.Basic):
                    func = sympy.lambdify(
                        sympy.symbols(geom.dim_keys),
                        value,
                        [{"amax": lambda xy, _: np.maximum(xy[0], xy[1])}, "numpy"],
                    )
                    weight[key] = func(
                        **{k: v for k, v in input.items() if k in geom.dim_keys}
                    )
                elif callable(value):
                    func = value
                    weight[key] = func(input)
                    if isinstance(weight[key], (int, float)):
                        weight[key] = np.full_like(
                            next(iter(input.values())), weight[key]
                        )
                else:
                    raise NotImplementedError(f"type of {type(value)} is invalid yet.")

        # wrap input, label, weight into a dataset
        if isinstance(dataloader_cfg["dataset"], str):
            dataloader_cfg["dataset"] = {"name": dataloader_cfg["dataset"]}
        dataloader_cfg["dataset"].update(
            {"input": input, "label": label, "weight": weight}
        )
        _dataset = dataset.build_dataset(dataloader_cfg["dataset"])

        # construct dataloader with dataset and dataloader_cfg
        super().__init__(_dataset, dataloader_cfg, loss, name)
