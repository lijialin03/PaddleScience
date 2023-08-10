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

import os
import os.path as osp
from typing import TYPE_CHECKING
from typing import Tuple

import numpy as np
import paddle
from matplotlib import pyplot as plt

from ppsci.utils import misc

if TYPE_CHECKING:
    from ppsci import solver


def visualize_func(solver: "solver.Solver", epoch_id: int):
    """Visualization program

    Args:
        solver (solver.Solver): Main Solver.
        epoch_id (int): Epoch id.
    """
    for _, _visualizer in solver.visualizer.items():
        all_input = misc.Prettydefaultdict(list)
        all_output = misc.Prettydefaultdict(list)

        input_dict = _visualizer.input_dict
        batch_size = _visualizer.batch_size
        num_samples = len(next(iter(input_dict.values())))
        batch_num = (num_samples + (batch_size - 1)) // batch_size

        for batch_id in range(batch_num):
            batch_input_dict = {}
            st = batch_id * batch_size
            ed = min(num_samples, (batch_id + 1) * batch_size)

            # prepare batch input dict
            for key in input_dict:
                if not paddle.is_tensor(input_dict[key]):
                    batch_input_dict[key] = paddle.to_tensor(
                        input_dict[key][st:ed], paddle.get_default_dtype()
                    )
                else:
                    batch_input_dict[key] = input_dict[key][st:ed]
                batch_input_dict[key].stop_gradient = False

            # forward
            with solver.no_grad_context_manager(solver.eval_with_no_grad):
                batch_output_dict = solver.forward_helper.visu_forward(
                    _visualizer.output_expr, batch_input_dict, solver.model
                )

            # collect batch data
            for key, batch_input in batch_input_dict.items():
                all_input[key].append(
                    batch_input.detach()
                    if solver.world_size == 1
                    else misc.all_gather(batch_input.detach())
                )
            for key, batch_output in batch_output_dict.items():
                all_output[key].append(
                    batch_output.detach()
                    if solver.world_size == 1
                    else misc.all_gather(batch_output.detach())
                )

        # concate all data
        for key in all_input:
            all_input[key] = paddle.concat(all_input[key])
        for key in all_output:
            all_output[key] = paddle.concat(all_output[key])

        # save visualization
        if solver.rank == 0:
            visual_dir = osp.join(solver.output_dir, "visual", f"epoch_{epoch_id}")
            os.makedirs(visual_dir, exist_ok=True)
            _visualizer.save(
                osp.join(visual_dir, _visualizer.prefix), {**all_input, **all_output}
            )


class VisualizeLossFig:
    def __init__(
        self,
        by_epoch: bool,
        loss_keys: Tuple[str, ...],
        output_dir: str = "./output/",
        smooth_step: int = 1,
    ) -> None:
        self.by_epoch = by_epoch
        self.loss_keys = loss_keys
        self.loss_log = []
        self.loss_log_by_epoch = []
        self.output_dir = output_dir
        self.smooth_step = smooth_step

    def update_loss_log(self, loss_dict, is_epoch_loss):
        if self.by_epoch and is_epoch_loss:
            loss_log_list = self.loss_log_by_epoch
        elif not self.by_epoch and not is_epoch_loss:
            loss_log_list = self.loss_log
        else:
            return

        loss_list = []
        for key in loss_dict:
            loss_list.append(loss_dict[key])
        loss_log_list.append(loss_list)

    def plot_loss_fig(self, loss_log_list, figname, smooth_step=1):
        # how many steps of loss are squeezed to one point, num_points is epoch/smooth_step
        plt.figure(300, figsize=(8, 6))
        font = {"weight": "normal", "size": 10}

        loss_log = np.array(loss_log_list)
        if loss_log.shape[0] % smooth_step != 0:
            vis_loss = loss_log[: -(loss_log.shape[0] % smooth_step), :].reshape(
                -1, smooth_step, loss_log.shape[1]
            )
        else:
            vis_loss = loss_log.reshape(-1, smooth_step, loss_log.shape[1])

        for i in range(vis_loss.shape[1]):
            plt.semilogy(np.arange(vis_loss.shape[0]) * smooth_step, vis_loss[:, i])
        plt.legend(
            self.loss_keys,
            loc="lower left",
            prop=font,
        )
        plt.xlabel(figname, fontdict=font)
        plt.ylabel("Loss", fontdict=font)
        plt.grid()
        plt.yticks(size=10)
        plt.xticks(size=10)
        plt.savefig(os.path.join(self.output_dir, f"{figname} Loss.jpg"))

    def vis_epoch_loss(self):
        if self.by_epoch:
            self.plot_loss_fig(self.loss_log_by_epoch, "Epoch")

    def vis_iteration_loss(self):
        if not self.by_epoch:
            self.plot_loss_fig(self.loss_log, "Iteration", self.smooth_step)
