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

from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.utils import initializer


class BaseBlock(nn.Layer):
    def __init__(
        self,
        layer: nn.Layer,
        bn: nn.Layer = None,
        w0: float = 1.0,
        coef1: float = 6.0,
        coef2: float = 1.0,
        weight_init=True,
        bias_init=True,
        use_act=False,
        use_sqrt=False,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.bn = bn if bn else None
        self.act = act_mod.Siren(w0) if use_act else None

        if weight_init:
            self.init_param(self.layer.weight, coef1, coef2, use_sqrt)
        else:
            self.init_zeros(self.layer.weight)
        if bias_init:
            self.init_param(self.layer.bias, coef1, coef2)
        else:
            self.init_zeros(self.layer.bias)

    def init_param(self, param, coef1: float = 6.0, coef2: float = 1.0, use_sqrt=True):
        in_features = param.shape[0]
        with paddle.no_grad():
            initializer.uniform_(
                param,
                -np.sqrt(coef1 / in_features) * coef2,
                np.sqrt(coef1 / in_features) * coef2,
            ) if use_sqrt else initializer.uniform_(
                param,
                -coef1 / in_features * coef2,
                coef1 / in_features * coef2,
            )

    def init_zeros(self, param):
        initializer.zeros_(param)

    def forward(self, x):
        y = x
        y = self.layer(y)
        if self.bn:
            y = self.bn(y)
        if self.act:
            y = self.act(y)
        return y


class LinearBlock(BaseBlock):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 1.0,
        coef1: float = 6.0,
        coef2: float = 1.0,
        weight_init=True,
        bias_init=True,
        use_act=False,
        use_sqrt=False,
    ) -> None:
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            linear, None, w0, coef1, coef2, weight_init, bias_init, use_act, use_sqrt
        )


class ConvBlock(BaseBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...], List[int]],
        stride: int = 1,
        w0: float = 1.0,
        coef1: float = 6.0,
        coef2: float = 1.0,
        weight_init=True,
        bias_init=True,
        use_bn=True,
        use_act=False,
        use_sqrt=False,
    ) -> None:
        conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding="SAME",
            data_format="NHWC",
        )
        bn = nn.BatchNorm(out_channels, data_layout="NHWC") if use_bn else None
        super().__init__(
            conv, bn, w0, coef1, coef2, weight_init, bias_init, use_act, use_sqrt
        )


class DenseSIRENModel(base.Arch):
    """DenseSIRENModel network."""

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: int,
        last_layer_init_scale: float,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.linears = []
        self.skip = (False, False, True, False, True, False)
        in_features = (
            len(input_keys),
            hidden_size,
            hidden_size,
            len(input_keys) + hidden_size,
            hidden_size,
            len(input_keys) + 2 * hidden_size,
        )
        out_features = (
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
            hidden_size,
            len(output_keys),
        )
        w0 = (60.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        coef1 = (1.0, 6.0, 6.0, 6.0, 6.0, 6.0)
        coef2 = (1.0, 1.0, 1.0, 1.0, 1.0, last_layer_init_scale)
        weight_init = (True, True, True, True, True, True)
        bias_init = (True, True, True, True, True, False)
        use_act = (True, True, True, True, True, False)
        use_sqrt = (False, True, True, True, True, True)

        # initialize layers
        for i in range(num_layers):
            self.linears.append(
                LinearBlock(
                    in_features[i],
                    out_features[i],
                    w0[i],
                    coef1[i],
                    coef2[i],
                    weight_init[i],
                    bias_init[i],
                    use_act[i],
                    use_sqrt[i],
                )
            )
        self.linears = nn.LayerList(self.linears)

    def forward_tensor(self, x):
        y = x
        short = y
        for i, layer in enumerate(self.linears):
            y = layer(y)
            if self.skip[i]:
                y = paddle.concat([short, y], axis=-1)
                short = y
        return y

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y


class ConvSIRENModel(base.Arch):
    """ResSIRENModel network."""

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: int,
        last_layer_init_scale: float,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.convs = []
        # self.skip = (True, True, True, True, False, False)

        kernel_size = (1, 1)
        stride = (1, 1, 1, 1, 1, 1)

        w0 = (60.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        coef1 = (1.0, 6.0, 6.0, 6.0, 6.0, 6.0)
        coef2 = (1.0, 1.0, 1.0, 1.0, 1.0, last_layer_init_scale)
        weight_init = (True, True, True, True, True, True)
        bias_init = (True, True, True, True, True, False)
        # use_bn = (True, True, True, True, True, True)
        use_bn = (False, False, False, False, False, False)
        use_act = (True, True, True, True, True, False)
        use_sqrt = (False, True, True, True, True, True)
        # in_channels = (len(self.input_keys), 8, 16, 32, 16, 8)
        # out_channels = (8, 16, 32, 16, 8, len(self.output_keys))
        # in_channels = (len(self.input_keys), 16, 32, 64, 32, 16)
        # out_channels = (16, 32, 64, 32, 16, len(self.output_keys))
        # in_channels = (len(self.input_keys), 8, 16, 32, 16, 8)  # 0->3 concat
        # out_channels = (8, 16, 32, 8, 8, len(self.output_keys))
        in_channels = (len(self.input_keys), 8, 24, 32, 16, 24)  # densenet
        out_channels = (8, 16, 32, 16, 8, len(self.output_keys))
        num_layers = len(in_channels)

        # initialize layers
        # self.convs.append(nn.Upsample(scale_factor=2, data_format="NHWC"))
        # self.convs.append(nn.Upsample(scale_factor=2, data_format="NHWC"))

        for i in range(num_layers):
            self.convs.append(
                ConvBlock(
                    in_channels[i],
                    out_channels[i],
                    kernel_size,
                    stride[i],
                    w0[i],
                    coef1[i],
                    coef2[i],
                    weight_init[i],
                    bias_init[i],
                    use_bn[i],
                    use_act[i],
                    use_sqrt[i],
                )
            )
        self.convs = nn.LayerList(self.convs)

    def forward_tensor(self, x):
        y = x
        short = y
        for i, layer in enumerate(self.convs):
            y = layer(y)

            # # orignal net
            # if self.skip[i]:
            #     y = paddle.concat([short, y], axis=-1)
            #     short = y

            # conv + dense
            if i == 0 or i == 3:
                short = y
            if i == 1 or i == 4:
                y = paddle.concat([short, y], axis=-1)
                # short = y
        return y

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y


class PWConvSIRENModel(base.Arch):
    """ResSIRENModel network."""

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_layers: int,
        hidden_size: int,
        last_layer_init_scale: float,
    ):
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.convs = []
        # self.skip = (True, True, True, True, False, False)

        kernel_size = [[1, 1]] * 4 + [[2, 2]] * 2
        # print(kernel_size)
        stride = (1, 1, 1, 1, 2, 2)

        w0 = (60.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        coef1 = (1.0, 6.0, 6.0, 6.0, 6.0, 6.0)
        coef2 = (1.0, 1.0, 1.0, 1.0, 1.0, last_layer_init_scale)
        weight_init = (True, True, True, True, True, True)
        bias_init = (True, True, True, True, True, False)
        # use_bn = (True, True, True, True, True, True)
        use_bn = (False, False, False, False, False, False)
        use_act = (True, True, True, True, True, False)
        use_sqrt = (False, True, True, True, True, True)
        in_channels = (len(self.input_keys), 8, 16, 32, 16, 8)  # densenet
        out_channels = (8, 16, 32, 16, 8, len(self.output_keys))
        num_layers = len(in_channels)

        # initialize layers
        self.convs.append(nn.Upsample(scale_factor=2, data_format="NHWC"))
        self.convs.append(nn.Upsample(scale_factor=2, data_format="NHWC"))

        for i in range(num_layers):
            self.convs.append(
                ConvBlock(
                    in_channels[i],
                    out_channels[i],
                    kernel_size[i],
                    stride[i],
                    w0[i],
                    coef1[i],
                    coef2[i],
                    weight_init[i],
                    bias_init[i],
                    use_bn[i],
                    use_act[i],
                    use_sqrt[i],
                )
            )
        self.convs = nn.LayerList(self.convs)

    def forward_tensor(self, x):
        y = x
        # short = y
        for i, layer in enumerate(self.convs):
            # print(y.shape)
            y = layer(y)

            # # orignal net
            # if self.skip[i]:
            #     y = paddle.concat([short, y], axis=-1)
            #     short = y

            # # conv + dense
            # if i == 0 or i == 3:
            #     short = y
            # if i == 1 or i == 4:
            #     y = paddle.concat([short, y], axis=-1)
            #     # short = y
        return y

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        y = self.concat_to_tensor(x, self.input_keys, axis=-1)
        y = self.forward_tensor(y)
        y = self.split_to_dict(y, self.output_keys, axis=-1)

        if self._output_transform is not None:
            y = self._output_transform(x, y)
        return y
