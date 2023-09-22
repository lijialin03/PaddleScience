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

from typing import Dict
from typing import Union

import numpy as np

from ppsci.autodiff import hessian
from ppsci.autodiff import jacobian
from ppsci.equation.pde import base


class LinearElasticity_v2(base.PDE):
    r"""Linear elasticity equations.
    Use either (E, nu) or (lambda_, mu) to define the material properties.

    $$
    \begin{cases}
        stress\_disp_{xx} = \lambda(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} + \dfrac{\partial w}{\partial z}) + 2\mu \dfrac{\partial u}{\partial x} - \sigma_{xx} \\
        stress\_disp_{yy} = \lambda(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} + \dfrac{\partial w}{\partial z}) + 2\mu \dfrac{\partial v}{\partial y} - \sigma_{yy} \\
        stress\_disp_{zz} = \lambda(\dfrac{\partial u}{\partial x} + \dfrac{\partial v}{\partial y} + \dfrac{\partial w}{\partial z}) + 2\mu \dfrac{\partial w}{\partial z} - \sigma_{zz} \\
        traction_{x} = n_x \sigma_{xx} + n_y \sigma_{xy} + n_z \sigma_{xz} \\
        traction_{y} = n_y \sigma_{yx} + n_y \sigma_{yy} + n_z \sigma_{yz} \\
        traction_{z} = n_z \sigma_{zx} + n_y \sigma_{zy} + n_z \sigma_{zz} \\
        navier_{x} = \rho(\dfrac{\partial^2 u}{\partial t^2}) - (\lambda + \mu)(\dfrac{\partial^2 u}{\partial x^2}+\dfrac{\partial^2 v}{\partial y \partial x} + \dfrac{\partial^2 w}{\partial z \partial x}) - \mu(\dfrac{\partial^2 u}{\partial x^2} + \dfrac{\partial^2 u}{\partial y^2} + \dfrac{\partial^2 u}{\partial z^2}) \\
        navier_{y} = \rho(\dfrac{\partial^2 v}{\partial t^2}) - (\lambda + \mu)(\dfrac{\partial^2 v}{\partial x \partial y}+\dfrac{\partial^2 v}{\partial y^2} + \dfrac{\partial^2 w}{\partial z \partial y}) - \mu(\dfrac{\partial^2 v}{\partial x^2} + \dfrac{\partial^2 v}{\partial y^2} + \dfrac{\partial^2 v}{\partial z^2}) \\
        navier_{z} = \rho(\dfrac{\partial^2 w}{\partial t^2}) - (\lambda + \mu)(\dfrac{\partial^2 w}{\partial x \partial z}+\dfrac{\partial^2 v}{\partial y \partial z} + \dfrac{\partial^2 w}{\partial z^2}) - \mu(\dfrac{\partial^2 w}{\partial x^2} + \dfrac{\partial^2 w}{\partial y^2} + \dfrac{\partial^2 w}{\partial z^2}) \\
    \end{cases}
    $$

    Args:
        E (Optional[float]): The Young's modulus. Defaults to None.
        nu (Optional[float]): The Poisson's ratio. Defaults to None.
        lambda_ (Optional[float]): Lamé's first parameter. Defaults to None.
        mu (Optional[float]): Lamé's second parameter (shear modulus). Defaults to None.
        rho (float, optional): Mass density. Defaults to 1.
        dim (int, optional): Dimension of the linear elasticity (2 or 3). Defaults to 3.
        time (bool, optional): Whether contains time data. Defaults to False.

    Examples:
        >>> import ppsci
        >>> pde = ppsci.equation.LinearElasticity(
        ...     E=None, nu=None, lambda_=1e4, mu=100, dim=3
        ... )
    """

    def __init__(
        self,
        param_dict: Dict[str, Union[float, np.ndarray]],
        rho: float = 1,
        dim: int = 3,
        time: bool = False,
    ):
        super().__init__()
        self.param_keys = (
            "E",
            "nu",
            "lambda_",
            "mu",
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
            "normal_x",
            "normal_y",
            "normal_z",
        )
        self.rho = rho
        self.dim = dim
        self.time = time
        self.param_dict = param_dict
        for key in self.param_keys:
            if key in self.param_dict:
                setattr(self, key, self.param_dict[key])

        def init_params(out):
            for key in self.param_keys:
                if key not in self.param_dict:
                    setattr(self, key, out[key] if key in out else None)

            if self.u__x is None:
                self.u__x = jacobian(out["u"], out["x"])
            if self.u__y is None:
                self.u__y = jacobian(out["u"], out["y"])
            if self.v__x is None:
                self.v__x = jacobian(out["v"], out["x"])
            if self.v__y is None:
                self.v__y = jacobian(out["v"], out["y"])
            if self.sigma_xx__x is None:
                self.sigma_xx__x = jacobian(out["sigma_xx"], out["x"])
            if self.sigma_xy__x is None:
                self.sigma_xy__x = jacobian(out["sigma_xy"], out["x"])
            if self.sigma_xy__y is None:
                self.sigma_xy__y = jacobian(out["sigma_xy"], out["y"])
            if self.sigma_yy__y is None:
                self.sigma_yy__y = jacobian(out["sigma_yy"], out["y"])

            if self.dim == 3:
                if self.u__z is None:
                    self.u__z = jacobian(out["u"], out["z"])
                if self.v__z is None:
                    self.v__z = jacobian(out["v"], out["z"])
                if self.w__x is None:
                    self.w__x = jacobian(out["w"], out["x"])
                if self.w__y is None:
                    self.w__y = jacobian(out["w"], out["y"])
                if self.w__z is None:
                    self.w__z = jacobian(out["w"], out["z"])
                if self.sigma_xz__x is None:
                    self.sigma_xz__x = jacobian(out["sigma_xz"], out["x"])
                if self.sigma_yz__y is None:
                    self.sigma_yz__y = jacobian(out["sigma_yz"], out["y"])
                if self.sigma_xz__z is None:
                    self.sigma_xz__z = jacobian(out["sigma_xz"], out["z"])
                if self.sigma_yz__z is None:
                    self.sigma_yz__z = jacobian(out["sigma_yz"], out["z"])
                if self.sigma_zz__z is None:
                    self.sigma_zz__z = jacobian(out["sigma_zz"], out["z"])

        # Stress equations
        def stress_disp_xx_compute_func(out):
            init_params(out)

            stress_disp_xx = (
                self.lambda_ * (self.u__x + self.v__y)
                + 2 * self.mu * self.u__x
                - self.sigma_xx
            )
            if self.dim == 3:
                stress_disp_xx += self.lambda_ * self.w__z
            return stress_disp_xx

        self.add_equation("stress_disp_xx", stress_disp_xx_compute_func)

        def stress_disp_yy_compute_func(out):
            init_params(out)
            stress_disp_yy = (
                self.lambda_ * (self.u__x + self.v__y)
                + 2 * self.mu * self.v__y
                - self.sigma_yy
            )
            if self.dim == 3:
                stress_disp_yy += self.lambda_ * self.w__z
            return stress_disp_yy

        self.add_equation("stress_disp_yy", stress_disp_yy_compute_func)

        if self.dim == 3:

            def stress_disp_zz_compute_func(out):
                init_params(out)
                stress_disp_zz = (
                    self.lambda_ * (self.u__x + self.v__y + self.w__z)
                    + 2 * self.mu * self.w__z
                    - self.sigma_zz
                )
                return stress_disp_zz

            self.add_equation("stress_disp_zz", stress_disp_zz_compute_func)

        def stress_disp_xy_compute_func(out):
            init_params(out)
            stress_disp_xy = self.mu * (self.u__y + self.v__x) - self.sigma_xy
            return stress_disp_xy

        self.add_equation("stress_disp_xy", stress_disp_xy_compute_func)

        if self.dim == 3:

            def stress_disp_xz_compute_func(out):
                init_params(out)
                stress_disp_xz = self.mu * (self.u__z + self.w__x) - self.sigma_xz
                return stress_disp_xz

            self.add_equation("stress_disp_xz", stress_disp_xz_compute_func)

            def stress_disp_yz_compute_func(out):
                init_params(out)
                stress_disp_yz = self.mu * (self.v__z + self.w__y) - self.sigma_yz
                return stress_disp_yz

            self.add_equation("stress_disp_yz", stress_disp_yz_compute_func)

        # Equations of equilibrium
        def equilibrium_x_compute_func(out):
            init_params(out)
            equilibrium_x = -self.sigma_xx__x - self.sigma_xy__y
            if self.dim == 3:
                equilibrium_x -= self.sigma_xz__z
            if self.time:
                t, u = out["t"], out["u"]
                equilibrium_x += self.rho * hessian(u, t)
            return equilibrium_x

        self.add_equation("equilibrium_x", equilibrium_x_compute_func)

        def equilibrium_y_compute_func(out):
            init_params(out)
            equilibrium_y = -self.sigma_xy__x - self.sigma_yy__y
            if self.dim == 3:
                equilibrium_y -= self.sigma_yz__z
            if self.time:
                t, v = out["t"], out["v"]
                equilibrium_y += self.rho * hessian(v, t)
            return equilibrium_y

        self.add_equation("equilibrium_y", equilibrium_y_compute_func)

        if self.dim == 3:

            def equilibrium_z_compute_func(out):
                init_params(out)
                equilibrium_z = -self.sigma_xz__x - self.sigma_yz__y - self.sigma_zz__z
                if self.time:
                    t, w = out["t"], out["w"]
                    equilibrium_z += self.rho * hessian(w, t)
                return equilibrium_z

            self.add_equation("equilibrium_z", equilibrium_z_compute_func)

        # Traction equations
        def traction_x_compute_func(out):
            init_params(out)
            traction_x = self.normal_x * self.sigma_xx + self.normal_y * self.sigma_xy
            if self.dim == 3:
                traction_x += self.normal_z * self.sigma_xz
            return traction_x

        self.add_equation("traction_x", traction_x_compute_func)

        def traction_y_compute_func(out):
            init_params(out)
            traction_y = self.normal_x * self.sigma_xy + self.normal_y * self.sigma_yy
            if self.dim == 3:
                traction_y += self.normal_z * self.sigma_yz
            return traction_y

        self.add_equation("traction_y", traction_y_compute_func)

        if self.dim == 3:

            def traction_z_compute_func(out):
                init_params(out)
                traction_z = (
                    self.normal_x * self.sigma_xz
                    + self.normal_y * self.sigma_yz
                    + self.normal_z * self.sigma_zz
                )
                return traction_z

            self.add_equation("traction_z", traction_z_compute_func)
