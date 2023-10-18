import os.path as osp

import model as model_module
import numpy as np
import paddle
import paddle.nn.functional as F
from skimage import measure

import ppsci
from ppsci.utils import config
from ppsci.utils import logger
from ppsci.utils import save_load


def pad_with_zeros(density_grid):
    c0 = np.full(
        (1, density_grid.shape[1], density_grid.shape[2]), 0.0, dtype=np.float32
    )
    density_grid = np.concatenate((c0, density_grid, c0), axis=0)
    c1 = np.full(
        (density_grid.shape[0], 1, density_grid.shape[2]), 0.0, dtype=np.float32
    )
    density_grid = np.concatenate((c1, density_grid, c1), axis=1)
    c2 = np.full(
        (density_grid.shape[0], density_grid.shape[1], 1), 0.0, dtype=np.float32
    )
    density_grid = np.concatenate((c2, density_grid, c2), axis=2)
    return density_grid


def save_solid(filename, densities, n_cells, threshold=0.25):
    nx, ny, nz = n_cells
    density_grid = np.reshape(densities, (nx, ny, nz * 2))
    density_grid = pad_with_zeros(density_grid)

    if np.amax(density_grid) < threshold or np.amin(density_grid) > threshold:
        print("cannot save density grid cause the levelset is empty")
        return

    verts, faces, normals, values = measure.marching_cubes(
        density_grid,
        level=threshold,
        spacing=[0.005, 0.005, 0.005],
        gradient_direction="ascent",
        method="lewiner",
    )

    with open(filename, "w") as file:
        for item in verts:
            file.write(f"v {item[0]} {item[1]} {item[2]}\n")

        for item in normals:
            file.write(f"vn {item[0]} {item[1]} {item[2]}\n")

        for item in faces:
            idx_0 = item[0] + 1
            idx_1 = item[1] + 1
            idx_2 = item[2] + 1
            file.write(f"f {idx_0}//{idx_0} {idx_1}//{idx_1} {idx_2}//{idx_2}\n")


if __name__ == "__main__":
    args = config.parse_args()
    SEED = 2023
    ppsci.utils.misc.set_random_seed(SEED)
    OUTPUT_DIR_VIS = "./eval/"

    # initialize logger
    logger.init_logger("ppsci", f"{OUTPUT_DIR_VIS}/train.log", "info")

    # set geometry
    BEAM_ORIGIN = (0.0, 0.0, 0.0)
    BEAM_DIM = (1.0, 0.5, 0.25)
    beam = ppsci.geometry.Cuboid(BEAM_ORIGIN, BEAM_DIM)
    # geometry bool operation
    geo = beam
    geom = {"geo": geo}

    def comput_volume(dim):
        return dim[0] * dim[1] * dim[2]

    volume = comput_volume(BEAM_DIM)
    volume_ratio = 0.5

    # set model
    input_keys = (
        "x_scaled",
        "y_scaled",
        "z_scaled",
        "sin_x_scaled",
        "sin_y_scaled",
        "sin_z_scaled",
    )
    disp_net = model_module.DenseSIRENModel(input_keys, ("u", "v", "w"), 6, 180, 0.001)
    density_net = model_module.DenseSIRENModel(input_keys, ("density",), 6, 180, 0.001)

    # input transform
    def transform_in(_in):
        x, y, z = _in["x"], _in["y"], _in["z"]
        x_scaled = 2.0 / BEAM_DIM[0] * x + (-1.0 - 2.0 * BEAM_ORIGIN[0] / BEAM_DIM[0])
        y_scaled = 2.0 / BEAM_DIM[1] * y + (-1.0 - 2.0 * BEAM_ORIGIN[1] / BEAM_DIM[1])
        z_scaled = 2.0 / BEAM_DIM[2] * z + (-1.0 - 2.0 * BEAM_ORIGIN[2] / BEAM_DIM[2])

        sin_x_scaled, sin_y_scaled, sin_z_scaled = (
            paddle.sin(x_scaled),
            paddle.sin(y_scaled),
            paddle.sin(z_scaled),
        )
        return {
            "x_scaled": x_scaled,
            "y_scaled": y_scaled,
            "z_scaled": z_scaled,
            "sin_x_scaled": sin_x_scaled,
            "sin_y_scaled": sin_y_scaled,
            "sin_z_scaled": sin_z_scaled,
        }

    def transform_out_disp(_in, _out):
        x_scaled, z_scaled = _in["x_scaled"], _in["z_scaled"]
        x = BEAM_DIM[0] / 2 * (1 + x_scaled) + BEAM_ORIGIN[0]
        z = BEAM_DIM[2] / 2 * (1 + z_scaled) + BEAM_ORIGIN[2]
        u, v, w = x * _out["u"], x * _out["v"], x * (z - BEAM_DIM[2]) * _out["w"]
        return {"u": u, "v": v, "w": w}

    def transform_out_density(_in, _out):
        density = _out["density"]
        alpha = 5.0
        offset = np.log(volume_ratio / (1.0 - volume_ratio))
        densities = F.sigmoid(alpha * density + offset)
        return {"densities": densities}

    disp_net.register_input_transform(transform_in)
    disp_net.register_output_transform(transform_out_disp)

    density_net.register_input_transform(transform_in)
    density_net.register_output_transform(transform_out_density)

    # load pretrained model
    save_load.load_pretrain(density_net, "./output_ntopo_3d_test/checkpoints/epoch_100")

    # add inferencer data
    n_cells = [100, 50, 25]
    cx = 0.5 * BEAM_DIM[0] / n_cells[0]
    cy = 0.5 * BEAM_DIM[1] / n_cells[1]
    cz = 0.5 * BEAM_DIM[2] / n_cells[2]
    x = np.linspace(
        BEAM_ORIGIN[0] + cx,
        BEAM_DIM[0] - cx,
        num=n_cells[0],
        dtype=paddle.get_default_dtype(),
    )
    y = np.linspace(
        BEAM_ORIGIN[1] + cy,
        BEAM_DIM[1] - cy,
        num=n_cells[1],
        dtype=paddle.get_default_dtype(),
    )
    z = np.linspace(
        BEAM_ORIGIN[2] + cz,
        BEAM_DIM[2] - cz,
        num=n_cells[2],
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

    densities = density_net(input_dict)["densities"].numpy().reshape(n_cells)

    # compute_mirrored_densities
    densities = np.concatenate(
        (densities, densities[:, :, range(n_cells[2] - 1, -1, -1)]), axis=2
    )

    save_solid(
        osp.join(OUTPUT_DIR_VIS, "density.obj"), densities, n_cells, threshold=0.99
    )
