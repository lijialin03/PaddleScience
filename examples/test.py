# from cmath import pi
# import open3d as o3d
# import numpy as np


# def Rx(rad):
#     R = np.array(
#         [
#             [1, 0, 0],
#             [0, np.cos(rad), -np.sin(rad)],
#             [0, np.sin(rad), np.cos(rad)],
#         ],
#         "float64",
#     )
#     return R


# print("####### test")
# # cylinders
# cylinder_radius = 0.1
# cylinder_height = 2.0

# cylinder_lower_center = (-0.75 + cylinder_radius, 0, 0)
# cylinder_lower = o3d.geometry.TriangleMesh.create_cylinder(
#     cylinder_radius, cylinder_height, resolution=1000, split=4
# )
# cylinder_lower = cylinder_lower.translate(cylinder_lower_center)
# cylinder_lower.compute_triangle_normals()
# cylinder_lower = cylinder_lower.rotate(Rx(np.pi / 2))
# cylinder_lower = cylinder_lower.translate((0, 0, -0.1 - cylinder_radius))
# cylinder_lower = o3d.t.geometry.TriangleMesh.from_legacy(cylinder_lower)

# cylinder_upper_center = (-0.75 + cylinder_radius, 0, 0)
# cylinder_upper = o3d.geometry.TriangleMesh.create_cylinder(
#     cylinder_radius, cylinder_height, resolution=1000, split=4
# )
# cylinder_upper = cylinder_upper.translate(cylinder_upper_center)
# cylinder_upper.compute_triangle_normals()
# cylinder_upper = cylinder_upper.rotate(Rx(np.pi / 2))
# cylinder_upper = cylinder_upper.translate((0, 0, 0.1 + cylinder_radius))
# cylinder_upper = o3d.t.geometry.TriangleMesh.from_legacy(cylinder_upper)

# # cylinder hole
# cylinder_hole_radius = 0.7
# cylinder_hole_height = 0.5
# cylinder_hole_center = (0.125, 0, 0)

# cylinder_hole = o3d.geometry.TriangleMesh.create_cylinder(
#     cylinder_hole_radius, cylinder_hole_height, resolution=1000, split=4
# )
# cylinder_hole.compute_triangle_normals()
# cylinder_hole = o3d.t.geometry.TriangleMesh.from_legacy(cylinder_hole)
# cylinder_hole = cylinder_hole.translate(cylinder_hole_center)

# # box support
# support_origin = (-1, -1, -1)
# support_dim = (0.25, 2, 2)
# support = o3d.geometry.TriangleMesh.create_box(
#     support_dim[0], support_dim[1], support_dim[2]
# )
# support.compute_triangle_normals()
# support.compute_vertex_normals()
# support = o3d.t.geometry.TriangleMesh.from_legacy(support)
# support = support.translate(support_origin)

# # box bracket
# bracket_origin = (-0.75, -1, -0.1)
# bracket_dim = (1.75, 2, 0.2)
# bracket = o3d.geometry.TriangleMesh.create_box(
#     bracket_dim[0], bracket_dim[1], bracket_dim[2]
# )
# bracket.compute_triangle_normals()
# bracket.compute_vertex_normals()
# bracket = o3d.t.geometry.TriangleMesh.from_legacy(bracket)
# bracket = bracket.translate(bracket_origin)

# # box auxs
# aux_lower_origin = (-0.75, -1, -0.1 - cylinder_radius)
# aux_lower_dim = (cylinder_radius, 2, cylinder_radius)
# aux_lower = o3d.geometry.TriangleMesh.create_box(
#     aux_lower_dim[0], aux_lower_dim[1], aux_lower_dim[2]
# )
# aux_lower.compute_triangle_normals()
# aux_lower = o3d.t.geometry.TriangleMesh.from_legacy(aux_lower)
# aux_lower = aux_lower.translate(aux_lower_origin)

# aux_upper_origin = (-0.75, -1, 0.1)
# aux_upper_dim = (cylinder_radius, 2, cylinder_radius)
# aux_upper = o3d.geometry.TriangleMesh.create_box(
#     aux_upper_dim[0], aux_upper_dim[1], aux_upper_dim[2]
# )
# aux_upper.compute_triangle_normals()
# aux_upper = o3d.t.geometry.TriangleMesh.from_legacy(aux_upper)
# aux_upper = aux_upper.translate(aux_upper_origin)

# # combination
# print("combination")
# curve_lower = aux_lower.boolean_difference(cylinder_lower)
# curve_upper = aux_upper.boolean_difference(cylinder_upper)

# geo1 = support.boolean_union(bracket).boolean_difference(cylinder_hole)
# geo2 = geo1.boolean_union(curve_lower).boolean_union(curve_upper)

# geo = geo2
# # geo6 = geo.fill_holes()
# geo.compute_triangle_normals()
# geo.compute_vertex_normals()
# o3d.t.io.write_triangle_mesh("test.stl", geo)


# import paddle
# import numpy as np

# a = 1
# b = 2
# a_t = paddle.to_tensor(a)
# b_t = paddle.to_tensor(b)

# n_list = [a, b]
# t_list = [a_t, b_t]

# # print(paddle.mean(t_list))
# print(np.mean(t_list))

# import paddle
# import ppsci

# res = ppsci.experimental.bessel_i0(paddle.to_tensor([0, 1, 2, 3, 4], dtype="float32"))
# print(res)

import csv

import numpy as np
import paddle

import ppsci
from ppsci.autodiff import jacobian
from ppsci.utils import logger
from ppsci.utils import save_load


def dict2csv(dic, filename):
    file = open(filename, "w", newline="")
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):
        dic1 = {key: float(dic[key][i]) for key in dic.keys()}
        csv_writer.writerow(dic1)
    file.close()


def gen_input_data(input_sample):
    disp_net = ppsci.arch.MLP(
        ("x", "y", "z"), ("u", "v", "w"), 6, 512, "silu", weight_norm=True
    )
    stress_net = ppsci.arch.MLP(
        ("x", "y", "z"),
        ("sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_xz", "sigma_yz"),
        6,
        512,
        "silu",
        weight_norm=True,
    )

    # wrap to a model_list
    model_forward = ppsci.arch.ModelList((disp_net, stress_net))
    save_load.load_pretrain(
        model=model_forward, path="./chassis/saved_model/control_arm_3_4/epoch_2000"
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


# initialize logger
logger.init_logger("ppsci", "./train.log", "info")
# set geometry
control_arm = ppsci.geometry.Mesh("./chassis/datasets/stl/control_arm.stl")
# geometry bool operation
geo = control_arm
geom = {"geo": geo}
# set bounds
BOUNDS_X, BOUNDS_Y, BOUNDS_Z = control_arm.bounds

BATCH_SIZE_TRAIN = 20000

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
label_dict = {
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
}
weight_dict = {
    "sdf": input_sample["sdf"],
}
vis_input_dict = {
    "x": input_sample["x"],
    "y": input_sample["y"],
    "z": input_sample["z"],
}

input_dict.update(label_dict)
input_dict.update(weight_dict)
input_dict.update(vis_input_dict)

dict2csv(input_dict, f"./chassis/datasets/data_inverse/train_{BATCH_SIZE_TRAIN}.csv")
