import argparse
import csv

import numpy as np
import paddle

import ppsci
from ppsci.autodiff import jacobian
from ppsci.utils import logger
from ppsci.utils import save_load


def parse_args():
    parser = argparse.ArgumentParser("paddlescience running script")
    parser.add_argument("-seed", "--seed", type=int, help="seed")
    parser.add_argument("-n", "--filename", type=str, help="name of file")
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size")
    parser.add_argument("-stl", "--stl", type=str, help="stl file path")
    parser.add_argument("-o", "--out", type=str, help="output file path")
    parser.add_argument(
        "-m", "--model_forward", type=str, help="trained forward-model path"
    )

    args = parser.parse_args()
    return args


def dict2csv(dic, filename):
    file = open(filename, "w", newline="")
    csv_writer = csv.DictWriter(file, fieldnames=list(dic.keys()))
    csv_writer.writeheader()
    for i in range(len(dic[list(dic.keys())[0]])):
        dic1 = {key: float(dic[key][i]) for key in dic.keys()}
        csv_writer.writerow(dic1)
    file.close()


def concate_lists_to_dict(data_list, data_keys):
    data_dict = {}
    for key in data_keys:
        value = np.concatenate([dict[key] for dict in data_list], axis=0)
        data_dict.update({key: value})
    return data_dict


def load_model(path):
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
    save_load.load_pretrain(model=model_forward, path=path)
    return model_forward


def gen_input_data(model_forward, stl_path):
    # set geometry
    geo = ppsci.geometry.Mesh(stl_path)

    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = geo.bounds

    input_sample = geo.sample_interior(
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

    input_dict = {
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

    return input_dict


def gen_mult_type_input_data(model_forward, stl_path):
    # set geometry
    geo = ppsci.geometry.Mesh(stl_path)

    # CIRCLE_LEFT_CENTER_XY = (-4.4, 0)
    # CIRCLE_LEFT_RADIUS = 1.65
    # CIRCLE_RIGHT_CENTER_XZ = (15.8, 0)
    # CIRCLE_RIGHT_RADIUS = 2.21

    # set bounds
    BOUNDS_X, BOUNDS_Y, BOUNDS_Z = geo.bounds
    BOUND_HR_X = 0.0

    # generate samples
    # low resolution
    input_sample_interior_lr = geo.sample_interior(
        int(BATCH_SIZE_TRAIN / 3),
        criteria=lambda x, y, z: (
            (BOUNDS_X[0] < x)
            & (x < BOUND_HR_X)
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )

    # high resolution
    input_sample_interior_hr = geo.sample_interior(
        int(BATCH_SIZE_TRAIN * 2 / 3),
        criteria=lambda x, y, z: (
            (BOUND_HR_X < x)
            & (x < BOUNDS_X[1])
            & (BOUNDS_Y[0] < y)
            & (y < BOUNDS_Y[1])
            & (BOUNDS_Z[0] < z)
            & (z < BOUNDS_Z[1])
        ),
    )

    input_sample = concate_lists_to_dict(
        [input_sample_interior_lr, input_sample_interior_hr],
        tuple(input_sample_interior_lr.keys()),
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

    input_dict = {
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

    return input_dict


if __name__ == "__main__":
    # parameters
    args = parse_args()

    SEED = 2023 if args.seed is None else args.seed
    ppsci.utils.misc.set_random_seed(SEED)

    BATCH_SIZE_TRAIN = 20000 if args.batch_size is None else args.batch_size
    STL_FILE = "./stl/control_arm.stl" if args.stl is None else args.stl
    FILE_NAME = "train" if args.filename is None else args.filename
    OUTPUT_FILE = (
        f"./data_inverse/{FILE_NAME}_{BATCH_SIZE_TRAIN}"
        if args.out is None
        else args.out
    )
    MODEL_FORWARD = (
        "../saved_model/control_arm_3_4/epoch_2000"
        if args.model_forward is None
        else args.model_forward
    )

    # initialize logger
    logger.init_logger("ppsci", "./train.log", "info")

    # load model forward
    model_forward = load_model(MODEL_FORWARD)
    # add input data
    # input_dict = gen_input_data(model_forward, STL_FILE)
    input_dict = gen_mult_type_input_data(model_forward, STL_FILE)

    dict2csv(input_dict, f"{OUTPUT_FILE}_{SEED}.csv")
    logger.info(f"finish generating data in seed {SEED}")
