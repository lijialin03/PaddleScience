import paddle
import tensorflow.compat.v1 as tf1


def load_tf_params(checkpoint_path):
    tf1.disable_v2_behavior()
    # Read data from checkpoint file
    reader = tf1.train.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    tf_dict = {}
    for key in var_to_shape_map:
        if "model" not in key:
            continue
        # print("tensor_name: ", key)
        value = reader.get_tensor(key)
        # print(value.shape)
        # print(type(value))
        # print(value)
        new_key = key.split("/.")[0]
        # print(new_key)
        tf_dict[new_key] = value
    return tf_dict


def load_paddle_params(path):
    param_dict = paddle.load(path + ".pdparams")
    for key in param_dict:
        print(key)
        # value = param_dict[key]
        # print(value.shape)
        # print(type(value))
        # print(value)
    #     exit()
    return param_dict


def rewrite_paddle_params_model_list(tf_dict, param_dict, model_idx):
    for key in param_dict:
        _, m_idx, _, n_layer, _, type_param = key.split(
            "."
        )  # model_list.0.linears.0.linear.weight
        if m_idx != str(model_idx):
            continue

        tf_key_name = (
            "model/dense" + n_layer + "/kernel"
            if type_param == "weight"
            else "model/dense" + n_layer + "/bias"
        )  # model/dense0/kernel

        # print(key, tf_key_name)
        if tf_key_name not in tf_dict.keys():
            print("Wrong name, please check it")
        else:
            if n_layer == "0":
                print(key)
                print(tf_dict[tf_key_name].mean())
                print(param_dict[key].mean())
            param_dict[key] = paddle.to_tensor(tf_dict[tf_key_name])

    return param_dict


def rewrite_paddle_params(tf_dict, param_dict):
    for key in param_dict:
        _, n_layer, _, type_param = key.split(".")  # linears.0.linear.weight

        tf_key_name = (
            "model/dense" + n_layer + "/kernel"
            if type_param == "weight"
            else "model/dense" + n_layer + "/bias"
        )  # model/dense0/kernel

        # print(key, tf_key_name)
        if tf_key_name not in tf_dict.keys():
            print("Wrong name, please check it")
        else:
            if n_layer == "0":
                print(key)
                print(tf_dict[tf_key_name].mean())
                print(param_dict[key].mean())
            param_dict[key] = paddle.to_tensor(tf_dict[tf_key_name])

    return param_dict


def remove_model_list_info(param_dict):
    param_dict_new = {}
    for key in param_dict:
        _, m_idx, linears, n_layer, linear, type_param = key.split(".")
        key_new = linears + "." + n_layer + "." + linear + "." + type_param
        print(key_new)
        param_dict_new[key_new] = param_dict[key]
    return param_dict_new


if __name__ == "__main__":
    checkpoint_path = "./init_params_3d/disp_model-000000"
    tf_dict_disp = load_tf_params(checkpoint_path)
    checkpoint_path = "./init_params_3d/density_model-000000"
    tf_dict_density = load_tf_params(checkpoint_path)
    # print(tf_dict)

    path = "./init_params_3d/paddle_init_disp"
    # path = "./init_params_3d/paddle_init_density"
    # path = "./init_params/paddle_init"
    # path = "./output_ntopo_test/checkpoints/latest"
    param_dict = load_paddle_params(path)

    param_dict = rewrite_paddle_params(tf_dict_disp, param_dict)

    # param_dict=remove_model_list_info(param_dict)

    paddle.save(param_dict, "./init_params_3d/paddle_init_only_disp.pdparams")
