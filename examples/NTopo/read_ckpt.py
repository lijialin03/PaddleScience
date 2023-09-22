import json

from tensorflow.python import pywrap_tensorflow

# ckpt_path = "../2ddata_gan/test_init_params/init_params.ckpt"
ckpt_path = "./init_params.ckpt"

reader = pywrap_tensorflow.NewCheckpointReader(
    ckpt_path
)  # tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
layer_dict = {}
# for key in var_to_shape_map:
# print(key)
# print("tensor_name: ", key)
# 打印key的值
# print(reader.get_tensor(key))
# if key == "discriminator/d_l5/weight":
#     print(reader.get_tensor(key).shape)
# model_name = key.split("/")[0]
# if model_name == "discriminator":
#     print(key)
#     if model_name == "generator":
#         layer_name = key.split("/")[1]
#         if layer_name in layer_dict:
#             layer_dict[layer_name].append(key)
#         else:
#             layer_dict[layer_name] = [key]
# for k in layer_dict:
#     print(k, ":")
#     if k == "g_cA0":
#         v = layer_dict[k]
#         for name in v:
#             print(name.split("/g_cA0/")[-1])
#     # print(layer_dict[k], end="\n")

# print(reader.get_tensor("beta1_power"))
# print(reader.get_tensor("Variable"))

dict = {}
for key in var_to_shape_map:
    if "Adam" not in key:
        dict[key] = reader.get_tensor(key).tolist()

with open("./tf_init_param_noAdam_.json", "w") as f:
    f.write(json.dumps(dict, indent=4))
