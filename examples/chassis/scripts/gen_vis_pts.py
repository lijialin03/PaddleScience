import numpy as np


def save_to_txt(path, input_x, input_y, input_z, num_list):
    num_x, num_y, num_z = num_list
    title = "Node Number	X Location (m)	Y Location (m)	Z Location (m)"
    with open(path, "w") as f:
        f.write(title)
        f.write("\n")
        for i in range(num_x * num_y * num_z):
            line = (
                str(i + 1)
                + "\t"
                + str(input_x[i // (num_y * num_z)])
                + "\t"
                + str(input_y[i // num_z % num_y])
                + "\t"
                + str(input_z[i % num_z])
                + "\n"
            )
            f.write(line)
    f.close()


def gen_chassis_pts():
    start_3d = (-5, -0.5, -0.5)
    end_3d = (5, 0.5, 0.5)

    num = 10
    input_x = np.linspace(start_3d[0], end_3d[0], num * 10)
    input_y = np.linspace(start_3d[1], end_3d[1], num)
    input_z = np.linspace(start_3d[2], end_3d[2], num)
    input_x = np.around(input_x, 5)
    input_y = np.around(input_y, 5)
    input_z = np.around(input_z, 5)

    # print(np.shape(input))
    # print(input)

    num_list = [num * 10, num, num]
    save_to_txt(
        "../datasets/data/chassis_input.txt", input_x, input_y, input_z, num_list
    )


def gen_sheet_pts():
    start_3d = (-1, -0.5, -5e-4)
    end_3d = (1, 0.5, 5e-4)

    num = 100
    input_x = np.linspace(start_3d[0], end_3d[0], num * 2)
    input_y = np.linspace(start_3d[1], end_3d[1], num)
    input_z = [0.0]

    # print(input_z)

    num_list = [num * 2, num, 1]
    save_to_txt(
        "/home/lijialin03/workspaces/PaddleScience/PaddleScience/examples/chassis/datasets/data/sheet_input.txt",
        input_x,
        input_y,
        input_z,
        num_list,
    )


if __name__ == "__main__":
    # chassis
    # gen_chassis_pts()

    # sheet
    gen_sheet_pts()
