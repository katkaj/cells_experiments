import json
import numpy
import tensor_load


class TensorToDats:

    def __init__(self, tensor, file_name_prefix):

        print("processing ", file_name_prefix)
        shape = tensor.shape

        cells_count      = shape[0]
        time_steps_count = shape[1]
        dims_count       = shape[2]

        for k in range(0, cells_count):

            file_name = file_name_prefix + str(k) + ".dat"

            file = open(file_name, "w")

            for j in range(0, time_steps_count):
                result = str(j) + " "
                for i in range(0, dims_count):
                    result+= str(tensor[k][j][i]) + " "
                result+= "\r\n"
                file.write(result)




result_path = "/home/michal/programming/cells_results/"

load_start_offset = 800
load_reshaped = False

#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/target.json", load_start_offset, load_reshaped)

target_tensor.print_info()

convert = TensorToDats(target_tensor.get(), "dat/target/")


'''
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3/net_4.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc8x8x3/net_4/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3/net_5.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc8x8x3/net_5/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3/net_6.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc8x8x3/net_6/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3/net_7.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc8x8x3/net_7/")

input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc8x8x3_no_gaussian/net_4/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc8x8x3_no_gaussian/net_5/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc8x8x3_no_gaussian/net_6/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc8x8x3_no_gaussian/net_7/")
'''



'''
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3/net_4.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc16x16x3/net_4/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3/net_5.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc16x16x3/net_5/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3/net_6.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc16x16x3/net_6/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3/net_7.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc16x16x3/net_7/")

input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc16x16x3_no_gaussian/net_4/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc16x16x3_no_gaussian/net_5/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc16x16x3_no_gaussian/net_6/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc16x16x3_no_gaussian/net_7/")
'''



input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3/net_4.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc40x20x3/net_4/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3/net_5.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc40x20x3/net_5/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3/net_6.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc40x20x3/net_6/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3/net_7.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc40x20x3/net_7/")

input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc40x20x3_no_gaussian/net_4/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc40x20x3_no_gaussian/net_5/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc40x20x3_no_gaussian/net_6/")
input_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
convert = TensorToDats(input_tensor.get(), "dat/disc40x20x3_no_gaussian/net_7/")
