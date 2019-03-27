import json
import numpy
import tensor_load

import libs_compute_errors

def save_results(file_name_prefix, json_result):

    json_file_name = file_name_prefix + ".json"

    json_out_file = open(json_file_name, "w")
    json.dump(json_result, json_out_file)


#program starts here :

load_start_offset = 800
load_reshaped = True

result_path = "/home/michal/programming/cells_results/"

#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/target.json", load_start_offset, load_reshaped)

target_tensor.print_info()




print("loading experiments")


experiment_disc8x8x3_no_gaussian_net_4_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_4_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_4_depth.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_5_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_5_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_5_depth.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_6_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_6_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_6_depth.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_7_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_7_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc8x8x3_no_gaussian/net_7_depth.json", load_start_offset, load_reshaped)

experiment_disc16x16x3_no_gaussian_net_4_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_4_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_4_depth.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_5_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_5_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_5_depth.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_6_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_6_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_6_depth.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_7_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_7_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc16x16x3_no_gaussian/net_7_depth.json", load_start_offset, load_reshaped)


experiment_disc40x20x3_no_gaussian_net_4_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_4_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_4_depth.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_5_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_5_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_5_depth.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_6_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_6_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_6_depth.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_7_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_7_depth_tensor = tensor_load.TensorLoad(result_path + "trajectory_result/disc40x20x3_no_gaussian/net_7_depth.json", load_start_offset, load_reshaped)


print()
print("computing errors")
print()

json_result = {}
json_result["results"] = []


json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_4_tensor.get(), "disc8x8x3_no_gaussian_net_4"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_4_depth_tensor.get(), "disc8x8x3_no_gaussian_net_4_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_5_tensor.get(), "disc8x8x3_no_gaussian_net_5"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_5_depth_tensor.get(), "disc8x8x3_no_gaussian_net_5_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_6_tensor.get(), "disc8x8x3_no_gaussian_net_6"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_6_depth_tensor.get(), "disc8x8x3_no_gaussian_net_6_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_7_tensor.get(), "disc8x8x3_no_gaussian_net_7"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_7_depth_tensor.get(), "disc8x8x3_no_gaussian_net_7_depth"))

json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_4_tensor.get(), "disc16x16x3_no_gaussian_net_4"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_4_depth_tensor.get(), "disc16x16x3_no_gaussian_net_4_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_5_tensor.get(), "disc16x16x3_no_gaussian_net_5"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_5_depth_tensor.get(), "disc16x16x3_no_gaussian_net_5_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_6_tensor.get(), "disc16x16x3_no_gaussian_net_6"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_6_depth_tensor.get(), "disc16x16x3_no_gaussian_net_6_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_7_tensor.get(), "disc16x16x3_no_gaussian_net_7"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_7_depth_tensor.get(), "disc16x16x3_no_gaussian_net_7_depth"))


json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_4_tensor.get(), "disc40x20x3_no_gaussian_net_4"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_4_depth_tensor.get(), "disc40x20x3_no_gaussian_net_4_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_5_tensor.get(), "disc40x20x3_no_gaussian_net_5"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_5_depth_tensor.get(), "disc40x20x3_no_gaussian_net_5_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_6_tensor.get(), "disc40x20x3_no_gaussian_net_6"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_6_depth_tensor.get(), "disc40x20x3_no_gaussian_net_6_depth"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_7_tensor.get(), "disc40x20x3_no_gaussian_net_7"))
json_result["results"].append(libs_compute_errors.compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_7_depth_tensor.get(), "disc40x20x3_no_gaussian_net_7_depth"))



print("saving results")
save_results("trajectory_result/errors_no_gaussian", json_result)


print("program done")
