import sys
sys.path.append('..')

import numpy
import tensor_load
import libs_compute_errors


def save_results(file_name_prefix, json_result):

    json_file_name = file_name_prefix + ".json"

    json_out_file = open(json_file_name, "w")
    json.dump(json_result, json_out_file)



#program starts here :

load_start_offset   = 800
load_reshaped       = True
decimation = 100

json_result = {}
json_result["results"] = []



#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad("discretization_experiment/target_trajectory.json", load_start_offset, load_reshaped)
target_tensor.print_info()


experiment_result = "discretization_experiment/12x16x16_all/net_0_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_12x16x16_all_0 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_12x16x16_all_0.print_info()

experiment_result = "discretization_experiment/12x16x16_single/net_0_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_12x16x16_single_0 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_12x16x16_single_0.print_info()

experiment_result = "discretization_experiment/12x16x16_all/net_1_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_12x16x16_all_1 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_12x16x16_all_1.print_info()

experiment_result = "discretization_experiment/12x16x16_single/net_1_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_12x16x16_single_1 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_12x16x16_single_1.print_info()

experiment_result = "discretization_experiment/24x32x32_all/net_0_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_24x32x32_all_0 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_24x32x32_all_0.print_info()

experiment_result = "discretization_experiment/24x32x32_single/net_0_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_24x32x32_single_0 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_24x32x32_single_0.print_info()

experiment_result = "discretization_experiment/24x32x32_all/net_1_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_24x32x32_all_1 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_24x32x32_all_1.print_info()

experiment_result = "discretization_experiment/24x32x32_single/net_1_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_24x32x32_single_1 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_24x32x32_single_1.print_info()

experiment_result = "discretization_experiment/48x64x64_all/net_0_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_48x64x64_all_0 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_48x64x64_all_0.print_info()

experiment_result = "discretization_experiment/48x64x64_single/net_0_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_48x64x64_single_0 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_48x64x64_single_0.print_info()

experiment_result = "discretization_experiment/48x64x64_all/net_1_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_48x64x64_all_1 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_48x64x64_all_1.print_info()

experiment_result = "discretization_experiment/48x64x64_single/net_1_trajectory_result.json"
print("loading prediction result ", experiment_result)
experiment_48x64x64_single_1 = tensor_load.TensorLoad(experiment_result, load_start_offset, load_reshaped)
experiment_48x64x64_single_1.print_info()


print("computing errors")

errors = libs_compute_errors.compute_errors(target_tensor.get(), experiment_12x16x16_all_0.get())
json_result["results"].append(errors)

errors = libs_compute_errors.compute_errors(target_tensor.get(), experiment_12x16x16_all_1.get())
json_result["results"].append(errors)

errors = libs_compute_errors.compute_errors(target_tensor.get(), experiment_24x32x32_all_0.get())
json_result["results"].append(errors)

errors = libs_compute_errors.compute_errors(target_tensor.get(), experiment_24x32x32_all_1.get())
json_result["results"].append(errors)

errors = libs_compute_errors.compute_errors(target_tensor.get(), experiment_48x64x64_all_0.get())
json_result["results"].append(errors)

errors = libs_compute_errors.compute_errors(target_tensor.get(), experiment_48x64x64_all_1.get())
json_result["results"].append(errors)


save_results("error_results", json_result)

print("program done")
