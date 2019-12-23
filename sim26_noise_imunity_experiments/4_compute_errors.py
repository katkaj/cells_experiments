import json
import numpy
import tensor_load

import libs_compute_errors_euklidian_distance

load_start_offset   = 800
load_reshaped       = True

result_path = "trajectory_result/"

#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad(result_path + "target.json", load_start_offset, load_reshaped)
target_tensor.print_info()



print("loading experiment")
experiment_tensor = tensor_load.TensorLoad(result_path + "noise_0_0/" + "net_1_all.json", load_start_offset, load_reshaped)
experiment_tensor.print_info()

errors = libs_compute_errors_euklidian_distance.compute_errors(target_tensor.get(), experiment_tensor.get(), verbose=True)

print(errors)
