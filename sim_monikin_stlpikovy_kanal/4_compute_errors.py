import json
import numpy
import tensor_load

import libs_compute_errors_euklidian_distance

load_start_offset   = 80
load_reshaped       = True


#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad( "trajectory_result/target_trajectory.json", load_start_offset, load_reshaped)
target_tensor.print_info()



print("loading experiment")

experiment_tensor = tensor_load.TensorLoad("trajectory_result/" + "noise_0_0_net_6_single.json", load_start_offset, load_reshaped)
libs_compute_errors_euklidian_distance.save_errors(target_tensor.get(), experiment_tensor.get(), "./errors/noise_0_0/net_6_single/")

#experiment_tensor = tensor_load.TensorLoad("trajectory_result/" + "noise_0_0_net_6_all.json", load_start_offset, load_reshaped)
#libs_compute_errors_euklidian_distance.save_errors(target_tensor.get(), experiment_tensor.get(), "./errors/noise_0_0/net_6_all/")


'''
experiment_tensor = tensor_load.TensorLoad("networks/" + "noise_0_0/" + "net_0_all/trajectory_result/trajectory.json", load_start_offset, load_reshaped)
libs_compute_errors_euklidian_distance.save_errors(target_tensor.get(), experiment_tensor.get(), "./errors/noise_0_0/net_0_all/")


experiment_tensor = tensor_load.TensorLoad("networks/" + "noise_0_0/" + "net_1_single/trajectory_result/trajectory.json", load_start_offset, load_reshaped)
libs_compute_errors_euklidian_distance.save_errors(target_tensor.get(), experiment_tensor.get(), "./errors/noise_0_0/net_1_single/")

experiment_tensor = tensor_load.TensorLoad("networks/" + "noise_0_0/" + "net_1_all/trajectory_result/trajectory.json", load_start_offset, load_reshaped)
libs_compute_errors_euklidian_distance.save_errors(target_tensor.get(), experiment_tensor.get(), "./errors/noise_0_0/net_1_all/")


experiment_tensor = tensor_load.TensorLoad("networks/" + "noise_0_0/" + "net_2_single/trajectory_result/trajectory.json", load_start_offset, load_reshaped)
libs_compute_errors_euklidian_distance.save_errors(target_tensor.get(), experiment_tensor.get(), "./errors/noise_0_0/net_2_single/")

experiment_tensor = tensor_load.TensorLoad("networks/" + "noise_0_0/" + "net_2_all/trajectory_result/trajectory.json", load_start_offset, load_reshaped)
libs_compute_errors_euklidian_distance.save_errors(target_tensor.get(), experiment_tensor.get(), "./errors/noise_0_0/net_2_all/")
'''

print("program done")
