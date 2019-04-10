import sys
import libs_python.pyphy as pyphy
import json
import numpy
import tensor_load

import libs_compute_errors



result_path         = "/home/michal/programming/cells_results/"
prediction_offset = 800

'''
dats_to_motion_tensor = pyphy.DatsToMotionTensor("parameters/testing_dats.json", "parameters/motion_tensor.json")

dats_to_motion_tensor.tensor().save_json("trajectory_result/target.json")




def process_trajectory(tensor_config, network_config, result_file_name):

    print("processing")
    print("tensor config : ", tensor_config)
    print("network config : ", network_config)
    print("result file name : ", result_file_name)

    tensor_interface = pyphy.TensorSpatial(tensor_config, dats_to_motion_tensor.tensor())

    prediction = pyphy.TrajectoryPrediction(dats_to_motion_tensor.tensor())

    prediction.process( network_config, tensor_interface, prediction_offset)
    prediction.get_result().save_json(result_file_name)


#process_trajectory("parameters/disc8x8x3/spatial_tensor.json", "networks/disc8x8x3/net_4/trained/cnn_config.json", "prediction_test_8.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_window_size_4.json", "networks/disc8x8x3_window_size_4/net_4/trained/cnn_config.json", "prediction_test_4.json")
'''


load_start_offset   = 800
load_reshaped       = True

print("loading target")
target_tensor = tensor_load.TensorLoad("/home/michal/programming/cells_results/trajectory_result/target.json", load_start_offset, load_reshaped)
target_tensor.print_info()

#prediction_test_8 = tensor_load.TensorLoad("prediction_test_8.json", load_start_offset, load_reshaped)
#prediction_test_8.print_info()

prediction_test = tensor_load.TensorLoad("trajectory_result/discretisation_8x8x3/window_size_4/point/net_4.json", load_start_offset, load_reshaped)
errors_test = libs_compute_errors.compute_errors(target_tensor.get(), prediction_test.get(), 0)

prediction_test = tensor_load.TensorLoad("trajectory_result/discretisation_8x8x3/window_size_4/point/net_5.json", load_start_offset, load_reshaped)
errors_test = libs_compute_errors.compute_errors(target_tensor.get(), prediction_test.get(), 0)


print("program done")
