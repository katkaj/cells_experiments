import sys
import libs_python.pyphy as pyphy



dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim_50/testing_dats.json", "experiments/sim_50/motion_tensor.json")

dats_to_motion_tensor.tensor().save_json("trajectory_result/sim_50/target.json")

prediction_offset = 100

values_modulo = dats_to_motion_tensor.get_values_modulo()


def process_trajectory(tensor_config, network_config, result_file_name):

    print("processing")
    print("tensor config : ", tensor_config)
    print("network config : ", network_config)
    print("result file name : ", result_file_name)

    tensor_interface = pyphy.TensorSpatial(tensor_config, dats_to_motion_tensor.tensor())

    prediction = pyphy.TrajectoryPrediction(dats_to_motion_tensor.tensor())

    prediction.process(network_config, tensor_interface, prediction_offset)
    prediction.get_result().save_json(result_file_name)

experiment_path = "experiments/sim26/discretisation_64x64x3/window_size_8/gaussian/noise_0_0/"

process_trajectory(experiment_path + "spatial_tensor.json", experiment_path + "net_1/trained/cnn_config.json", "trajectory_result/sim_50/noise_gaussian_net_0.json")

print("program done")
