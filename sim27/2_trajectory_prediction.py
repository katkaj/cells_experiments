import sys
sys.path.append('..')
import libs_python.pyphy as pyphy

dats_to_motion_tensor = pyphy.DatsToMotionTensor("testing_dats.json", "motion_tensor.json")
dats_to_motion_tensor.tensor().save_json("discretization_experiment/target_trajectory.json")

prediction_offset = 800

def process_trajectory(tensor_config, network_config, result_file_name):

    print("processing")
    print("tensor config : ", tensor_config)
    print("network config : ", network_config)
    print("result file name : ", result_file_name)

    tensor_interface = pyphy.TensorSpatial(tensor_config, dats_to_motion_tensor.tensor())

    prediction = pyphy.TrajectoryPrediction(dats_to_motion_tensor.tensor())

    prediction.process( network_config, tensor_interface, prediction_offset)
    prediction.get_result().save_json(result_file_name)



experiment_path = "discretization_experiment/12x16x16_all/"
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_0/trained/cnn_config.json",  experiment_path + "net_0_trajectory_result.json")
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_1/trained/cnn_config.json",  experiment_path + "net_1_trajectory_result.json")

experiment_path = "discretization_experiment/24x32x32_all/"
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_0/trained/cnn_config.json",  experiment_path + "net_0_trajectory_result.json")
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_1/trained/cnn_config.json",  experiment_path + "net_1_trajectory_result.json")

experiment_path = "discretization_experiment/48x64x64_all/"
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_0/trained/cnn_config.json",  experiment_path + "net_0_trajectory_result.json")
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_1/trained/cnn_config.json",  experiment_path + "net_1_trajectory_result.json")

experiment_path = "discretization_experiment/12x16x16_single/"
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_0/trained/cnn_config.json",  experiment_path + "net_0_trajectory_result.json")
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_1/trained/cnn_config.json",  experiment_path + "net_1_trajectory_result.json")

experiment_path = "discretization_experiment/24x32x32_single/"
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_0/trained/cnn_config.json",  experiment_path + "net_0_trajectory_result.json")
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_1/trained/cnn_config.json",  experiment_path + "net_1_trajectory_result.json")

experiment_path = "discretization_experiment/48x64x64_single/"
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_0/trained/cnn_config.json",  experiment_path + "net_0_trajectory_result.json")
process_trajectory( experiment_path + "spatial_tensor.json", experiment_path + "net_1/trained/cnn_config.json",  experiment_path + "net_1_trajectory_result.json")


print("program done")
