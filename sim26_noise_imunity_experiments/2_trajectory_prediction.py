import sys
sys.path.append('..')
import libs_python.pyphy as pyphy



dats_to_motion_tensor = pyphy.DatsToMotionTensor("testing_dats.json", "motion_tensor.json")
dats_to_motion_tensor.tensor().save_json("trajectory_result/target.json")

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




def experiment(path):
    process_trajectory(path + "spatial_tensor_single.json", path + "net_0_single/trained/network_config.json", path + "net_0_single/trajectory_result/" + "trajectory.json")
    process_trajectory(path + "spatial_tensor_single.json", path + "net_1_single/trained/network_config.json", path + "net_1_single/trajectory_result/" + "trajectory.json")
    process_trajectory(path + "spatial_tensor_single.json", path + "net_2_single/trained/network_config.json", path + "net_2_single/trajectory_result/" + "trajectory.json")

    process_trajectory(path + "spatial_tensor_all.json", path + "net_0_all/trained/network_config.json", path + "net_0_all/trajectory_result/" + "trajectory.json")
    process_trajectory(path + "spatial_tensor_all.json", path + "net_1_all/trained/network_config.json", path + "net_1_all/trajectory_result/" + "trajectory.json")
    process_trajectory(path + "spatial_tensor_all.json", path + "net_2_all/trained/network_config.json", path + "net_2_all/trajectory_result/" + "trajectory.json")

    



experiment("networks/noise_0_0/")
experiment("networks/noise_5_0/")
experiment("networks/noise_10_0/")
experiment("networks/noise_15_0/")
experiment("networks/noise_20_0/")
experiment("networks/noise_25_0/")
experiment("networks/noise_30_0/")


print("program done")
