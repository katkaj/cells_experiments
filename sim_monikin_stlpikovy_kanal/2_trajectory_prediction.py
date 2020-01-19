import sys
sys.path.append('..')
import libs_python.pyphy as pyphy



dats_to_motion_tensor = pyphy.DatsToMotionTensor("testing_dats.json", "motion_tensor.json")
dats_to_motion_tensor.tensor()._print()


#dats_to_motion_tensor.tensor().save_json("trajectory_result/target_trajectory.json")

prediction_offset = 80

def process_trajectory(tensor_config, network_config, result_file_name):

    print("processing")
    print("tensor config : ", tensor_config)
    print("network config : ", network_config)
    print("result file name : ", result_file_name)

    tensor_interface = pyphy.TensorSpatial(tensor_config, dats_to_motion_tensor.tensor())

    prediction = pyphy.TrajectoryPrediction(dats_to_motion_tensor.tensor())

    prediction.process( network_config, tensor_interface, prediction_offset)
    prediction.get_result().save_json(result_file_name)


process_trajectory("networks/disc32x16x10/noise_0_0/spatial_tensor_single.json", "networks/disc32x16x10/noise_0_0/net_6_single/trained/network_config.json", "trajectory_result/noise_0_0_net_6_single.json")
#process_trajectory("networks/disc32x16x10/noise_0_0/spatial_tensor_all.json", "networks/disc32x16x10/noise_0_0/net_6_all/trained/network_config.json", "trajectory_result/noise_0_0_net_6_all.json")

#process_trajectory("networks/disc32x16x10/noise_0_5/spatial_tensor_single.json", "networks/disc32x16x10/noise_0_5/net_6_single/trained/network_config.json", "trajectory_result/noise_0_5_net_6_single.json")
#process_trajectory("networks/disc32x16x10/noise_0_5/spatial_tensor_all.json", "networks/disc32x16x10/noise_0_5/net_6_all/trained/network_config.json", "trajectory_result/noise_0_5_net_6_all.json")

#process_trajectory("networks/disc32x16x10/noise_0_10/spatial_tensor_single.json", "networks/disc32x16x10/noise_0_10/net_6_single/trained/network_config.json", "trajectory_result/noise_0_10_net_6_single.json")
#process_trajectory("networks/disc32x16x10/noise_0_10/spatial_tensor_all.json", "networks/disc32x16x10/noise_0_10/net_6_all/trained/network_config.json", "trajectory_result/noise_0_10_net_6_all.json")



'''
process_trajectory("networks/disc32x16x10/noise_5_0/spatial_tensor_single.json", "networks/disc32x16x10/noise_5_0/net_6_single/trained/network_config.json", "trajectory_result/noise_5_0_net_6_single.json")
process_trajectory("networks/disc32x16x10/noise_5_0/spatial_tensor_all.json", "networks/disc32x16x10/noise_5_0/net_6_all/trained/network_config.json", "trajectory_result/noise_5_0_net_6_all.json")

process_trajectory("networks/disc32x16x10/noise_10_0/spatial_tensor_single.json", "networks/disc32x16x10/noise_10_0/net_6_single/trained/network_config.json", "trajectory_result/noise_10_0_net_6_single.json")
process_trajectory("networks/disc32x16x10/noise_10_0/spatial_tensor_all.json", "networks/disc32x16x10/noise_10_0/net_6_all/trained/network_config.json", "trajectory_result/noise_10_0_net_6_all.json")
'''

print("program done")
