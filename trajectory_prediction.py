import sys
import libs_python.pyphy as pyphy



dats_to_motion_tensor = pyphy.DatsToMotionTensor("parameters/testing_dats.json", "parameters/motion_tensor.json")

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



process_trajectory("parameters/disc8x8x3/spatial_tensor_window_size_4.json", "networks/disc8x8x3_window_size_4/net_4/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/gaussian/net_4.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_window_size_4.json", "networks/disc8x8x3_window_size_4/net_5/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/gaussian/net_5.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_window_size_4.json", "networks/disc8x8x3_window_size_4/net_6/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/gaussian/net_6.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_window_size_4.json", "networks/disc8x8x3_window_size_4/net_7/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/gaussian/net_7.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_depth_window_size_4.json", "networks/disc8x8x3_window_size_4/net_4_depth/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/gaussian/net_4_depth.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_depth_window_size_4.json", "networks/disc8x8x3_window_size_4/net_5_depth/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/gaussian/net_5_depth.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_depth_window_size_4.json", "networks/disc8x8x3_window_size_4/net_6_depth/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/gaussian/net_6_depth.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_depth_window_size_4.json", "networks/disc8x8x3_window_size_4/net_7_depth/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/gaussian/net_7_depth.json")

process_trajectory("parameters/disc16x16x3/spatial_tensor_window_size_4.json", "networks/disc16x16x3_window_size_4/net_4/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_4.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_window_size_4.json", "networks/disc16x16x3_window_size_4/net_5/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_5.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_window_size_4.json", "networks/disc16x16x3_window_size_4/net_6/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_6.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_window_size_4.json", "networks/disc16x16x3_window_size_4/net_7/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_7.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_depth_window_size_4.json", "networks/disc16x16x3_window_size_4/net_4_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_4_depth.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_depth_window_size_4.json", "networks/disc16x16x3_window_size_4/net_5_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_5_depth.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_depth_window_size_4.json", "networks/disc16x16x3_window_size_4/net_6_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_6_depth.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_depth_window_size_4.json", "networks/disc16x16x3_window_size_4/net_7_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_7_depth.json")

process_trajectory("parameters/disc40x20x3/spatial_tensor_window_size_4.json", "networks/disc40x20x3_window_size_4/net_4/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/gaussian/net_4.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_window_size_4.json", "networks/disc40x20x3_window_size_4/net_5/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/gaussian/net_5.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_window_size_4.json", "networks/disc40x20x3_window_size_4/net_6/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/gaussian/net_6.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_window_size_4.json", "networks/disc40x20x3_window_size_4/net_7/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/gaussian/net_7.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_depth_window_size_4.json", "networks/disc40x20x3_window_size_4/net_4_depth/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/gaussian/net_4_depth.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_depth_window_size_4.json", "networks/disc40x20x3_window_size_4/net_5_depth/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/gaussian/net_5_depth.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_depth_window_size_4.json", "networks/disc40x20x3_window_size_4/net_6_depth/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/gaussian/net_6_depth.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_depth_window_size_4.json", "networks/disc40x20x3_window_size_4/net_7_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/gaussian/net_7_depth.json")






process_trajectory("parameters/disc8x8x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc8x8x3_no_gaussian_window_size_4/net_4/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/point/net_4.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc8x8x3_no_gaussian_window_size_4/net_5/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/point/net_5.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc8x8x3_no_gaussian_window_size_4/net_6/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/point/net_6.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc8x8x3_no_gaussian_window_size_4/net_7/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/point/net_7.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc8x8x3_no_gaussian_window_size_4/net_4_depth/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/point/net_4_depth.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc8x8x3_no_gaussian_window_size_4/net_5_depth/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/point/net_5_depth.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc8x8x3_no_gaussian_window_size_4/net_6_depth/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/point/net_6_depth.json")
process_trajectory("parameters/disc8x8x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc8x8x3_no_gaussian_window_size_4/net_7_depth/trained/cnn_config.json", "trajectory_result/discretisation_8x8x3/window_size_4/point/net_7_depth.json")

process_trajectory("parameters/disc16x16x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc16x16x3_no_gaussian_window_size_4/net_4/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_4.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc16x16x3_no_gaussian_window_size_4/net_5/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_5.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc16x16x3_no_gaussian_window_size_4/net_6/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_6.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc16x16x3_no_gaussian_window_size_4/net_7/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_7.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc16x16x3_no_gaussian_window_size_4/net_4_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_4_depth.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc16x16x3_no_gaussian_window_size_4/net_5_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_5_depth.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc16x16x3_no_gaussian_window_size_4/net_6_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_6_depth.json")
process_trajectory("parameters/disc16x16x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc16x16x3_no_gaussian_window_size_4/net_7_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_7_depth.json")

process_trajectory("parameters/disc40x20x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc40x20x3_no_gaussian_window_size_4/net_4/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/point/net_4.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc40x20x3_no_gaussian_window_size_4/net_5/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/point/net_5.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc40x20x3_no_gaussian_window_size_4/net_6/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/point/net_6.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_no_gaussian_window_size_4.json", "networks/disc40x20x3_no_gaussian_window_size_4/net_7/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/point/net_7.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc40x20x3_no_gaussian_window_size_4/net_4_depth/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/point/net_4_depth.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc40x20x3_no_gaussian_window_size_4/net_5_depth/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/point/net_5_depth.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc40x20x3_no_gaussian_window_size_4/net_6_depth/trained/cnn_config.json", "trajectory_result/discretisation_40x20x3/window_size_4/point/net_6_depth.json")
process_trajectory("parameters/disc40x20x3/spatial_tensor_no_gaussian_depth_window_size_4.json", "networks/disc40x20x3_no_gaussian_window_size_4/net_7_depth/trained/cnn_config.json", "trajectory_result/discretisation_16x16x3/window_size_4/point/net_7_depth.json")



print("program done")
