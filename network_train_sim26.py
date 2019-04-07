import sys
import libs_python.pyphy as pyphy

#1, load data from dats files and create motion tensor with normalised columns

#load training data
training_dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim26/training_dats.json", "experiments/sim26/motion_tensor.json")

#load testing data, for normalisation use range from testing
testing_dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim26/testing_dats.json", "experiments/sim26/motion_tensor.json", training_dats_to_motion_tensor.tensor())


def experiment(tensor_config, network_config):

    #2, create network input making class - TensorSpatial
    training_tensor = pyphy.TensorSpatial(tensor_config, training_dats_to_motion_tensor.tensor())
    testing_tensor  = pyphy.TensorSpatial(tensor_config, testing_dats_to_motion_tensor.tensor())


    #3, create dataset
    testing_count = 20000
    dataset = pyphy.DatasetTrajectoryRuntime(training_tensor, training_tensor, testing_count)

    #4, run experiments, train network
    experiment = pyphy.RegressionExperiment(dataset, network_config)
    experiment.run()


#discretisation_8x8x3
experiment("experiments/sim26/discretisation_8x8x3/window_size_1/gaussian/spatial_tensor.json","experiments/sim26/discretisation_8x8x3/window_size_1/gaussian/net_0/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_1/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_8x8x3/window_size_1/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_1/point/spatial_tensor.json","experiments/sim26/discretisation_8x8x3/window_size_1/point/net_0/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_1/point/spatial_tensor_depth.json","experiments/sim26/discretisation_8x8x3/window_size_1/point/net_0_depth/")

experiment("experiments/sim26/discretisation_8x8x3/window_size_4/gaussian/spatial_tensor.json","experiments/sim26/discretisation_8x8x3/window_size_4/gaussian/net_0/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_4/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_8x8x3/window_size_4/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_4/point/spatial_tensor.json","experiments/sim26/discretisation_8x8x3/window_size_4/point/net_0/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_4/point/spatial_tensor_depth.json","experiments/sim26/discretisation_8x8x3/window_size_4/point/net_0_depth/")

experiment("experiments/sim26/discretisation_8x8x3/window_size_8/gaussian/spatial_tensor.json","experiments/sim26/discretisation_8x8x3/window_size_8/gaussian/net_0/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_8/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_8x8x3/window_size_8/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_8/point/spatial_tensor.json","experiments/sim26/discretisation_8x8x3/window_size_8/point/net_0/")
experiment("experiments/sim26/discretisation_8x8x3/window_size_8/point/spatial_tensor_depth.json","experiments/sim26/discretisation_8x8x3/window_size_8/point/net_0_depth/")


#discretisation_16x16x3
experiment("experiments/sim26/discretisation_16x16x3/window_size_1/gaussian/spatial_tensor.json","experiments/sim26/discretisation_16x16x3/window_size_1/gaussian/net_0/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_1/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_16x16x3/window_size_1/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_1/point/spatial_tensor.json","experiments/sim26/discretisation_16x16x3/window_size_1/point/net_0/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_1/point/spatial_tensor_depth.json","experiments/sim26/discretisation_16x16x3/window_size_1/point/net_0_depth/")

experiment("experiments/sim26/discretisation_16x16x3/window_size_4/gaussian/spatial_tensor.json","experiments/sim26/discretisation_16x16x3/window_size_4/gaussian/net_0/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_4/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_16x16x3/window_size_4/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_4/point/spatial_tensor.json","experiments/sim26/discretisation_16x16x3/window_size_4/point/net_0/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_4/point/spatial_tensor_depth.json","experiments/sim26/discretisation_16x16x3/window_size_4/point/net_0_depth/")

experiment("experiments/sim26/discretisation_16x16x3/window_size_8/gaussian/spatial_tensor.json","experiments/sim26/discretisation_16x16x3/window_size_8/gaussian/net_0/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_8/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_16x16x3/window_size_8/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_8/point/spatial_tensor.json","experiments/sim26/discretisation_16x16x3/window_size_8/point/net_0/")
experiment("experiments/sim26/discretisation_16x16x3/window_size_8/point/spatial_tensor_depth.json","experiments/sim26/discretisation_16x16x3/window_size_8/point/net_0_depth/")



#discretisation_40x20x3
experiment("experiments/sim26/discretisation_40x20x3/window_size_1/gaussian/spatial_tensor.json","experiments/sim26/discretisation_40x20x3/window_size_1/gaussian/net_0/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_1/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_40x20x3/window_size_1/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_1/point/spatial_tensor.json","experiments/sim26/discretisation_40x20x3/window_size_1/point/net_0/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_1/point/spatial_tensor_depth.json","experiments/sim26/discretisation_40x20x3/window_size_1/point/net_0_depth/")

experiment("experiments/sim26/discretisation_40x20x3/window_size_4/gaussian/spatial_tensor.json","experiments/sim26/discretisation_40x20x3/window_size_4/gaussian/net_0/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_4/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_40x20x3/window_size_4/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_4/point/spatial_tensor.json","experiments/sim26/discretisation_40x20x3/window_size_4/point/net_0/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_4/point/spatial_tensor_depth.json","experiments/sim26/discretisation_40x20x3/window_size_4/point/net_0_depth/")

experiment("experiments/sim26/discretisation_40x20x3/window_size_8/gaussian/spatial_tensor.json","experiments/sim26/discretisation_40x20x3/window_size_8/gaussian/net_0/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_8/gaussian/spatial_tensor_depth.json","experiments/sim26/discretisation_40x20x3/window_size_8/gaussian/net_0_depth/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_8/point/spatial_tensor.json","experiments/sim26/discretisation_40x20x3/window_size_8/point/net_0/")
experiment("experiments/sim26/discretisation_40x20x3/window_size_8/point/spatial_tensor_depth.json","experiments/sim26/discretisation_40x20x3/window_size_8/point/net_0_depth/")


print("program done")
