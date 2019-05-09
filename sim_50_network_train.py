import sys
import libs_python.pyphy as pyphy

#1, load data from dats files and create motion tensor with normalised columns

#load training data
training_dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim_50/training_dats.json", "experiments/sim_50/motion_tensor.json")

#load testing data, for normalisation use range from testing
testing_dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim_50/testing_dats.json", "experiments/sim_50/motion_tensor.json", training_dats_to_motion_tensor.tensor())



def experiment(tensor_config, network_config):

    print("\n\n\n\n")
    print("startin experiment")
    print(tensor_config)
    print(network_config)

    #2, create network input making class - TensorSpatial
    print("creating training tensor")
    training_tensor = pyphy.TensorSpatial(tensor_config, training_dats_to_motion_tensor.tensor())

    print("creating testing tensor")
    testing_tensor  = pyphy.TensorSpatial(tensor_config, testing_dats_to_motion_tensor.tensor())

    #3, create dataset
    testing_count = 1000
    print("creating dataset")
    dataset = pyphy.DatasetTrajectoryRuntime(training_tensor, training_tensor, testing_count)

    #4, run experiments, train network
    print("training")
    experiment = pyphy.RegressionExperiment(dataset, network_config)
    experiment.run()

    print("training done")
    print("\n\n\n\n")





#discretisation_64x64x16

#net 0
experiment("experiments/sim_50/discretisation_64x64x16/window_size_8/gaussian/spatial_tensor.json","experiments/sim_50/discretisation_64x64x16/window_size_8/gaussian/net_0/")
experiment("experiments/sim_50/discretisation_64x64x16/window_size_8/gaussian/spatial_tensor_depth.json","experiments/sim_50/discretisation_64x64x16/window_size_8/gaussian/net_0_depth/")
experiment("experiments/sim_50/discretisation_64x64x16/window_size_8/point/spatial_tensor.json","experiments/sim_50/discretisation_64x64x16/window_size_8/point/net_0/")
experiment("experiments/sim_50/discretisation_64x64x16/window_size_8/point/spatial_tensor_depth.json","experiments/sim_50/discretisation_64x64x16/window_size_8/point/net_0_depth/")

#net 1
experiment("experiments/sim_50/discretisation_64x64x16/window_size_8/gaussian/spatial_tensor.json","experiments/sim_50/discretisation_64x64x16/window_size_8/gaussian/net_1/")
experiment("experiments/sim_50/discretisation_64x64x16/window_size_8/gaussian/spatial_tensor_depth.json","experiments/sim_50/discretisation_64x64x16/window_size_8/gaussian/net_1_depth/")
experiment("experiments/sim_50/discretisation_64x64x16/window_size_8/point/spatial_tensor.json","experiments/sim_50/discretisation_64x64x16/window_size_8/point/net_1/")
experiment("experiments/sim_50/discretisation_64x64x16/window_size_8/point/spatial_tensor_depth.json","experiments/sim_50/discretisation_64x64x16/window_size_8/point/net_1_depth/")


print("program done")
