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

experiment_path = "experiments/sim26/discretisation_64x64x16/window_size_8/gaussian/"

#experiments with output noise
experiment(experiment_path + "noise_0_0/spatial_tensor.json", experiment_path + "noise_0_0/net_1/")
experiment(experiment_path + "noise_0_0/spatial_tensor_depth.json", experiment_path + "noise_0_0/net_1_depth/")
experiment(experiment_path + "noise_0_5/spatial_tensor.json", experiment_path + "noise_0_5/net_1/")
experiment(experiment_path + "noise_0_5/spatial_tensor_depth.json", experiment_path + "noise_0_5/net_1_depth/")
experiment(experiment_path + "noise_0_10/spatial_tensor.json", experiment_path + "noise_0_10/net_1/")
experiment(experiment_path + "noise_0_10/spatial_tensor_depth.json", experiment_path + "noise_0_10/net_1_depth/")
experiment(experiment_path + "noise_0_15/spatial_tensor.json", experiment_path + "noise_0_15/net_1/")
experiment(experiment_path + "noise_0_15/spatial_tensor_depth.json", experiment_path + "noise_0_15/net_1_depth/")
experiment(experiment_path + "noise_0_20/spatial_tensor.json", experiment_path + "noise_0_20/net_1/")
experiment(experiment_path + "noise_0_20/spatial_tensor_depth.json", experiment_path + "noise_0_20/net_1_depth/")
experiment(experiment_path + "noise_0_30/spatial_tensor.json", experiment_path + "noise_0_30/net_1/")
experiment(experiment_path + "noise_0_30/spatial_tensor_depth.json", experiment_path + "noise_0_30/net_1_depth/")
experiment(experiment_path + "noise_0_40/spatial_tensor.json", experiment_path + "noise_0_40/net_1/")
experiment(experiment_path + "noise_0_40/spatial_tensor_depth.json", experiment_path + "noise_0_40/net_1_depth/")
experiment(experiment_path + "noise_0_50/spatial_tensor.json", experiment_path + "noise_0_50/net_1/")
experiment(experiment_path + "noise_0_50/spatial_tensor_depth.json", experiment_path + "noise_0_50/net_1_depth/")



#experiments with input noise
experiment(experiment_path + "noise_0_0/spatial_tensor.json", experiment_path + "noise_0_0/net_1/")
experiment(experiment_path + "noise_0_0/spatial_tensor_depth.json", experiment_path + "noise_0_0/net_1_depth/")
experiment(experiment_path + "noise_5_0/spatial_tensor.json", experiment_path + "noise_5_0/net_1/")
experiment(experiment_path + "noise_5_0/spatial_tensor_depth.json", experiment_path + "noise_5_0/net_1_depth/")
experiment(experiment_path + "noise_10_0/spatial_tensor.json", experiment_path + "noise_10_0/net_1/")
experiment(experiment_path + "noise_10_0/spatial_tensor_depth.json", experiment_path + "noise_10_0/net_1_depth/")
experiment(experiment_path + "noise_15_0/spatial_tensor.json", experiment_path + "noise_15_0/net_1/")
experiment(experiment_path + "noise_15_0/spatial_tensor_depth.json", experiment_path + "noise_15_0/net_1_depth/")
experiment(experiment_path + "noise_20_0/spatial_tensor.json", experiment_path + "noise_20_0/net_1/")
experiment(experiment_path + "noise_20_0/spatial_tensor_depth.json", experiment_path + "noise_20_0/net_1_depth/")
experiment(experiment_path + "noise_30_0/spatial_tensor.json", experiment_path + "noise_30_0/net_1/")
experiment(experiment_path + "noise_30_0/spatial_tensor_depth.json", experiment_path + "noise_30_0/net_1_depth/")
experiment(experiment_path + "noise_40_0/spatial_tensor.json", experiment_path + "noise_40_0/net_1/")
experiment(experiment_path + "noise_40_0/spatial_tensor_depth.json", experiment_path + "noise_40_0/net_1_depth/")
experiment(experiment_path + "noise_50_0/spatial_tensor.json", experiment_path + "noise_50_0/net_1/")
experiment(experiment_path + "noise_50_0/spatial_tensor_depth.json", experiment_path + "noise_50_0/net_1_depth/")



print("program done")
