import sys
import libs_python.pyphy as pyphy

#1, load data from dats files and create motion tensor with normalised columns

#load training data
training_dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim_dense_collisions_01/training_dats.json", "experiments/sim_dense_collisions_01/motion_tensor.json")

#load testing data, for normalisation use range from testing
testing_dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim_dense_collisions_01/testing_dats.json", "experiments/sim_dense_collisions_01/motion_tensor.json", training_dats_to_motion_tensor.tensor())



def experiment(tensor_config, network_config):

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




#discretisation_64x64x16
'''
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_1/gaussian/spatial_tensor.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_1/gaussian/net_0/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_1/gaussian/spatial_tensor_depth.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_1/gaussian/net_0_depth/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_1/point/spatial_tensor.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_1/point/net_0/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_1/point/spatial_tensor_depth.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_1/point/net_0_depth/")

experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_4/gaussian/spatial_tensor.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_4/gaussian/net_0/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_4/gaussian/spatial_tensor_depth.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_4/gaussian/net_0_depth/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_4/point/spatial_tensor.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_4/point/net_0/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_4/point/spatial_tensor_depth.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_4/point/net_0_depth/")
'''

'''
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_8/gaussian/spatial_tensor.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_8/gaussian/net_0/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_8/gaussian/spatial_tensor_depth.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_8/gaussian/net_0_depth/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_8/point/spatial_tensor.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_8/point/net_0/")
experiment("experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_8/point/spatial_tensor_depth.json","experiments/sim_dense_collisions_01/discretisation_64x64x16/window_size_8/point/net_0_depth/")
'''

print("program done")
