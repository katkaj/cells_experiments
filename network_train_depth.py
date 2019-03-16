import sys
import libs_python.pyphy as pyphy

#1, load data from dats files and create motion tensor with normalised columns

#load training data
training_dats_to_motion_tensor = pyphy.DatsToMotionTensor("parameters/training_dats.json", "parameters/motion_tensor.json")

#load testing data, for normalisation use range from testing
testing_dats_to_motion_tensor = pyphy.DatsToMotionTensor("parameters/testing_dats.json", "parameters/motion_tensor.json", training_dats_to_motion_tensor.tensor())


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


experiment("parameters/disc16x16x3/spatial_tensor.json","networks/disc16x16x3/net_4/")
experiment("parameters/disc16x16x3/spatial_tensor.json","networks/disc16x16x3/net_5/")
experiment("parameters/disc16x16x3/spatial_tensor.json","networks/disc16x16x3/net_6/")
experiment("parameters/disc16x16x3/spatial_tensor.json","networks/disc16x16x3/net_7/")

 
'''
experiment("parameters/disc16x16x3/spatial_tensor_depth.json","networks/disc16x16x3/net_4_depth/")
experiment("parameters/disc16x16x3/spatial_tensor_depth.json","networks/disc16x16x3/net_5_depth/")
experiment("parameters/disc16x16x3/spatial_tensor_depth.json","networks/disc16x16x3/net_6_depth/")
experiment("parameters/disc16x16x3/spatial_tensor_depth.json","networks/disc16x16x3/net_7_depth/")

experiment("parameters/disc40x20x3/spatial_tensor.json","networks/disc40x20x3/net_4/")
experiment("parameters/disc40x20x3/spatial_tensor.json","networks/disc40x20x3/net_5/")
experiment("parameters/disc40x20x3/spatial_tensor.json","networks/disc40x20x3/net_6/")
experiment("parameters/disc40x20x3/spatial_tensor.json","networks/disc40x20x3/net_7/")

experiment("parameters/disc40x20x3/spatial_tensor_depth.json","networks/disc40x20x3/net_4_depth/")
experiment("parameters/disc40x20x3/spatial_tensor_depth.json","networks/disc40x20x3/net_5_depth/")
experiment("parameters/disc40x20x3/spatial_tensor_depth.json","networks/disc40x20x3/net_6_depth/")
experiment("parameters/disc40x20x3/spatial_tensor_depth.json","networks/disc40x20x3/net_7_depth/")
'''

print("program done")
