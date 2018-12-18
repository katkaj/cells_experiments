import sys
sys.path.insert(0, '../physical_modeling/libs_python')
import pyphy


#1, load data from dats files and create motion tensor with normalised columns

#load training data
training_dats_to_motion_tensor = pyphy.DatsToMotionTensor("parameters/training_dats.json", "parameters/motion_tensor.json")

#load testing data, for normalisation use range from testing
testing_dats_to_motion_tensor = pyphy.DatsToMotionTensor("parameters/testing_dats.json", "parameters/motion_tensor.json", training_dats_to_motion_tensor.tensor())

#2, create network input making class - TensorSpatial
training_tensor = pyphy.TensorSpatial("parameters/spatial_tensor.json", training_dats_to_motion_tensor.tensor())
testing_tensor  = pyphy.TensorSpatial("parameters/spatial_tensor.json", testing_dats_to_motion_tensor.tensor())

'''
#2, create network input making class - TensorNoSpatial
training_tensor = pyphy.TensorNoSpatial("parameters/no_spatial_tensor.json", training_dats_to_motion_tensor.tensor())
testing_tensor  = pyphy.TensorNoSpatial("parameters/no_spatial_tensor.json", testing_dats_to_motion_tensor.tensor())
'''

#3, create dataset
dataset = pyphy.DatasetTrajectory(training_tensor, training_tensor)



#4, run experiments, train network

experiment_12 = pyphy.RegressionExperiment(dataset, "networks/experiment_empty/")
experiment_12.run()


print("program done")