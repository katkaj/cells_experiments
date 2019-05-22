import sys
sys.path.append('..')
import libs_python.pyphy as pyphy


def train_network(training_tensor_config, testing_tensor_config, network_config):

    print("\n\n\n\n")
    print("starting training experiment")
    print("training_tensor_config ", training_tensor_config)
    print("testing_tensor_config ", testing_tensor_config)

    print(network_config)

    #2, create network input making class - TensorSpatial
    print("creating training tensor")
    training_tensor = pyphy.TensorSpatial(training_tensor_config, training_dats_to_motion_tensor.tensor())

    print("creating testing tensor")
    testing_tensor  = pyphy.TensorSpatial(testing_tensor_config, testing_dats_to_motion_tensor.tensor())

    #3, create dataset
    testing_count = 10000
    print("creating dataset")
    dataset = pyphy.DatasetTrajectoryRuntime(training_tensor, training_tensor, testing_count)

    #4, run experiments, train network
    print("training")
    experiment = pyphy.RegressionExperiment(dataset, network_config)
    experiment.run()

    print("training done")
    print("\n\n\n\n")



#load data from dats files and create motion tensor with normalised columns

#load training data
training_dats_to_motion_tensor = pyphy.DatsToMotionTensor("training_dats.json", "motion_tensor.json")

#load testing data, for normalisation use range from testing tensor
testing_dats_to_motion_tensor = pyphy.DatsToMotionTensor("testing_dats.json", "motion_tensor.json", training_dats_to_motion_tensor.tensor())

#train four networks

experiment_path = "discretization_experiment/12x16x16_all/"
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_0/")
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_1/")

experiment_path = "discretization_experiment/24x32x32_all/"
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_0/")
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_1/")

experiment_path = "discretization_experiment/48x64x64_all/"
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_0/")
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_1/")

experiment_path = "discretization_experiment/12x16x16_single/"
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_0/")
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_1/")

experiment_path = "discretization_experiment/24x32x32_single/"
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_0/")
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_1/")

experiment_path = "discretization_experiment/48x64x64_single/"
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_0/")
train_network( experiment_path + "spatial_tensor.json", experiment_path + "spatial_tensor.json",  experiment_path + "net_1/")

print("program done")
