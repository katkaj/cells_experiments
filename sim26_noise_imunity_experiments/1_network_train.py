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
    testing_count = 5000
    print("creating dataset")
    dataset = pyphy.DatasetTrajectoryRuntime(training_tensor, training_tensor, testing_count)

    #4, run experiments, train network
    print("training")
    experiment = pyphy.RegressionExperiment(dataset, network_config, "network_config.json")
    experiment.run()

    print("training done")
    print("\n\n\n\n")


def experiment(experiment_path, skip = False):
    #train networks

    if skip == True:
        train_network(experiment_path + "spatial_tensor_single.json",   experiment_path + "spatial_tensor_single.json",  experiment_path + "net_0_single/")
        train_network(experiment_path + "spatial_tensor_all.json",      experiment_path + "spatial_tensor_all.json",     experiment_path + "net_0_all/")
        train_network(experiment_path + "spatial_tensor_single.json",   experiment_path + "spatial_tensor_single.json",  experiment_path + "net_1_single/")
        train_network(experiment_path + "spatial_tensor_all.json",      experiment_path + "spatial_tensor_all.json",     experiment_path + "net_1_all/")


    train_network(experiment_path + "spatial_tensor_single.json",   experiment_path + "spatial_tensor_single.json",  experiment_path + "net_2_single/")
    train_network(experiment_path + "spatial_tensor_all.json",      experiment_path + "spatial_tensor_all.json",     experiment_path + "net_2_all/")


#load data from dats files and create motion tensor with normalised columns

#load training data
training_dats_to_motion_tensor = pyphy.DatsToMotionTensor("training_dats.json", "motion_tensor.json")

#load testing data, for normalisation use range from testing tensor
testing_dats_to_motion_tensor = pyphy.DatsToMotionTensor("testing_dats.json", "motion_tensor.json", training_dats_to_motion_tensor.tensor())

experiment("networks/noise_0_0/", True)

experiment("networks/noise_5_0/")
experiment("networks/noise_10_0/")
experiment("networks/noise_15_0/")
experiment("networks/noise_20_0/")
experiment("networks/noise_25_0/")
experiment("networks/noise_30_0/")


print("program done")
