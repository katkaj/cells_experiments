import json
import numpy
import tensor_loading
import matplotlib.pyplot as plt

def compute_errors(target_tensor, computed_tensor, id = 0, verbose = False):

    error = target_tensor - computed_tensor

    error_absolute_tensor   = numpy.absolute(error)
    target_absolute_tensor  = numpy.absolute(target_tensor) + 0.000000001

    relative_error_tensor   = error_absolute_tensor/target_absolute_tensor
    relative_error          = relative_error_tensor.mean()*100.0


    decimal_places = 3

    max     = error.max()
    min     = error.min()
    mean    = error.mean()
    sigma   = numpy.std(error)


    rms      = numpy.sqrt(numpy.mean(numpy.square(error)))
    ams      =  numpy.mean(numpy.absolute(error))

    max     = round(max, decimal_places)
    min     = round(min, decimal_places)
    mean    = round(mean, decimal_places)
    sigma   = round(sigma, decimal_places)
    rms     = round(rms, decimal_places)
    ams     = round(ams, decimal_places)
    relative_error = round(relative_error, decimal_places)

    if verbose == True:
        print("id       ", id)
        print("mean     ", mean)
        print("min      ", min)
        print("max      ", max)
        print("sigma    ", sigma)
        print("rms      ", rms)
        print("abs err  ", ams)
        print("error relative  ", relative_error)
        print()
    else:
        #print("ID","mean", "standart_deviation", "root_mean_square_error", "ams", "relative_error[%]")
        print(id, mean, sigma, rms, ams, relative_error)


#program starts here :

load_start_offset = 800

#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_loading.TensorLoad("trajectory_result/target.json", load_start_offset)

target_tensor.print_info()

print("loading experiment 0")
experiment_0_tensor = tensor_loading.TensorLoad("trajectory_result/experiment_0.json", load_start_offset)

print("loading experiment 1")
experiment_1_tensor = tensor_loading.TensorLoad("trajectory_result/experiment_1.json", load_start_offset)

print("loading experiment 2")
experiment_2_tensor = tensor_loading.TensorLoad("trajectory_result/experiment_2.json", load_start_offset)

print("loading experiment 3")
experiment_3_tensor = tensor_loading.TensorLoad("trajectory_result/experiment_3.json", load_start_offset)

print("loading experiment 4")
experiment_4_tensor = tensor_loading.TensorLoad("trajectory_result/experiment_4.json", load_start_offset)

print("loading experiment 5")
experiment_5_tensor = tensor_loading.TensorLoad("trajectory_result/experiment_5.json", load_start_offset)

print("loading experiment 6")
experiment_6_tensor = tensor_loading.TensorLoad("trajectory_result/experiment_6.json", load_start_offset)

print("loading experiment 7")
experiment_7_tensor = tensor_loading.TensorLoad("trajectory_result/experiment_7.json", load_start_offset)

print("\n\n")

compute_errors(target_tensor.get(), experiment_0_tensor.get(), 0)
compute_errors(target_tensor.get(), experiment_1_tensor.get(), 1)
compute_errors(target_tensor.get(), experiment_2_tensor.get(), 2)
compute_errors(target_tensor.get(), experiment_3_tensor.get(), 3)
compute_errors(target_tensor.get(), experiment_4_tensor.get(), 4)
compute_errors(target_tensor.get(), experiment_5_tensor.get(), 5)
compute_errors(target_tensor.get(), experiment_6_tensor.get(), 6)
compute_errors(target_tensor.get(), experiment_7_tensor.get(), 7)
