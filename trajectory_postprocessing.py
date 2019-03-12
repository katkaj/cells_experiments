import json
import numpy
import tensor_load
import matplotlib.pyplot as plt


def compute_axis_error(target_matrix, computed_matrix):
    error = target_matrix - computed_matrix

    eps = 10**-20

    max     = target_matrix.max()
    min     = target_matrix.min()
    mean    = error.mean()
    sigma   = numpy.std(error)

    rms             = numpy.sqrt(numpy.mean(numpy.square(error)))
    rms_relative    = 100.0*rms/(numpy.absolute(max - min) + eps)

    decimal_places = 2
    mean = round(mean, decimal_places)
    sigma = round(sigma, decimal_places)
    rms = round(rms, decimal_places)
    rms_relative = round(rms_relative, decimal_places)


    #return [rms, rms_relative]

    return [mean, sigma, rms, rms_relative]


def compute_errors(target_tensor, computed_tensor, id = 0, verbose = False):

    result_x = compute_axis_error(target_tensor[0], computed_tensor[0])
    result_y = compute_axis_error(target_tensor[1], computed_tensor[1])
    result_z = compute_axis_error(target_tensor[2], computed_tensor[2])
    result_total = compute_axis_error(target_tensor, computed_tensor)

    print(id, end=" ")
    #print(result_x[0], result_x[1], result_x[2], result_x[3], end=" ")
    #print(result_y[0], result_y[1], result_y[2], result_y[3], end=" ")
    #print(result_z[0], result_z[1], result_z[2], result_z[3], end=" ")
    print(result_total[0], result_total[1], result_total[2], result_total[3], end=" ")
    print()

#program starts here :

load_start_offset = 800
load_reshaped = True

#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad("trajectory_result/target.json", load_start_offset, load_reshaped)

target_tensor.print_info()

print("loading experiment 0")
experiment_0_tensor = tensor_load.TensorLoad("trajectory_result/experiment_0.json", load_start_offset, load_reshaped)

print("loading experiment 1")
experiment_1_tensor = tensor_load.TensorLoad("trajectory_result/experiment_1.json", load_start_offset, load_reshaped)

print("loading experiment 2")
experiment_2_tensor = tensor_load.TensorLoad("trajectory_result/experiment_2.json", load_start_offset, load_reshaped)

print("loading experiment 3")
experiment_3_tensor = tensor_load.TensorLoad("trajectory_result/experiment_3.json", load_start_offset, load_reshaped)

print("loading experiment 4")
experiment_4_tensor = tensor_load.TensorLoad("trajectory_result/experiment_4.json", load_start_offset, load_reshaped)

print("loading experiment 5")
experiment_5_tensor = tensor_load.TensorLoad("trajectory_result/experiment_5.json", load_start_offset, load_reshaped)

print("loading experiment 6")
experiment_6_tensor = tensor_load.TensorLoad("trajectory_result/experiment_6.json", load_start_offset, load_reshaped)

print("loading experiment 7")
experiment_7_tensor = tensor_load.TensorLoad("trajectory_result/experiment_7.json", load_start_offset, load_reshaped)

print("\n\n")

compute_errors(target_tensor.get(), experiment_0_tensor.get(), 0)
compute_errors(target_tensor.get(), experiment_1_tensor.get(), 1)
compute_errors(target_tensor.get(), experiment_2_tensor.get(), 2)
compute_errors(target_tensor.get(), experiment_3_tensor.get(), 3)
compute_errors(target_tensor.get(), experiment_4_tensor.get(), 4)
compute_errors(target_tensor.get(), experiment_5_tensor.get(), 5)
compute_errors(target_tensor.get(), experiment_6_tensor.get(), 6)
compute_errors(target_tensor.get(), experiment_7_tensor.get(), 7)

'''
0 -9.803 38.882 40.098 15.876 16.103
1 -36.993 215.021 218.18 45.865 46.714
2 -3.854 23.106 23.425 9.933 14.158
3 -0.32 16.087 16.09 7.615 12.418
4 -1.263 11.439 11.508 4.194 8.447
5 -0.246 11.274 11.276 4.659 8.72
6 -0.852 12.262 12.292 4.559 9.245
7 -1.674 11.9 12.018 4.097 8.288
'''
