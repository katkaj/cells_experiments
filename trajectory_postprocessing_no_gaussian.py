import json
import pandas
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

    absolute        = numpy.mean(numpy.absolute(error))

    decimal_places = 2
    mean = round(mean, decimal_places)
    sigma = round(sigma, decimal_places)
    absolute = round(absolute, decimal_places)
    rms = round(rms, decimal_places)
    rms_relative = round(rms_relative, decimal_places)


    #return [rms, rms_relative]

    return [mean, sigma, absolute, rms, rms_relative]


def compute_errors(target_tensor, computed_tensor, id = 0, verbose = False):

    json_result = {}

    result_x = compute_axis_error(target_tensor[0], computed_tensor[0])
    result_y = compute_axis_error(target_tensor[1], computed_tensor[1])
    result_z = compute_axis_error(target_tensor[2], computed_tensor[2])
    result_total = compute_axis_error(target_tensor, computed_tensor)

    print(id, end=" ")
    #print(result_x[0], result_x[1], result_x[2], result_x[3])
    #print(result_y[0], result_y[1], result_y[2], result_y[3])
    #print(result_z[0], result_z[1], result_z[2], result_z[3])
    print(result_total[0], result_total[1], result_total[2], result_total[3], end=" ")
    print()

    json_result["id"] = id

    json_result["x"] = {}
    json_result["x"]["mean"] = result_x[0]
    json_result["x"]["sigma"] = result_x[1]
    json_result["x"]["error absolute"] = result_x[2]
    json_result["x"]["rms"] = result_x[3]
    json_result["x"]["rms_relative"] = result_x[4]

    json_result["y"] = {}
    json_result["y"]["mean"] = result_y[0]
    json_result["y"]["sigma"] = result_y[1]
    json_result["y"]["error absolute"] = result_x[2]
    json_result["y"]["rms"] = result_y[3]
    json_result["y"]["rms_relative"] = result_y[4]

    json_result["z"] = {}
    json_result["z"]["mean"] = result_z[0]
    json_result["z"]["sigma"] = result_z[1]
    json_result["z"]["error absolute"] = result_x[2]
    json_result["z"]["rms"] = result_z[3]
    json_result["z"]["rms_relative"] = result_z[4]

    json_result["total"] = {}
    json_result["total"]["mean"] = result_total[0]
    json_result["total"]["sigma"] = result_total[1]
    json_result["total"]["error absolute"] = result_x[2]
    json_result["total"]["rms"] = result_total[3]
    json_result["total"]["rms_relative"] = result_total[4]

    return json_result
    
def save_results(file_name_prefix, json_result):

    json_file_name = file_name_prefix + ".json"

    json_out_file = open(json_file_name, "w")
    json.dump(json_result, json_out_file)


#program starts here :

load_start_offset = 800
load_reshaped = True

#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad("trajectory_result/target.json", load_start_offset, load_reshaped)

target_tensor.print_info()




print("loading experiments")


experiment_disc8x8x3_no_gaussian_net_4_tensor = tensor_load.TensorLoad("trajectory_result/disc8x8x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_4_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc8x8x3_no_gaussian/net_4_depth.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_5_tensor = tensor_load.TensorLoad("trajectory_result/disc8x8x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_5_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc8x8x3_no_gaussian/net_5_depth.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_6_tensor = tensor_load.TensorLoad("trajectory_result/disc8x8x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_6_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc8x8x3_no_gaussian/net_6_depth.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_7_tensor = tensor_load.TensorLoad("trajectory_result/disc8x8x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
experiment_disc8x8x3_no_gaussian_net_7_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc8x8x3_no_gaussian/net_7_depth.json", load_start_offset, load_reshaped)

experiment_disc16x16x3_no_gaussian_net_4_tensor = tensor_load.TensorLoad("trajectory_result/disc16x16x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_4_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc16x16x3_no_gaussian/net_4_depth.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_5_tensor = tensor_load.TensorLoad("trajectory_result/disc16x16x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_5_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc16x16x3_no_gaussian/net_5_depth.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_6_tensor = tensor_load.TensorLoad("trajectory_result/disc16x16x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_6_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc16x16x3_no_gaussian/net_6_depth.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_7_tensor = tensor_load.TensorLoad("trajectory_result/disc16x16x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
experiment_disc16x16x3_no_gaussian_net_7_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc16x16x3_no_gaussian/net_7_depth.json", load_start_offset, load_reshaped)

'''
experiment_disc40x20x3_no_gaussian_net_4_tensor = tensor_load.TensorLoad("trajectory_result/disc40x20x3_no_gaussian/net_4.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_4_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc40x20x3_no_gaussian/net_4_depth.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_5_tensor = tensor_load.TensorLoad("trajectory_result/disc40x20x3_no_gaussian/net_5.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_5_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc40x20x3_no_gaussian/net_5_depth.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_6_tensor = tensor_load.TensorLoad("trajectory_result/disc40x20x3_no_gaussian/net_6.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_6_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc40x20x3_no_gaussian/net_6_depth.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_7_tensor = tensor_load.TensorLoad("trajectory_result/disc40x20x3_no_gaussian/net_7.json", load_start_offset, load_reshaped)
experiment_disc40x20x3_no_gaussian_net_7_depth_tensor = tensor_load.TensorLoad("trajectory_result/disc40x20x3_no_gaussian/net_7_depth.json", load_start_offset, load_reshaped)
'''

print()
print("computing errors")
print()

json_result = {}
json_result["results"] = []


json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_4_tensor.get(), "disc8x8x3_no_gaussian_net_4"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_4_depth_tensor.get(), "disc8x8x3_no_gaussian_net_4_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_5_tensor.get(), "disc8x8x3_no_gaussian_net_5"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_5_depth_tensor.get(), "disc8x8x3_no_gaussian_net_5_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_6_tensor.get(), "disc8x8x3_no_gaussian_net_6"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_6_depth_tensor.get(), "disc8x8x3_no_gaussian_net_6_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_7_tensor.get(), "disc8x8x3_no_gaussian_net_7"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc8x8x3_no_gaussian_net_7_depth_tensor.get(), "disc8x8x3_no_gaussian_net_7_depth"))

json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_4_tensor.get(), "disc16x16x3_no_gaussian_net_4"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_4_depth_tensor.get(), "disc16x16x3_no_gaussian_net_4_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_5_tensor.get(), "disc16x16x3_no_gaussian_net_5"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_5_depth_tensor.get(), "disc16x16x3_no_gaussian_net_5_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_6_tensor.get(), "disc16x16x3_no_gaussian_net_6"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_6_depth_tensor.get(), "disc16x16x3_no_gaussian_net_6_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_7_tensor.get(), "disc16x16x3_no_gaussian_net_7"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc16x16x3_no_gaussian_net_7_depth_tensor.get(), "disc16x16x3_no_gaussian_net_7_depth"))

'''
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_4_tensor.get(), "disc40x20x3_no_gaussian_net_4"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_4_depth_tensor.get(), "disc40x20x3_no_gaussian_net_4_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_5_tensor.get(), "disc40x20x3_no_gaussian_net_5"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_5_depth_tensor.get(), "disc40x20x3_no_gaussian_net_5_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_6_tensor.get(), "disc40x20x3_no_gaussian_net_6"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_6_depth_tensor.get(), "disc40x20x3_no_gaussian_net_6_depth"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_7_tensor.get(), "disc40x20x3_no_gaussian_net_7"))
json_result["results"].append(compute_errors(target_tensor.get(), experiment_disc40x20x3_no_gaussian_net_7_depth_tensor.get(), "disc40x20x3_no_gaussian_net_7_depth"))
'''


print("saving results")
save_results("trajectory_result/errors_no_gaussian", json_result)


print("program done")
