import json
import numpy
import tensor_load

import libs_compute_errors

def save_results(file_name_prefix, json_result):

    json_file_name = file_name_prefix + ".json"

    json_out_file = open(json_file_name, "w")
    json.dump(json_result, json_out_file)


#program starts here :

load_start_offset   = 800
load_reshaped       = True

result_path         = "/home/michal/programming/cells_results/sim26/"
discretisations     = ["discretisation_8x8x3", "discretisation_16x16x3", "discretisation_40x20x3"]
window_sizes        = ["window_size_4", "window_size_8"]
filter_modes        = ["gaussian", "point"]
networks            = ["net_4.json", "net_4_depth.json", "net_5.json", "net_5_depth.json", "net_6.json", "net_6_depth.json", "net_7.json", "net_7_depth.json"]

json_result = {}
json_result["results"] = []

#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad(result_path + "target.json", load_start_offset, load_reshaped)
target_tensor.print_info()


print("computing errors for ")

id = 0
for discretisation in discretisations:
    print("discretisation", discretisation)
    for window_size in window_sizes:
        print("\twindow_size", window_size)
        for filter_mode in filter_modes:
            print("\t\tfilter_mode", filter_mode)
            for network in networks:
                print("\t\t\tnetworks", network)
                json_file_name = result_path + discretisation + "/" + window_size + "/" + filter_mode + "/" + network

                experiment_tensor = tensor_load.TensorLoad(json_file_name, load_start_offset, load_reshaped)

                parameters= { }
                parameters["id"]             = id;
                parameters["discretisation"] = discretisation
                parameters["window_size"]    = window_size
                parameters["filter_mode"]    = filter_mode
                parameters["network"]        = network

                errors = libs_compute_errors.compute_errors(target_tensor.get(), experiment_tensor.get())
                print("\t\t\t\t", errors["total"]["rms_relative"])

                experiment_result = {**parameters, **errors}
                #print(experiment_result)

                json_result["results"].append(experiment_result)

                id+= 1

print("computing done")

print("saving results")
save_results(result_path + "/errors", json_result)

print("program done")
