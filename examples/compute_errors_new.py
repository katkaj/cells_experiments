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
networks            = ["net_4", "net_4_depth", "net_5", "net_5_depth", "net_6", "net_6_depth", "net_7", "net_7_depth"]

json_result = {}
json_result["results"] = []

#load target values -> ground truth (real trajectory)
print("loading target")
target_tensor = tensor_load.TensorLoad(result_path + "target.json", load_start_offset, load_reshaped)
target_tensor.print_info()


print("computing errors for ")


decimation = 100
id = 0
for discretisation in discretisations:
    print("discretisation", discretisation)
    for window_size in window_sizes:
        print("\twindow_size", window_size)
        for filter_mode in filter_modes:
            print("\t\tfilter_mode", filter_mode)
            for network in networks:
                print("\t\t\tnetworks", network)
                json_file_name = result_path + discretisation + "/" + window_size + "/" + filter_mode + "/" + network + ".json"

                experiment_tensor = tensor_load.TensorLoad(json_file_name, load_start_offset, load_reshaped)

                parameters= { }
                parameters["id"]             = id;
                parameters["discretisation"] = discretisation
                parameters["window_size"]    = window_size
                parameters["filter_mode"]    = filter_mode
                parameters["network"]        = network

                errors = libs_compute_errors.compute_errors(target_tensor.get(), experiment_tensor.get())


                error_field_file_name = result_path + discretisation + "/" + window_size + "/" + filter_mode + "/" + network + "_error_field.dat"

                libs_compute_errors.compute_error_field(target_tensor.get(), experiment_tensor.get(), decimation, error_field_file_name, load_reshaped)

                print("\t\t\t\t", errors["total"]["rms_relative"])

                experiment_result = {**parameters, **errors}
                #print(experiment_result)

                json_result["results"].append(experiment_result)

                id+= 1

print("computing done")

print("saving results")
save_results(result_path + "/errors_new", json_result)

print("program done")
