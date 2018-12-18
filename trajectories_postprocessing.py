import json
import numpy


#print basic tensor informations
def print_tensor_info(tensor):

    #extract tensor dimensions -> from json to int
    width   = tensor["width"]
    height  = tensor["height"]
    depth   = tensor["depth"]

    #extract tensor extremes in all 3 axis to floats
    min_x = tensor["extremes"][0]["min"]
    max_x = tensor["extremes"][0]["max"]
    min_y = tensor["extremes"][1]["min"]
    max_y = tensor["extremes"][1]["max"]
    min_z = tensor["extremes"][2]["min"]
    max_z = tensor["extremes"][2]["max"]

    #print some info

    print("width = ", width)        #6 axis : 3 for positions, 3 for velocity
    print("height = ", height)      #count of time steps
    print("depth = ", depth)        #count of cells in file
    print("\n")


    print("min x = ", min_x)
    print("max x = ", max_x)
    print("\n")
    print("min y = ", min_y)
    print("max y = ", max_y)
    print("\n")
    print("min z = ", min_z)
    print("max z = ", max_z)
    print("\n")


#print trajectory of selected cell
def print_cell_trajectory(tensor, cell_idx):

    height = tensor["height"]

    for time in range(0, height):
        #print only each 100 step
        if (time%100) == 0:

            x = tensor["values"][cell_idx][time][0] #x axis on index 0
            y = tensor["values"][cell_idx][time][1] #y axis on index 1
            z = tensor["values"][cell_idx][time][2] #z axis on index 2

            print(time, " : ", x, y, z)

def json_to_matrix(json_tensor):

    width   = json_tensor["width"]//2
    height  = json_tensor["height"]
    depth   = json_tensor["depth"]

    result_matrix = numpy.zeros((depth, height, width))
    #nacitat, data denormovat
    min_x = json_tensor["extremes"][0]["min"]
    max_x = json_tensor["extremes"][0]["max"]
    min_y = json_tensor["extremes"][1]["min"]
    max_y = json_tensor["extremes"][1]["max"]
    min_z = json_tensor["extremes"][2]["min"]
    max_z = json_tensor["extremes"][2]["max"]

    for i in range(0, depth):
        for j in range(0,height):
            x_raw = json_tensor["values"][i][j][0]
            y_raw = json_tensor["values"][i][j][1]
            z_raw = json_tensor["values"][i][j][2]

            x = max_x*x_raw + min_x
            y = max_y*y_raw + min_y
            z = max_z*z_raw + min_z

            result_matrix[i][j][0] = x
            result_matrix[i][j][1] = y
            result_matrix[i][j][2] = z

    return result_matrix




#program starts here :

#load target values -> ground truth (real trajectory)
with open("trajectory_result/target.json") as f:
    target_tensor = json.load(f)

target_result_matrix = json_to_matrix(target_tensor)

print(target_result_matrix)



#load target values -> ground truth (real trajectory)
with open("trajectory_result/experiment_0.json") as f:
    experiment_0_tensor = json.load(f)

experiment_0_result_matrix = json_to_matrix(experiment_0_tensor)

print(experiment_0_result_matrix)



#print some info
print_tensor_info(target_tensor)

#print trajectory of cell 20
#print_cell_trajectory(target_tensor, 20)
