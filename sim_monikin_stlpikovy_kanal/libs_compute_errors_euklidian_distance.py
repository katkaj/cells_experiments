import json
import numpy
import tensor_load
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def compute_axis_error(target_tensor, computed_tensor):
    error = target_tensor - computed_tensor

    eps = 10**-20

    max     = target_tensor.max()
    min     = target_tensor.min()
    mean    = error.mean()
    sigma   = numpy.std(error)

    rms             = numpy.sqrt(numpy.mean(numpy.square(error)))
    rms_relative    = 100.0*rms/(numpy.absolute(max - min) + eps)

    absolute        = numpy.mean(numpy.absolute(error))

    decimal_places  = 2
    mean            = round(mean, decimal_places)
    sigma           = round(sigma, decimal_places)
    absolute        = round(absolute, decimal_places)
    rms             = round(rms, decimal_places)
    rms_relative    = round(rms_relative, decimal_places)


    #return [rms, rms_relative]

    return [mean, sigma, absolute, rms, rms_relative]

def histogram(values):
    #values = error_euclidian.reshape(time_steps*cells_count)
    #n, bins, patches = plt.hist(values, bins='auto', facecolor='blue', alpha=0.5)
    n, bins = numpy.histogram(values, bins='auto', density = True)

    #plt.xlabel('Residual')
    #plt.ylabel('Probability')
    #plt.title('Histogram of trajectories residuals')

    #plt.show()
    #plt.savefig('histogram.png')
    return [n, bins]

def compute_euklidian_distance_error(target_tensor, computed_tensor):
    error = target_tensor - computed_tensor

    dim   = len(error)
    time_steps  = len(error[0])
    cells_count   = len(error[0][0])

    error_euclidian = numpy.zeros((time_steps, cells_count))          #residuals of target and predicted value in euclidian distance
    for time in range(0, time_steps):
        for cell in range(0, cells_count):
            err = 0.0
            err+= error[0][time][cell]*error[0][time][cell]
            err+= error[1][time][cell]*error[1][time][cell]
            err+= error[2][time][cell]*error[2][time][cell]
            err = err**0.5
            error_euclidian[time][cell] = err                                      # s tym cyklom to je ok?

    eps = 10**-20

    max_err_x     = target_tensor[0].max()
    max_err_y     = target_tensor[1].max()
    max_err_z     = target_tensor[2].max()
    min_err_x     = target_tensor[0].min()
    min_err_y     = target_tensor[1].min()
    min_err_z     = target_tensor[2].min()
    max_distance  = numpy.sqrt(max_err_x*max_err_x + max_err_y*max_err_y + max_err_z*max_err_z) - numpy.sqrt(min_err_x*min_err_x + min_err_y*min_err_y + min_err_z*min_err_z)
    
    mean_err    = error_euclidian.mean()
    sigma_err   = numpy.std(error_euclidian)
    rms_err             = numpy.sqrt(numpy.mean(numpy.square(error_euclidian)))
    rms_relative_err    = 100.0*rms_err/(max_distance + eps)



    mean_err_cell           = numpy.zeros(cells_count)
    std_err_cell            = numpy.zeros(cells_count)
    rms_err_cell            = numpy.zeros(cells_count) 
    rms_relative_err_cell   = numpy.zeros(cells_count)  
    absolute_err            = numpy.zeros(cells_count) 
    
    for cell in range(cells_count):       
        mean_err_cell[cell]  = numpy.mean(error_euclidian[:,cell])
        std_err_cell[cell]   = numpy.std(error_euclidian[:,cell])

        
        rms_err_cell[cell]  = numpy.sqrt(numpy.mean(numpy.square(error_euclidian[:,cell])))

        absolute_err[cell]  = numpy.mean(numpy.absolute(error_euclidian[:,cell]))


        max_err_cell_x    = target_tensor[0,:,cell].max()
        max_err_cell_y    = target_tensor[1,:,cell].max()
        max_err_cell_z    = target_tensor[2,:,cell].max()
        min_err_cell_x    = target_tensor[0,:,cell].min()
        min_err_cell_y    = target_tensor[1,:,cell].min()
        min_err_cell_z    = target_tensor[2,:,cell].min()
        max_distance_cell           = numpy.sqrt(max_err_cell_x*max_err_cell_x + max_err_cell_y*max_err_cell_y + max_err_cell_z*max_err_cell_z) - numpy.sqrt(min_err_cell_x*min_err_cell_x + min_err_cell_y*min_err_cell_y + min_err_cell_z*min_err_cell_z)
        rms_relative_err_cell[cell] = 100.0*rms_err_cell[cell]/(max_distance_cell + eps)

       

    histogram_n, histogram_bins = histogram(error_euclidian.reshape(time_steps*cells_count))

    return [mean_err,sigma_err,rms_err,rms_relative_err], [mean_err_cell, std_err_cell, rms_err_cell, rms_relative_err_cell, absolute_err], [histogram_n, histogram_bins]


def compute_errors(target_tensor, computed_tensor, verbose = False):

    json_result = {}
    result_x = compute_axis_error(target_tensor[0], computed_tensor[0])
    result_y = compute_axis_error(target_tensor[1], computed_tensor[1])
    result_z = compute_axis_error(target_tensor[2], computed_tensor[2])
    result_total = compute_axis_error(target_tensor, computed_tensor)     #co je tu max a min?
    
    
    result_total_euklidian, result_cell_euklidian, result_histogram_cell = compute_euklidian_distance_error(target_tensor, computed_tensor)

    if verbose:
        print(result_x[0], result_x[1], result_x[2], result_x[3], result_x[4])
        print(result_y[0], result_y[1], result_y[2], result_y[3], result_y[4])
        print(result_z[0], result_z[1], result_z[2], result_z[3], result_z[4])
        print(result_total[0], result_total[1], result_total[2], result_total[3], result_total[4])
        #print(result_total_euklidian[0], result_total_euklidian[1], result_total_euklidian[2], result_total_euklidian[3], result_total_euklidian[4], result_total_euklidian[5], result_total_euklidian[6], result_total_euklidian[7], result_total_euklidian[8], end=" ")
        print()

    json_result["x"] = {}
    json_result["x"]["mean"] = result_x[0]
    json_result["x"]["sigma"] = result_x[1]
    json_result["x"]["error_absolute"] = result_x[2]
    json_result["x"]["rms"] = result_x[3]
    json_result["x"]["rms_relative"] = result_x[4]

    json_result["y"] = {}
    json_result["y"]["mean"] = result_y[0]
    json_result["y"]["sigma"] = result_y[1]
    json_result["y"]["error_absolute"] = result_y[2]
    json_result["y"]["rms"] = result_y[3]
    json_result["y"]["rms_relative"] = result_y[4]

    json_result["z"] = {}
    json_result["z"]["mean"] = result_z[0]
    json_result["z"]["sigma"] = result_z[1]
    json_result["z"]["error_absolute"] = result_z[2]
    json_result["z"]["rms"] = result_z[3]
    json_result["z"]["rms_relative"] = result_z[4]

    json_result["total"] = {}
    json_result["total"]["mean"] = result_total[0]
    json_result["total"]["sigma"] = result_total[1]
    json_result["total"]["error_absolute"] = result_total[2]
    json_result["total"]["rms"] = result_total[3]
    json_result["total"]["rms_relative"] = result_total[4]


    json_result["euklidian"] = {}
    json_result["euklidian"]["mean"] = result_total_euklidian[0]
    json_result["euklidian"]["sigma"] = result_total_euklidian[1]
    json_result["euklidian"]["rms"] = result_total_euklidian[2]
    json_result["euklidian"]["rms_relative"] = result_total_euklidian[3]
    
    
    json_result["euklidian"]["cells"] = {}
    
    for cell in range(computed_tensor.shape[2]):
        json_result["euklidian"]["cells"][cell] = {}
        json_result["euklidian"]["cells"][cell][0]        = result_cell_euklidian[0][cell]  #mean
        json_result["euklidian"]["cells"][cell][1]        = result_cell_euklidian[1][cell]  #sigma
        json_result["euklidian"]["cells"][cell][2]        = result_cell_euklidian[2][cell]  #rms
        json_result["euklidian"]["cells"][cell][3]        = result_cell_euklidian[3][cell]  #rms_relative
        json_result["euklidian"]["cells"][cell][4]        = result_cell_euklidian[4][cell]  #absolute_err
        

    return json_result, result_histogram_cell



def save_errors(target_tensor, computed_tensor, path):
    json_result, result_histogram_cell = compute_errors(target_tensor, computed_tensor)

    json_file = open(path + "errors.json", 'w')
    json.dump(json_result, json_file)

    errors_file = open(path + "errors.txt", 'w')

    str_res = ""
    str_res+= "x mean " + str(json_result["x"]["mean"]) + "\n"
    str_res+= "x sigma " + str(json_result["x"]["sigma"]) + "\n"
    str_res+= "x error_absolute " + str(json_result["x"]["error_absolute"]) + "\n"
    str_res+= "x rms " + str(json_result["x"]["rms"]) + "\n"
    str_res+= "x rms_relative " + str(json_result["x"]["rms_relative"]) + "\n"
    str_res+= "\n"
    
    str_res+= "y mean " + str(json_result["y"]["mean"]) + "\n"
    str_res+= "y sigma " + str(json_result["y"]["sigma"]) + "\n"
    str_res+= "y error_absolute " + str(json_result["y"]["error_absolute"]) + "\n"
    str_res+= "y rms " + str(json_result["y"]["rms"]) + "\n"
    str_res+= "y rms_relative " + str(json_result["y"]["rms_relative"]) + "\n"
    str_res+= "\n"

    str_res+= "z mean " + str(json_result["z"]["mean"]) + "\n"
    str_res+= "z sigma " + str(json_result["z"]["sigma"]) + "\n"
    str_res+= "z error_absolute " + str(json_result["z"]["error_absolute"]) + "\n"
    str_res+= "z rms " + str(json_result["z"]["rms"]) + "\n"
    str_res+= "z rms_relative " + str(json_result["z"]["rms_relative"]) + "\n"
    str_res+= "\n"

    
    str_res+= "total mean " + str(json_result["total"]["mean"]) + "\n"
    str_res+= "total sigma " + str(json_result["total"]["sigma"]) + "\n"
    str_res+= "total error_absolute " + str(json_result["total"]["error_absolute"]) + "\n"
    str_res+= "total rms " + str(json_result["total"]["rms"]) + "\n"
    str_res+= "total rms_relative " + str(json_result["total"]["rms_relative"]) + "\n"
    str_res+= "\n"


    str_res+= "euklidian mean " + str(json_result["euklidian"]["mean"]) + "\n"
    str_res+= "euklidian sigma " + str(json_result["euklidian"]["sigma"]) + "\n"
   #str_res+= "euklidian error_absolute " + str(json_result["euklidian"]["error_absolute"]) + "\n"
    str_res+= "euklidian rms " + str(json_result["euklidian"]["rms"]) + "\n"
    str_res+= "euklidian rms_relative " + str(json_result["euklidian"]["rms_relative"]) + "\n"
    str_res+= "\n"


    errors_file.write(str_res)


    cells_errors_file = open(path + "cells_errors.txt", 'w')
    cells_errors_file.write("cell, mean, sigma, rms, rms_relative, absolute_err\n")
    for cell in range(computed_tensor.shape[2]):
        str_res = str(cell) + " "
        str_res+= str(json_result["euklidian"]["cells"][cell][0]) + " "
        str_res+= str(json_result["euklidian"]["cells"][cell][1]) + " "
        str_res+= str(json_result["euklidian"]["cells"][cell][2]) + " "
        str_res+= str(json_result["euklidian"]["cells"][cell][3]) + " "
        str_res+= str(json_result["euklidian"]["cells"][cell][4]) + " "
        str_res+= "\n"
        cells_errors_file.write(str_res)

