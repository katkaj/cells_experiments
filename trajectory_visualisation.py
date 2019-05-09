import sys
import libs_python.pyphy as pyphy

target_trajectory = "/home/michal/programming/cells_results/sim26/target.json"
results_path = "/home/michal/programming/cells_results/sim26/discretisation_16x16x3/window_size_8/point/"


target = pyphy.MotionTensor()
target.load_json(target_trajectory)

experiment_4 = pyphy.MotionTensor()
experiment_4.load_json(results_path + "net_4.json")


#experiment_7 = pyphy.MotionTensor()
#experiment_7.load_json(results_path + "net_7.json")
#experiment_no_gaussian.load_json("trajectory_result/disc8x8x3_no_gaussian/net_7_depth.json")

visualisation = pyphy.MotionTensorVisualisation(100)

while True:
    visualisation.start()
    visualisation.render(target,        1.0, 0.0, 0.0)
    visualisation.render(experiment_4,  0.0, 0.0, 1.0)
    #visualisation.render(experiment_7,  0.0, 0.8, 0.0)
    visualisation.finish()
