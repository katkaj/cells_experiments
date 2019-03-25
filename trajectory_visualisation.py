import sys
import libs_python.pyphy as pyphy



target = pyphy.MotionTensor()
target.load_json("trajectory_result/target.json")

experiment_gaussian = pyphy.MotionTensor()
experiment_gaussian.load_json("trajectory_result/disc16x16x3/net_6.json")
#experiment_gaussian.load_json("trajectory_result/disc8x8x3/net_6_depth.json")

experiment_no_gaussian = pyphy.MotionTensor()
experiment_no_gaussian.load_json("trajectory_result/disc16x16x3_no_gaussian/net_6.json")
#experiment_no_gaussian.load_json("trajectory_result/disc8x8x3_no_gaussian/net_7_depth.json")

visualisation = pyphy.MotionTensorVisualisation()

while True:
    visualisation.start()
    visualisation.render(target,        1.0, 0.0, 0.0)
    visualisation.render(experiment_gaussian,  0.0, 0.0, 1.0)
    visualisation.render(experiment_no_gaussian,  0.0, 1.0, 0.0)
    visualisation.finish()
