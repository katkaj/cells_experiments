import sys
import libs_python.pyphy as pyphy



target = pyphy.MotionTensor()
target.load_json("trajectory_result/target.json")

experiment = pyphy.MotionTensor()
experiment.load_json("trajectory_result/disc40x20x3/net_7_depth.json")

visualisation = pyphy.MotionTensorVisualisation()

while True:
    visualisation.start()
    visualisation.render(target,        1.0, 0.0, 0.0)
    visualisation.render(experiment,  0.0, 0.0, 1.0)
    visualisation.finish()
