import sys
import libs_python.pyphy as pyphy



target = pyphy.MotionTensor()
target.load_json("trajectory_result/target.json")

experiment_0 = pyphy.MotionTensor()
experiment_0.load_json("trajectory_result/experiment_7.json")

visualisation = pyphy.MotionTensorVisualisation()

while True:
    visualisation.start()
    visualisation.render(target,        1.0, 0.0, 0.0)
    visualisation.render(experiment_0,  0.0, 0.0, 1.0)
    visualisation.finish()
