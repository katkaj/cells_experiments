import sys
import libs_python.pyphy as pyphy

dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim26/training_dats.json", "experiments/sim26/motion_tensor.json")
dats_to_motion_tensor.tensor().save_json("trajectory_result/sim26.json")


target = pyphy.MotionTensor()
target.load_json("trajectory_result/sim26.json")

predicted = pyphy.MotionTensor()
predicted.load_json("trajectory_result/sim_50/noise_gaussian_net_0.json")


visualisation = pyphy.MotionTensorVisualisation()

while True:
    visualisation.start()
    visualisation.render(target, 1.0, 0.0, 0.0)
    visualisation.render(predicted, 0.0, 1.0, 0.0)

    visualisation.finish()
