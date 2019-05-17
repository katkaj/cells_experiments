import sys
import libs_python.pyphy as pyphy

dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim_monika/training_dats.json", "experiments/sim_monika/motion_tensor.json")
dats_to_motion_tensor.tensor().save_json("trajectory_result/target_monika.json")


target = pyphy.MotionTensor()
target.load_json("trajectory_result/target_monika.json")

visualisation = pyphy.MotionTensorVisualisation()

while True:
    visualisation.start()
    visualisation.render(target, 1.0, 0.0, 0.0)
    visualisation.finish()
