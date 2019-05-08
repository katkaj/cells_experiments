import sys
import libs_python.pyphy as pyphy

dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim_50/testing_dats.json", "experiments/sim_50/motion_tensor.json")
#dats_to_motion_tensor = pyphy.DatsToMotionTensor("experiments/sim_dense_collisions_01/testing_dats.json", "experiments/sim_dense_collisions_01/motion_tensor.json")

dats_to_motion_tensor.tensor().save_json("target_02.json")


target = pyphy.MotionTensor()
target.load_json("target_02.json")

visualisation = pyphy.MotionTensorVisualisation()

while True:
    visualisation.start()
    visualisation.render(target, 1.0, 0.0, 0.0)
    visualisation.finish()
