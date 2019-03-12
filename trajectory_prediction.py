import sys
import libs_python.pyphy as pyphy



dats_to_motion_tensor = pyphy.DatsToMotionTensor("parameters/testing_dats.json", "parameters/motion_tensor.json")

dats_to_motion_tensor.tensor().save_json("trajectory_result/target.json")

prediction_offset = 800


print("processing experiment 7")

tensor_interface = pyphy.TensorSpatial("parameters/spatial_tensor.json", dats_to_motion_tensor.tensor())

prediction = pyphy.TrajectoryPrediction(dats_to_motion_tensor.tensor())

prediction.process( "networks/experiment_7/trained/cnn_config.json", tensor_interface, prediction_offset)
prediction.get_result().save_json("trajectory_result/experiment_7.json")

print("program done")
