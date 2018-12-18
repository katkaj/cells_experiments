import sys
sys.path.insert(0, '../physical_modeling/libs_python')
import pyphy


dats_to_motion_tensor = pyphy.DatsToMotionTensor("parameters/testing_dats.json", "parameters/motion_tensor.json")

dats_to_motion_tensor.tensor().save_json("trajectory_result/target.json")

prediction_offset = 800


print("processing experiment 7")

tensor_interface = pyphy.TensorSpatial("parameters/spatial_tensor.json", dats_to_motion_tensor.tensor())

prediction = pyphy.TrajectoryPrediction(dats_to_motion_tensor.tensor())

prediction.process( "networks/experiment_7/trained/cnn_config.json", tensor_interface, prediction_offset)
prediction.get_result().save_json("trajectory_result/experiment_7.json")

print("program done")
