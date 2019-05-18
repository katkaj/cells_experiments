import sys
sys.path.append('..')
import libs_python.pyphy as pyphy


#convert dats files to json trajectory
dats_to_motion_tensor = pyphy.DatsToMotionTensor("testing_dats.json", "motion_tensor.json")

#save into "target_trajectory.json"
dats_to_motion_tensor.tensor().save_json("trajectory_result/target_trajectory.json")

#load trajectory json
target = pyphy.MotionTensor()
target.load_json("trajectory_result/target_trajectory.json")


#load trajectory result
result = pyphy.MotionTensor()
result.load_json("trajectory_result/net_0_single.json")

#use only 200 points for render
visualisation = pyphy.MotionTensorVisualisation(200)

while True:
    visualisation.start()
    visualisation.render(target, 1.0, 0.0, 0.0)
    visualisation.render(result, 0.0, 0.0, 1.0)
    visualisation.finish()
