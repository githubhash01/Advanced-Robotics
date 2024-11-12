
# TODO: reached if q is within epsilon away from cube. assume that if it collides eveyrthing is messed up.
## TODO: plto path for each forward kinematics
# TODO: change gains over a range
# TODO: with start position and end postioin, run 10 times and how many times correctly.

# test 1: Get the success rate of moving the cube from the start to the end position.
from control import control_main

success = control_main()
print(success)

# test inverse geeomtry with rotate cube
# find the boundaries of IG where there is a solutions. easy rotation vs hard roation if right hand is still ro the right hook.
# if target position is rotated...