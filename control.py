#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""
import time

from inverse_geometry import get_hand_jacobian, find_cube_from_configuration
import pinocchio
from config import RIGHT_HAND, LEFT_HAND, EPSILON, RIGHT_HOOK, LEFT_HOOK
from bezier import Bezier
from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
from config import DT
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose, setcubeplacement
from path import computepath
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pybullet as p

# PD gains set by Steve Thonneau - they work well for this problem
Kp = 400
Kv = 2 * np.sqrt(Kp)
Kgrip = 80 # Gain for gripping the cube

N_BEZIERS = 1


# Function to create a trajectory
def maketraj(path, q0, q1, T):
    """
    Inputs:

    path: a list of configurations
    q0: the initial configuration
    q1: the final configuration
    T: the total time

    Outputs:

    q_of_t: a function q_of_t(t) that returns the configuration at time t
    vq_of_t: a function vq_of_t(t) that returns the velocity at time t
    vvq_of_t: a function vvq_of_t(t) that returns the acceleration at time t
    """
    path = [q0]*3 + path + [q1]*3

    if N_BEZIERS == 1:
        q_of_t = Bezier(pointlist=path, t_min=0.0, t_max=T, mult_t=1.0)

        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)

    else:
        # Split the trajectory into `sub_traj` trajectories
        q_of_t = []
        vq_of_t = []
        vvq_of_t = []
        for i in range(N_BEZIERS):
            start = int(i * len(path) / N_BEZIERS)
            end = int((i + 1) * len(path) / N_BEZIERS)
            t_min = i * T / N_BEZIERS
            t_max = (i + 1) * T / N_BEZIERS
            q_of_t.append(Bezier(pointlist=path[start:end], t_min=t_min, t_max=t_max, mult_t=1.0))
            vq_of_t.append(q_of_t[-1].derivative(1))
            vvq_of_t.append(vq_of_t[-1].derivative(1))

        q_of_t = CollectionSubTrajectories(q_of_t)
        vq_of_t = CollectionSubTrajectories(vq_of_t)
        vvq_of_t = CollectionSubTrajectories(vvq_of_t)

    return q_of_t, vq_of_t, vvq_of_t


class CollectionSubTrajectories:
    """
    Given a list of sub-trajectories, when we call time t, we will call the corresponding sub-trajectory
    """
    def __init__(self, trajs):
        self.trajs = trajs

    def __call__(self, t):
        for i, traj in enumerate(self.trajs):
            if traj.T_min_ <= t <= traj.T_max_:
                return traj(t)
        raise ValueError("Time t is out of range")


# Control law
def contact_controller(sim, robot, trajs, tcurrent, viz=None):
    """
    Inputs:

    sim: the simulation object
    robot: the robot object
    trajs: a tuple of three functions representing the desired trajectory
    tcurrent: the current time
    viz: the visualisation object

    Outputs:

    None*

    (the function should update the simulation object with relevant torques)
    """

    reached = False  # For the TESTS

    q, q_dot = sim.getpybulletstate()

    if viz is not None:
        viz.display(q)

    # Step 1) Calculate the desired acceleration

    q_reference = trajs[0](tcurrent)
    q_dot_reference = trajs[1](tcurrent)
    q_dot_dot_reference = trajs[2](tcurrent)

    # For the TEST: get the corresponding position of the right EE in task space
    joint_id = robot.model.getJointId("RARM_JOINT5")
    pinocchio.forwardKinematics(robot.model, robot.data, q_reference)
    reference_p = deepcopy(robot.data.oMi[joint_id].translation)
    ### END OF CODE FOR TESTS

    # Move the robot back to the actual configuration
    pinocchio.forwardKinematics(robot.model, robot.data, q)
    pinocchio.updateFramePlacements(robot.model, robot.data)

    # for tests: compute the corresponding position of the right EE in task space
    actual_p = deepcopy(robot.data.oMi[joint_id].translation)

    q_dot_dot_desired = q_dot_dot_reference + Kp * (q_reference - q) + Kv * (q_dot_reference - q_dot)

    # Step 2) Add a term to bring the hands closer together
    # Distance-based correction
    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)
    oMlh = robot.data.oMf[left_hand_id]
    oMrh = robot.data.oMf[right_hand_id]
    lhOrh = oMlh.inverse() * oMrh
    grip_error = np.linalg.norm(pinocchio.log(lhOrh).vector)

    ### FOR THE TESTS: check if the task has been successfully reached
    # Check if I am epsilon away from CUBE_PLACEMENT_TARGET
    # find cube from robot in inverse geome
    translation_lh = oMlh.translation
    translation_rh = oMrh.translation

    # get the average of the two translations
    current_cube_pos = (translation_rh + translation_lh) / 2

    dist_to_goal = np.linalg.norm(CUBE_PLACEMENT_TARGET.translation - current_cube_pos)

    #print(CUBE_PLACEMENT_TARGET.translation, current_cube_pos)

    if dist_to_goal < 0.015:  # distance is within 1.5 cm
        reached = True
    ## End of code for the TESTS

    # find the rotation of the cube by using find_cube_from_configuration
    grip_force = Kgrip * grip_error  # Gain for bringing hands closer together

    # Define the original contact forces in the world frame
    f_c_left_hand = np.array([0, -grip_force, 0, 0, 0, 0])
    f_c_right_hand = np.array([0, -grip_force, 0, 0, 0, 0])

    f_c = np.hstack((f_c_left_hand, f_c_right_hand))

    jacobian = get_hand_jacobian(robot, q)

    # Step 3) Calculate the torques
    pinocchio.computeJointJacobians(robot.model, robot.data, q)
    M = pinocchio.crba(robot.model, robot.data, q)
    h = pinocchio.nle(robot.model, robot.data, q, q_dot)

    # extra consideration for the gravity with a factor due to weight of the cube
    g = pinocchio.computeGeneralizedGravity(robot.model, robot.data, q) * 1.2

    torques = M @ q_dot_dot_desired + h + jacobian.T @ f_c + g

    sim.step(torques)
    return reached, dist_to_goal, reference_p, actual_p, q, q_reference

def control_main(render=True):
    robot, sim, cube = setupwithpybullet()

    viz = None
    #robot, sim, cube, viz = setupwithpybulletandmeshcat("http://127.0.0.1:7003/static/")

    if not render:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")
        raise ValueError("Invalid initial or end configuration")

    start_time = time.time()
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    #setting initial configuration
    sim.setqsim(q0)

    #TODO this is just a random trajectory, you need to do this yourself
    total_time = 4.0
    trajs = maketraj(path, q0, qe, total_time)
    tcur = 0.

    movement_start_time = time.time() - start_time

    predicted_path = []
    actual_path = []
    qs_actual = []
    qs_reference = []
    qs_diff = []

    reached = False
    dist_to_goal = np.inf
    while tcur < total_time:
        reached, dist_to_goal, p_reference, actual_p, q, q_reference = rununtil(contact_controller, DT, sim,
                                                                                robot, trajs, tcur, viz=viz)
        tcur += DT
        predicted_path.append(deepcopy(p_reference))
        actual_path.append(deepcopy(actual_p))
        qs_actual.append(deepcopy(q))
        qs_reference.append(deepcopy(q_reference))
        qs_diff.append(deepcopy(q_reference - q))

    return reached, dist_to_goal, predicted_path, actual_path, qs_actual, qs_reference, qs_diff, movement_start_time


if __name__ == "__main__":
    control_main()