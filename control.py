#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import pybullet as p

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

# PD gains set by Steve Thonneau - they work well for this problem
Kp = 300
Kv = 2 * np.sqrt(Kp)
Kgrip = 60 # Gain for gripping the cube

N_BEZIERS = None

COLLISION_STATUS = "collision"
REACHED_STATUS = "reached"

def maketraj(path, q0, q1, T, sub_traj=N_BEZIERS):
    path = [q0]*1 + path + [q1]*5

    if sub_traj is None:
        # Only one trajectory
        q_of_t = Bezier(pointlist=path, t_min=0.0, t_max=T, mult_t=1.0)  # TODO - what is mult_t
        vq_of_t = q_of_t.derivative(1)
        vvq_of_t = vq_of_t.derivative(1)
        return q_of_t, vq_of_t, vvq_of_t

    # Split the trajectory into `sub_traj` trajectories
    q_of_t = []
    vq_of_t = []
    vvq_of_t = []
    for i in range(sub_traj):
        start = int(i * len(path) / sub_traj)
        end = int((i + 1) * len(path) / sub_traj)
        t_min = i * T / sub_traj
        t_max = (i + 1) * T / sub_traj
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


def contact_controller(sim, robot, trajs, tcurrent, cube, total_time):
    """

    tau = M* desired_q_double_dot + h + J.T * f_c

    Step 1) Calculate desired q double dot

    desired_q_double_dot = reference_q_double_dot + Kp * (reference_q - q) + Kv * (reference_q_dot - q_dot)

    Step 2) Calculate torques

    tau = M* desired_q_double_dot + h + J.T * f_c

    """
    global Kgrip

    reached = False

    # Step 1) Calculate desired q double dot

    q, q_dot = sim.getpybulletstate()
    q_reference = trajs[0](tcurrent)

    ## compute reference position
    # Get the index of the joint named "RIGHT_HAND"
    joint_id = robot.model.getJointId("RARM_JOINT5")
    pinocchio.forwardKinematics(robot.model, robot.data, q_reference)
    reference_p = deepcopy(robot.data.oMi[joint_id].translation)

    pinocchio.forwardKinematics(robot.model, robot.data, q)
    pinocchio.updateFramePlacements(robot.model, robot.data)
    actual_p = deepcopy(robot.data.oMi[joint_id].translation)

    q_dot_reference = trajs[1](tcurrent)
    q_dot_dot_reference = trajs[2](tcurrent)

    q_dot_dot_desired = q_dot_dot_reference + Kp * (q_reference - q) + Kv * (q_dot_reference - q_dot)
    # Step 2) Add a term to bring the hands closer together
    # Distance-based correction
    left_hand_id = robot.model.getFrameId(LEFT_HAND)
    right_hand_id = robot.model.getFrameId(RIGHT_HAND)
    oMlh = robot.data.oMf[left_hand_id]
    oMrh = robot.data.oMf[right_hand_id]
    lhOrh = oMlh.inverse() * oMrh
    grip_error = np.linalg.norm(pinocchio.log(lhOrh).vector)

    # Check if I am epsilon away from CUBE_PLACEMENT_TARGET
    # find cube from robot in inverse geome
    #current_cube_pos = find_cube_from_configuration(robot).translation
    translation_lh = oMlh.translation
    translation_rh = oMrh.translation

    # get the average of the two translations
    current_cube_pos = (translation_rh + translation_lh) / 2

    dist_to_goal = np.linalg.norm(CUBE_PLACEMENT_TARGET.translation - current_cube_pos)

    #print(CUBE_PLACEMENT_TARGET.translation, current_cube_pos)

    if dist_to_goal < 0.015:  # distance is within 1.5 cm
        reached = True

    #if collision(robot, q):
    #    status = COLLISION_STATUS

    grip_force = Kgrip * grip_error  # Gain for bringing hands closer together (adjust as needed)
    f_c_left_hand = np.array([0, -grip_force, 0, 0, 0, 0])
    f_c_right_hand = np.array([0, -grip_force, 0, 0, 0, 0])
    f_c = np.hstack((f_c_left_hand, f_c_right_hand))

    # in the final part of the trajectory, we want to release the cube
    if abs(tcurrent - total_time) < DT:
        print("Releasing the cube")
        f_c = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

    jacobian = get_hand_jacobian(robot, q)

    # Step 3) Calculate the torques
    pinocchio.computeJointJacobians(robot.model, robot.data, q)
    M = pinocchio.crba(robot.model, robot.data, q)
    h = pinocchio.nle(robot.model, robot.data, q, q_dot)
    # extra consideration for the gravity with a factor due to weight of the cube
    g = pinocchio.computeGeneralizedGravity(robot.model, robot.data, q) * 1.2

    torques = M @ q_dot_dot_desired + h + jacobian.T @ f_c + g

    # find the contact forces
    contact_forces_calculated = jacobian.T @ f_c

    #if tcurrent < 0.2:
    #    print(np.linalg.norm(contact_forces_calculated))

    sim.step(torques)

    return reached, dist_to_goal, reference_p, actual_p, q, q_reference

def control_main(render=True):
    robot, sim, cube = setupwithpybullet()

    if not render:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")
        raise ValueError("Invalid initial or end configuration")

    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    #setting initial configuration
    sim.setqsim(q0)

    #TODO this is just a random trajectory, you need to do this yourself
    total_time = 4.
    trajs = maketraj(path, q0, qe, total_time)
    tcur = 0.

    predicted_path = []
    actual_path = []
    qs_actual = []
    qs_reference = []
    qs_diff = []

    reached = False
    dist_to_goal = np.inf
    while tcur < total_time:
        reached, dist_to_goal, p_reference, actual_p, q, q_reference = rununtil(contact_controller, DT, sim,
                                                                                robot, trajs, tcur, cube=cube,
                                                                                total_time=total_time)
        tcur += DT
        predicted_path.append(deepcopy(p_reference))
        actual_path.append(deepcopy(actual_p))
        qs_actual.append(deepcopy(q))
        qs_reference.append(deepcopy(q_reference))
        qs_diff.append(deepcopy(q_reference - q))

    return reached, dist_to_goal, predicted_path, actual_path, qs_actual, qs_reference, qs_diff


if __name__ == "__main__":
    control_main()






    
    