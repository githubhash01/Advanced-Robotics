#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
from inverse_geometry import get_hand_jacobian
import pinocchio
from config import RIGHT_HAND, LEFT_HAND, EPSILON
from bezier import Bezier
from inverse_geometry import find_cube_from_configuration

# PD gains set by Steve Thonneau - they work well for this problem
Kp = 500
Kv = 2 * np.sqrt(Kp)
Kgrip = 100 # Gain for gripping the cube


def maketraj(path, q0, q1, T):
    path = [q0]*5 + path + [q1]*5
    q_of_t = Bezier(pointlist=path, t_min=0.0, t_max=T, mult_t=1.0)  # TODO - what is mult_t
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)

    return q_of_t, vq_of_t, vvq_of_t

def contact_controller(sim, robot, trajs, tcurrent, viz=None):
    """

    tau = M* desired_q_double_dot + h + J.T * f_c

    Step 1) Calculate desired q double dot

    desired_q_double_dot = reference_q_double_dot + Kp * (reference_q - q) + Kv * (reference_q_dot - q_dot)

    Step 2) Calculate torques

    tau = M* desired_q_double_dot + h + J.T * f_c

    """
    global Kgrip

    # Step 1) Calculate desired q double dot

    q, q_dot = sim.getpybulletstate()

    if viz is not None:
        viz.display(q)

    pinocchio.forwardKinematics(robot.model, robot.data, q)
    pinocchio.updateFramePlacements(robot.model, robot.data)

    q_reference = trajs[0](tcurrent)
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

    # find the rotation of the cube by using find_cube_from_configuration
    grip_force = Kgrip * grip_error  # Gain for bringing hands closer together (adjust as needed)

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
    g = pinocchio.computeGeneralizedGravity(robot.model, robot.data, q) * 1

    torques = M @ q_dot_dot_desired + h + jacobian.T @ f_c + g

    sim.step(torques)

if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    viz = None
    #robot, sim, cube, viz = setupwithpybulletandmeshcat("http://127.0.0.1:7003/static/")

    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose, setcubeplacement
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    #setting initial configuration
    sim.setqsim(q0)

    #TODO this is just a random trajectory, you need to do this yourself
    total_time = 4.0
    trajs = maketraj(path, q0, qe, total_time)
    tcur = 0.
    
    while tcur < total_time:
        rununtil(contact_controller, DT, sim, robot, trajs, tcur, viz=viz)
        tcur += DT




    
    