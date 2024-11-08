#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import numpy as np
import time
from inverse_geometry import get_hand_jacobian, get_hand_cube_errors
import pinocchio
from config import RIGHT_HAND, LEFT_HAND

from bezier import Bezier
    
# in my solution these gains were good enough for all joints but you might want to tune this.
#Kp = 300.               # proportional gain (P of PD)
#Kv = 2 * np.sqrt(Kp)   # derivative gain (D of PD)

Kh = 1


def maketraj(path, q0, q1, T):
    path = [q0] + path + [q1]
    q_of_t = Bezier(pointlist=path, t_min=0.0, t_max=T, mult_t=1.0)  # TODO - what is mult_t
    vq_of_t = q_of_t.derivative(1)
    vvq_of_t = vq_of_t.derivative(1)

    return q_of_t, vq_of_t, vvq_of_t

def controllaw(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()
    #TODO 
    torques = [0.0 for _ in sim.bulletCtrlJointsInPinOrder]
    sim.step(torques)

def my_controller(sim, robot, trajs, tcurrent, cube):
    """

    tau = M* desired_q_double_dot + h + J.T * f_c

    Step 1) Calculate desired q double dot

    desired_q_double_dot = reference_q_double_dot + Kp * (reference_q - q) + Kv * (reference_q_dot - q_dot)

    Step 2) Calculate torques

    tau = M* desired_q_double_dot + h + J.T * f_c

    """
    Kp = 2500 # crazy number - but does the job
    Kv = 2 * np.sqrt(Kp)

    # Step 1) Calculate desired q double dot

    q, q_dot = sim.getpybulletstate()

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

    lhMrh = oMlh.inverse() * oMrh
    grip_gain = 80.0  # Gain for bringing hands closer together (adjust as needed)
    grip_gain =  np.linalg.norm(pinocchio.log(lhMrh).vector) * grip_gain

    f_c_left_hand = np.array([0, -grip_gain, 0, 0, 0, 0])  # Force toward the cube
    f_c_right_hand = np.array([0, -grip_gain, 0, 0, 0, 0])  # Opposite force toward the cube
    f_c = np.hstack((f_c_left_hand, f_c_right_hand))
    jacobian = get_hand_jacobian(robot, q)

    # Step 3) Calculate the torques
    pinocchio.computeJointJacobians(robot.model, robot.data, q)
    M = pinocchio.crba(robot.model, robot.data, q)
    h = pinocchio.nle(robot.model, robot.data, q, q_dot)

    torques = M @ q_dot_dot_desired + h + jacobian.T @ f_c

    sim.step(torques)


def hashimos_controller(sim, robot, trajs, tcurrent, cube):

    """
    Control in Cartesian space:


    Step 1) Calculate desired velocity of end effector

    V_dot_desired = V_dot_reference - Kv * (V - V_reference) - Kp * (X - X_reference)

    Step 2) Use desired velocity to calculate desired acceleration in congifuration space

    q_dot_dot_desired = J^# (V_dot_desired - J_dot * q_dot)

    Step 3) Use desired acceleration to calculate torques

    tau = M * desired_q_double_dot + h

    """

    Kp = 50
    Kv = 1 * np.sqrt(Kp)

    # Step 1) Calculate end-effector position and velocity for the current state

    q, q_dot = sim.getpybulletstate()

    # Forward kinematics for the actual state (current robot configuration)
    pinocchio.forwardKinematics(robot.model, robot.data, q)
    pinocchio.updateFramePlacements(robot.model, robot.data)

    # Get the actual end-effector position and velocity
    rh_frame_id = robot.model.getFrameId(RIGHT_HAND)
    lh_frame_id = robot.model.getFrameId(LEFT_HAND)

    # Right hand
    oMrh = robot.data.oMf[rh_frame_id]
    x_rh = pinocchio.log(oMrh).vector
    v_rh = pinocchio.getFrameVelocity(robot.model, robot.data, rh_frame_id)

    # Left hand
    oMlh = robot.data.oMf[lh_frame_id]
    x_lh = pinocchio.log(oMlh).vector
    v_lh = pinocchio.getFrameVelocity(robot.model, robot.data, lh_frame_id)

    # Combine to get the current end-effector state
    x = np.hstack((x_lh, x_rh))
    v = np.hstack((v_lh, v_rh))

    # Compute the Jacobian for the actual state
    jacobian_rh = pinocchio.computeFrameJacobian(robot.model, robot.data, q, rh_frame_id, pinocchio.LOCAL)
    jacobian_lh = pinocchio.computeFrameJacobian(robot.model, robot.data, q, lh_frame_id, pinocchio.LOCAL)
    jacobian = np.vstack((jacobian_lh, jacobian_rh))

    # Compute the Jacobian derivative for the actual state
    jacobian_dot_rh = pinocchio.getFrameJacobianTimeVariation(robot.model, robot.data, rh_frame_id, pinocchio.LOCAL)
    jacobian_dot_lh = pinocchio.getFrameJacobianTimeVariation(robot.model, robot.data, lh_frame_id, pinocchio.LOCAL)
    jacobian_dot = np.vstack((jacobian_dot_lh, jacobian_dot_rh))

    # Step 2) Calculate reference end-effector position and velocity

    # Get the reference joint configuration, velocity, and acceleration
    q_reference, q_dot_reference, q_dot_dot_reference = trajs[0](tcurrent), trajs[1](tcurrent), trajs[2](tcurrent)

    # Forward kinematics for the reference state
    pinocchio.forwardKinematics(robot.model, robot.data, q_reference)
    pinocchio.updateFramePlacements(robot.model, robot.data)

    # Get the reference end-effector position and velocity
    oMrh_ref = robot.data.oMf[rh_frame_id]
    x_rh_ref = pinocchio.log(oMrh_ref).vector

    v_rh_ref = pinocchio.getFrameVelocity(robot.model, robot.data, rh_frame_id)

    oMlh_ref = robot.data.oMf[lh_frame_id]
    x_lh_ref = pinocchio.log(oMlh_ref).vector

    v_lh_ref = pinocchio.getFrameVelocity(robot.model, robot.data, lh_frame_id)

    # Combine to get the reference end-effector state
    x_ref = np.hstack((x_lh_ref, x_rh_ref))
    v_ref = np.hstack((v_lh_ref, v_rh_ref))

    # Step 3: Calculate the reference Cartesian acceleration (v_dot_reference)

    jacobian_rh_reference = pinocchio.computeFrameJacobian(robot.model, robot.data, q_reference, rh_frame_id, pinocchio.LOCAL)
    jacobian_lh_reference = pinocchio.computeFrameJacobian(robot.model, robot.data, q_reference, lh_frame_id, pinocchio.LOCAL)

    jacobian_reference = np.vstack((jacobian_lh_reference, jacobian_rh_reference))

    # Compute the time derivative of the Jacobian for the left hand
    jacobian_dot_lh_reference = pinocchio.getFrameJacobianTimeVariation(robot.model, robot.data, lh_frame_id,
                                                                        pinocchio.LOCAL)

    # Compute the time derivative of the Jacobian for the right hand
    jacobian_dot_rh_reference = pinocchio.getFrameJacobianTimeVariation(robot.model, robot.data, rh_frame_id,
                                                                        pinocchio.LOCAL)

    # Stack the Jacobian time derivatives for both hands
    jacobian_dot_reference = np.vstack((jacobian_dot_lh_reference, jacobian_dot_rh_reference))

    V_dot_reference = jacobian_reference @ q_dot_dot_reference + jacobian_dot_reference @ q_dot_reference

    # Step 4: Calculate the desired acceleration in configuration space (q_dot_dot_desired)

    V_dot_desired = V_dot_reference - Kv * (v - v_ref) - Kp * (x - x_ref)

    # Step 5: Calculate the desired q_dot_dot

    q_dot_dot_desired = np.linalg.pinv(jacobian) @ (V_dot_desired - (jacobian_dot @ q_dot))

    # Step 6: Calculate the torques

    pinocchio.computeJointJacobians(robot.model, robot.data, q)
    M = pinocchio.crba(robot.model, robot.data, q)
    h = pinocchio.nle(robot.model, robot.data, q, q_dot)

    torques = M @ q_dot_dot_desired + h

    sim.step(torques)



def hashim_controller(sim, robot, trajs, tcurrent, cube):
    """

    tau = M* desired_q_double_dot + h + J.T * f_c

    Step 1) Calculate desired q double dot

    desired_q_double_dot = reference_q_double_dot + Kp * (reference_q - q) + Kv * (reference_q_dot - q_dot)

    Step 2) Calculate torques

    tau = M* desired_q_double_dot + h + J.T * f_c

    """
    Kp = 2500 # crazy number - but does the job
    Kv = 1 * np.sqrt(Kp)

    # Step 1) Calculate desired q double dot

    q, q_dot = sim.getpybulletstate()

    q_reference = trajs[0](tcurrent)
    q_dot_reference = trajs[1](tcurrent)
    q_dot_dot_reference = trajs[2](tcurrent)

    q_dot_dot_desired = q_dot_dot_reference + Kp * (q_reference - q) + Kv * (q_dot_reference - q_dot)

    # Step 2) Calculate torques

    pinocchio.computeJointJacobians(robot.model, robot.data, q)
    M = pinocchio.crba(robot.model, robot.data, q)
    h = pinocchio.nle(robot.model, robot.data, q, q_dot)

    # Step 3) Add the contact forces
    J = get_hand_jacobian(robot, q)
    gripping_force_magnitude = 10
    f_left = np.array([0, 0, gripping_force_magnitude, 0, 0, 0])  # Force toward the cube
    f_right = np.array([0, 0, -gripping_force_magnitude, 0, 0, 0])  # Opposite force toward the cube
    f_c = np.hstack((f_left, f_right))

    torques = M @ q_dot_dot_desired + h + J.T @ f_c

    sim.step(torques)


def contact_controller(sim, robot, trajs, tcurrent, cube):
    q, vq = sim.getpybulletstate()
    q_d = trajs[0](tcurrent)
    v_d = trajs[1](tcurrent)
    a_d = trajs[2](tcurrent)

    # Current non-linear effects using current state
    pinocchio.computeJointJacobians(robot.model, robot.data, q)
    M = pinocchio.crba(robot.model, robot.data, q)
    J = get_hand_jacobian(robot, q)
    h = pinocchio.nle(robot.model, robot.data, q, vq)

    # Set desired contact forces
    f_left = np.array([0, -10, 9.81])
    tau_left = np.array([0, 0, 0])
    f_c_left = np.hstack((f_left, tau_left))

    f_right = np.array([0, -10, 9.81])
    tau_right = np.array([0, 0, 0])
    f_c_right = np.hstack((f_right, tau_right))

    f_c = np.hstack((f_c_left, f_c_right))

    # PD-based trajectory tracking torques
    Kp, Kv = 250, 1  # Example gains; tune as needed
    tau_pd = Kp * (q_d - q) + Kv * (v_d - vq)

    # Combined control law
    torques = M @ a_d + J.T @ f_c + h + tau_pd
    sim.step(torques)





if __name__ == "__main__":
        
    from tools import setupwithpybullet, setupwithpybulletandmeshcat, rununtil
    from config import DT
    
    robot, sim, cube = setupwithpybullet()
    
    
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET    
    from inverse_geometry import computeqgrasppose, setcubeplacement
    from path import computepath
    
    q0,successinit = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, robot.q0, cube, CUBE_PLACEMENT_TARGET,  None)
    path = computepath(robot, cube, q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    # set the cube placement far away

    #setting initial configuration
    sim.setqsim(q0)

    #TODO this is just a random trajectory, you need to do this yourself
    total_time = 4
    trajs = maketraj(path, q0, qe, total_time)

    tcur = 0.
    
    while tcur < total_time:
        rununtil(my_controller, DT, sim, robot, trajs, tcur, cube)
        tcur += DT




    
    