#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv, inv, norm, svd, eig
from pinocchio.pinocchio_pywrap.rpy import rotate
import time

from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, EPSILON
from tools import setcubeplacement, jointlimitsviolated, projecttojointlimits, collision, getcubeplacement, \
    distanceToObstacle
from setup_meshcat import updatevisuals
import quadprog




"""

Grasp Pose Problem: 

- given a robot, cube compute a valid pose where the robot is grasping the cube, if one exists
- otherwise, return the original configuration, and a failure flag

Constraints:

1) no collisions of robot with obstacles
2) robot is within joint limits

"""

"""
1) Use Inverse Kinematics to compute the grasp pose

    - do the right hand first
    - find the nullspace of the right hand
    - use the nullspace to compute the left hand
"""

# DEPRECATED
# TODO - ensure analytic still works
# Solves inverse kinematics using pseudo-inverse and nullspace projection
def inverse_kinematics_analytic_step(robot, qcurrent, cube, time_step):
    cube_reached = False
    jacobian = get_hand_jacobian(robot, qcurrent)
    jacobian_rh, jacobian_lh = jacobian[6:, :], jacobian[:6, :]

    x_dot, distanceRH, distanceLH = get_hand_cube_errors(robot, qcurrent, cube)

    if distanceRH < EPSILON and distanceLH < EPSILON:
        cube_reached = True
        return qcurrent, cube_reached

    x_dot_lh, x_dot_rh = x_dot[:6], x_dot[6:]

    # Use Nullspace Projection to compute the joint velocities
    q_dot_rh = pinv(jacobian_rh) @ x_dot_rh

    # Computing the Projector Matrix
    projector_rh = np.eye(robot.nv) - pinv(jacobian_rh) @ jacobian_rh

    # Computing the Left Hand Task
    q_dot_lh = q_dot_rh + pinv(jacobian_lh @ projector_rh) @ (x_dot_lh - jacobian_lh @ q_dot_rh)
    q_dot = q_dot_lh + q_dot_rh

    # Integrate the joint velocities to get the next joint position
    qnext = pin.integrate(robot.model, qcurrent, q_dot * time_step)

    # If the joint positions is not respecting the joint limits, project it back
    if jointlimitsviolated(robot, qnext):
        qnext = projecttojointlimits(robot, qnext)

    return qnext, cube_reached

def inverse_kinematics_analytic(robot, q, cube, time_step, viz):
    cube_reached = False

    for i in range(1000):

        q, cube_reached = inverse_kinematics_analytic_step(robot, q, cube, time_step)

        if viz:
            viz.display(q)
            time.sleep(0.0001)

        if cube_reached and not collision(robot, q):
            return q, True

    return robot.q0, False

"""

2) Use Optimization to compute the grasp pose

    - use a quadratic program for optimisation 
    - include constraints for joint limits and collision avoidance
    - include a cost function to minimize the distance of start pose and end pose

"""

# Finds the cube placement from the current configuration of the robot
def find_cube_from_configuration(robot):
    rh_frame_id = robot.model.getFrameId(RIGHT_HAND)
    oMrh = robot.data.oMf[rh_frame_id]
    # get the translation of the right hand
    translation_rh= oMrh.translation

    lh_frame_id = robot.model.getFrameId(LEFT_HAND)
    oMlh = robot.data.oMf[lh_frame_id]
    # get the translation of the left hand
    translation_lh= oMlh.translation

    # get the average of the two translations
    average_translation = (translation_rh + translation_lh) / 2

    # create a new cube placement
    cube_placement = pin.SE3(rotate('z', 0.), average_translation)
    return cube_placement

# Gets the error between robot hands and a cube
def get_hand_cube_errors(robot, q, cube):

    pin.framesForwardKinematics(robot.model, robot.data, q)

    # 6D error between right hand and target placement on the cube
    rh_frame_id = robot.model.getFrameId(RIGHT_HAND)
    oMrh = robot.data.oMf[rh_frame_id]
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)
    rhMcubeR = oMrh.inverse() * oMcubeR

    x_dot_rh = pin.log(rhMcubeR).vector

    # 6D error between LH frame and cube
    lh_frameid = robot.model.getFrameId(LEFT_HAND)
    oMlh = robot.data.oMf[lh_frameid]
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)
    lhMcubeL = oMlh.inverse() * oMcubeL

    x_dot_lh = pin.log(lhMcubeL).vector

    # overall error used for optimisation x_dot = [x_dot_lh, x_dot_rh]
    x_dot = np.hstack((x_dot_lh, x_dot_rh)).flatten()

    return x_dot

# Gets the jacobian of the hand of the robot
def get_hand_jacobian(robot, q):
    """
    Input:

    robot: the robot model
    q: the current configuration of the robot

    Returns:

    jacobian (J = [J_lh, J_rh]): the jacobian for the left and right hand

    """
    pin.computeJointJacobians(robot.model, robot.data, q)
    rh_frameid = robot.model.getFrameId(RIGHT_HAND)
    lh_frameid = robot.model.getFrameId(LEFT_HAND)

    jacobian_rh = pin.computeFrameJacobian(robot.model, robot.data, q, rh_frameid, pin.LOCAL)
    jacobian_lh = pin.computeFrameJacobian(robot.model, robot.data, q, lh_frameid, pin.LOCAL)

    jacobian = np.vstack((jacobian_lh, jacobian_rh))
    return jacobian

# Solves the inverse kinematic problem over a single time step
def inverse_kinematics_quadprog_step(robot, qcurrent, cube, time_step):
    cube_reached = False

    x_dot = get_hand_cube_errors(robot, qcurrent, cube)

    """
    if distanceRH < EPSILON and distanceLH < EPSILON:
        cube_reached = True
        return qcurrent, cube_reached
    """

    if norm(x_dot) < 2 * EPSILON:
        cube_reached = True
        return qcurrent, cube_reached

    jacobian = get_hand_jacobian(robot, qcurrent)

    # Compute the Hessian matrix H and vector c for the QP objective function
    H = jacobian.T @ jacobian
    H = H + 1e-6 * np.eye(H.shape[0])  # Ensures positive semi-definiteness

    c = jacobian.T @ x_dot  # Linear term to drive the error to zero
    c = np.array(c).flatten()

    # Set up joint velocity constraints
    q_dot_min = (robot.model.lowerPositionLimit - qcurrent) / time_step
    q_dot_max = (qcurrent - robot.model.upperPositionLimit) / time_step

    # Construct G and h for the inequality constraints
    G = np.vstack((np.eye(robot.model.nq), -np.eye(robot.model.nq)))
    h = np.hstack((q_dot_max, -q_dot_min))

    # Solve the QP to find the next joint velocities
    # TODO - include the joint limits in the optimisation
    q_dot = quadprog.solve_qp(H, c)[0]

    # Update q current to the next position
    qnext = pin.integrate(robot.model, qcurrent, q_dot * time_step)

    # If the joint positions is not respecting the joint limits, project it back
    if jointlimitsviolated(robot, qnext):
        qnext = projecttojointlimits(robot, qnext)

    return qnext, cube_reached

def inverse_kinematics(robot, q, cube, time_step, viz):

    cube_reached = False
    for i in range(1000):

        q, cube_reached = inverse_kinematics_quadprog_step(robot, q, cube, time_step)

        if viz:
            viz.display(q)

        #Old Version:
        #if cube_reached and not collision(robot, q):
        #    return q, cube_reached

        #print("Distance to obstacle: ", distanceToObstacle(robot, q))
        # TODO - find out why the collision check is not working when we reverse the order and start from q0
        if cube_reached and distanceToObstacle(robot, q) > 0:
            return q, cube_reached

    #print("Failed to find a valid grasp pose")
    return robot.q0, False

def compute_grasp_pose_constrained(robot, q_start, cube, cube_target, max_distance, viz=None):

    setcubeplacement(robot, cube, cube_target)

    cube_original = find_cube_from_configuration(robot)
    q_current = q_start.copy()

    for i in range(1000):

        q_next, cube_reached = inverse_kinematics_quadprog_step(robot, q_current, cube, time_step=0.1)

        # Constraint: Robot is not in collision
        if collision(robot, q_next):
            return q_current, False

        # Constraint: Cube is not in collision - tougher one
        cube_next = find_cube_from_configuration(robot)
        setcubeplacement(robot, cube, cube_next)
        cube_in_collision = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
        if cube_in_collision:
            return q_current, False
        setcubeplacement(robot, cube, cube_target)

        # Constraint: Distance travelled is less than max_distance
        if np.linalg.norm(cube_next.translation - cube_original.translation) > max_distance:
            return q_current, False

        q_current = q_next

        if viz:
            viz.display(q_current)

        if cube_reached:
            return q_current, True

    return q_current, False

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    qnext, success = inverse_kinematics(robot, qcurrent, cube, time_step=0.02, viz=viz)

    return qnext, success


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    print(successinit)

    # set the cube to the target placement
    setcubeplacement(robot, cube, CUBE_PLACEMENT_TARGET)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz)
    print(successend)
    updatevisuals(viz, robot, cube, q0)