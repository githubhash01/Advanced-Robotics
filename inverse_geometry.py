#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv
from pinocchio.pinocchio_pywrap.rpy import rotate
import time
import quadprog

from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, EPSILON, DT
from tools import setcubeplacement, jointlimitsviolated, projecttojointlimits, collision, getcubeplacement, \
    distanceToObstacle

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

# DEPRECATED - we now use the quadprog version, for reasons discussed in the report
# Solves the inverse kinematic problem over a single time step using pseudo-inverse and nullspace projection
def inverse_kinematics_analytic_step(robot, qcurrent, cube, time_step):
    cube_reached = False
    jacobian = get_hand_jacobian(robot, qcurrent)
    jacobian_rh, jacobian_lh = jacobian[6:, :], jacobian[:6, :]

    x_dot, distance_lh, distance_rh = get_hand_cube_errors(robot, qcurrent, cube)

    """
    if norm(x_dot) < 2 * EPSILON:
        cube_reached = True
        return qcurrent, cube_reached
    """

    if distance_lh <= EPSILON and distance_rh <= EPSILON:
        #print("Both hands reached")
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

# DEPRECATED - we now use the quadprog version, for reasons discussed in the report
# Solves inverse geometry by repeatedly calling the analytic step
def inverse_kinematics_analytic(robot, q, cube, time_step, viz):
    cube_reached = False

    for i in range(1000):

        q, cube_reached = inverse_kinematics_analytic_step(robot, q, cube, time_step)

        if viz:
            viz.display(q)
            #time.sleep(0.0001)

        if cube_reached and not collision(robot, q):
            return q, True

    return robot.q0, False

"""

2) Use Optimization to compute the grasp pose

    - use a quadratic program for optimisation 
    - include constraints for joint limits and collision avoidance
    - include a cost function to minimize the distance of start pose and end pose

"""

# Finds the cube placement by inferring from a grasp pose configuration of the robot
def find_cube_from_configuration(robot):
    """
    Input: robot model

    Returns: inferred cube placement SE3
    """

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

    # TODO - consider the rotation of the cube
    cube_placement = pin.SE3(rotate('z', 0), average_translation)
    return cube_placement

# Gets the error between robot hands and a cube
def get_hand_cube_errors(robot, q, cube):
    """
    Input:

    robot: the robot model
    q: the current configuration of the robot
    cube: the cube model

    Returns:

    x_dot: the 6D error between the left and right hand and the cube hook placements
    """
    pin.framesForwardKinematics(robot.model, robot.data, q)

    # 6D error between right hand and target placement on the cube
    rh_frame_id = robot.model.getFrameId(RIGHT_HAND)
    oMrh = robot.data.oMf[rh_frame_id]
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK)
    rhMcubeR = oMrh.inverse() * oMcubeR

    distance_rh = np.linalg.norm(rhMcubeR.translation)

    x_dot_rh = pin.log(rhMcubeR).vector

    # 6D error between LH frame and cube
    lh_frameid = robot.model.getFrameId(LEFT_HAND)
    oMlh = robot.data.oMf[lh_frameid]
    oMcubeL = getcubeplacement(cube, LEFT_HOOK)
    lhMcubeL = oMlh.inverse() * oMcubeL
    distance_lh = np.linalg.norm(lhMcubeL.translation)

    x_dot_lh = pin.log(lhMcubeL).vector

    # overall error used for optimisation x_dot = [x_dot_lh, x_dot_rh]
    x_dot = np.hstack((x_dot_lh, x_dot_rh)).flatten()

    return x_dot, distance_lh, distance_rh

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

    """
    Input:

    robot: the robot model
    qcurrent: the current configuration of the robot
    cube: the cube model
    time_step: the time step for integration

    Returns:

    qnext: the next configuration of the robot

    Notes:

    Minimize     1/2 x^T G x - a^T x
    Subject to   C.T x >= b
    
    G = J.T @ J
    a = J.T @ x_dot
    C = [-I, I]
    b = [q_dot_min, - q_dot_max]

    """
    cube_reached = False

    x_dot, distance_lh, distance_rh =  get_hand_cube_errors(robot, qcurrent, cube)

    if distance_lh <= EPSILON and distance_rh <= EPSILON:
        #print("Both hands reached")
        cube_reached = True
        return qcurrent, cube_reached

    jacobian = get_hand_jacobian(robot, qcurrent)

    # Compute the Hessian matrix H and vector c for the QP objective function
    G = jacobian.T @ jacobian
    G = G + 1e-6 * np.eye(G.shape[0])  # Ensures positive semi-definiteness

    a = (jacobian.T @ x_dot).flatten()

    # Set up joint velocity constraints
    q_dot_min = (robot.model.lowerPositionLimit - qcurrent) / time_step
    q_dot_max = (robot.model.upperPositionLimit - qcurrent) / time_step

    # Construct G and h for the inequality constraints
    C = np.vstack((np.eye(robot.model.nq), -np.eye(robot.model.nq)))
    b = np.hstack((q_dot_min, -q_dot_max))

    # Solve the QP to find the next joint velocities
    # TODO - include the joint limits in the optimisation
    #q_dot = quadprog.solve_qp(G, a, C.T, b)[0]
    q_dot = quadprog.solve_qp(G, a)[0]
    # Update q current to the next position
    qnext = pin.integrate(robot.model, qcurrent, q_dot * time_step)

    # If the joint positions is not respecting the joint limits, project it back
    if jointlimitsviolated(robot, qnext):
        qnext = projecttojointlimits(robot, qnext)

    return qnext, cube_reached

# Repeatedly calls the quadprog step to solve the inverse kinematics problem
def inverse_kinematics(robot, q, cube, time_step, viz):
    """
    Input:

    robot: the robot model
    q: the current configuration of the robot
    cube: the cube model
    time_step: the time step for integration
    viz: the visualisation object

    Returns:

    qnext: the next configuration of the robot
    cube_reached: a flag indicating if the cube is reached
    """

    cube_reached = False
    for i in range(4000):

        q, cube_reached = inverse_kinematics_quadprog_step(robot, q, cube, time_step)

        if viz:
            viz.display(q)

        if cube_reached and not collision(robot, q):
            return q, True

        if cube_reached and collision(robot, q):
            return q, False

    return robot.q0, cube_reached


"""
!!! BONUS MARKS !!!

- A better option to Linear Interpolation to check if a new node can be added

Used as part of the enhanced RRT algorithm to find a collision-free path between two configurations
avoids the need to do linear interpolation between the nearest node and the random node, by checking
for constraints on the path whilst finding the grasp pose of the random node - far more efficient, 
leads to a much faster RRT algorithm. Constraints include, no collisions with the cube, no collisions 
with the robot. Additionally, there is a delta_q and a delta_cube parameter to limit the distance that 
can be travelled both in joint space and in cartesian space. Including both constraints makes the 
movements much more natural and also more reliable. 

"""
# Does IK with constraints, used for collision-free motion planning in RRT
def compute_grasp_pose_constrained(robot, q_start, cube, cube_target, delta_cube, delta_q, viz=None):
    """
    Inputs:

    robot: the robot model
    q_start: the initial configuration of the robot
    cube: the cube model
    cube_target: the target cube placement
    max_distance: the maximum distance the end effectors (and cube) can travel in cartesian space
    viz: the visualisation object

    Returns:

    q_current: the final configuration of the robot
    cube_reached: a flag indicating if the cube is reached
    """

    setcubeplacement(robot, cube, cube_target)

    cube_original = find_cube_from_configuration(robot)
    q_current = q_start.copy()

    for i in range(1000):

        q_next, cube_reached = inverse_kinematics_quadprog_step(robot, q_current, cube, time_step=0.02)

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
        if np.linalg.norm(cube_next.translation - cube_original.translation) > delta_cube:
            return q_current, False

        # Constraint: Distance travelled in joint space is less than delta_q
        if np.linalg.norm(q_next - q_start) > delta_q: # TODO - add a parameter delta_q
            return q_current, False

        q_current = q_next

        if viz:
            viz.display(q_current)

        if cube_reached:
            return q_current, True

    return q_current, False

# Computes the grasp pose of the robot
def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    qnext, success = inverse_kinematics(robot, qcurrent, cube, time_step=0.02, viz=viz)

    return qnext, success

"""
!!! ATTEMPTED EDGE CASE !!!

We considered that in some cases, the robot may not reach the cube on first attempt due to collision, 
even though there exists a valid configuration. In such cases, we randomise the configuration and
attempt to find a valid configuration that is not in collision. 

This method was a last minute addition, and while it sometimes works in inverse geometry, 
we did not have time to integrate it into RRT, but we believe it could be useful in the future 

"""
def computegrasppose_random(robot, qcurrent, cube, cubetarget, viz=None):
    setcubeplacement(robot, cube, cubetarget)
    qnext, success = inverse_kinematics(robot, qcurrent, cube, time_step=0.02, viz=viz)

    if not success:
        # attempt to find a configuration that is not in collision by randomising
        for attempt in range(100):
            print(f"Attempting random configuration {attempt}")
            # generate a random 15 x1 vector
            q_random = np.random.rand(robot.nq) * 2 * np.pi - np.pi
            # project to joint limits
            q_random = projecttojointlimits(robot, q_random)

            qnext, success = inverse_kinematics(robot, q_random, cube, time_step=0.02, viz=viz)
            if success:
                break

    return qnext, success

if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()

    total_time = 0
    q = robot.q0.copy()

    q0, successinit = computegrasppose_random(robot, q, cube, CUBE_PLACEMENT, viz=None)
    print(successinit)

    setcubeplacement(robot, cube, CUBE_PLACEMENT_TARGET)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, viz=None)

    print(successend)
    updatevisuals(viz, robot, cube, q0)
