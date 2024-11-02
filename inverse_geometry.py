#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, EPSILON
from tools import setcubeplacement, setupwithmeshcat, jointlimitsviolated, projecttojointlimits
from setup_meshcat import updatevisuals
import time

from tools import setcubeplacement

"""

Grasp Pose Problem: 

- given a robot, cube compute a valid pose where the robot is grasping the cube, if one exists
- otherwise, return the original configuration, and a failure flag

Constraints:

1) no collisions of robot with obstacles
2) robot is within joint limits
 
Two Approaches for Computing the Grasp Pose

1) Use Inverse Kinematics to compute the grasp pose

    - do the right hand first
    - find the nullspace of the right hand
    - use the nullspace to compute the left hand
    
2) Use Optimization to compute the grasp pose

    - use a quadratic program for optimisation 
    - include constraints for joint limits and collision avoidance
    - include a cost function to minimize the distance of start pose and end pose
    
"""

# Solves generic optimisation problems using quadprog
def quadprog_optimization(H, c, G=None, h=None, A=None, b=None):
    import quadprog
    """
    Input:
    
    H: the Hessian matrix for the quadratic program
    c: the linear term for the quadratic program
    G: the inequality constraint matrix
    h: the inequality constraint vector
    A: the equality constraint matrix
    b: the equality constraint vector
    
    Returns:
    
    x_opt: the optimal solution to the quadratic program
    
    Quadratic Formulation: w.r.t x: 0.5 * x^T * H * x + c^T * x s.t. Gx <= h, Ax = b
    
    """
    epsilon = 1e-6
    H = H + epsilon * np.eye(H.shape[0])  # Ensures positive semi-definiteness

    # Ensure that c, h, and b are 1D arrays as required by quadprog
    c = np.array(c).flatten()

    if G is not None and h is not None:
        G = G.T
        h = np.array(h).flatten()
    else:
        # If no inequalities are provided, provide empty arrays for G and h
        G = np.zeros((H.shape[0], 1))
        h = np.zeros(1)

    if A is not None and b is not None:
        A = np.array(A)
        b = np.array(b).flatten()
    else:
        # If no equalities are provided, provide empty arrays for A and b
        A = np.zeros((0, H.shape[0]))
        b = np.zeros(0)

    # Solve the QP problem using quadprog
    x_opt = quadprog.solve_qp(H, c)[0] # [0] extracts the solution

    return x_opt

# Gets the error between robot hands and a cube
def get_hand_cube_errors(robot, q, cube):
    """
    Input:

    robot: the robot model
    q: the current configuration of the robot
    cube: the cube we are trying to grasp

    Returns:

    x_dot: the error between the robot hands and the cube x = [x_lh, x_rh]

    """
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
    x_dot = np.hstack((x_dot_lh, x_dot_rh))
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
    x_dot = get_hand_cube_errors(robot, qcurrent, cube).flatten()
    jacobian = get_hand_jacobian(robot, qcurrent)

    # Compute the Hessian matrix H and vector c for the QP objective function
    H = jacobian.T @ jacobian
    c = jacobian.T @ x_dot  # Linear term to drive the error to zero

    # Set up joint velocity constraints
    q_dot_min = (robot.model.lowerPositionLimit - qcurrent) / time_step
    q_dot_max = (qcurrent - robot.model.upperPositionLimit) / time_step

    # Construct G and h for the inequality constraints
    G = np.vstack((np.eye(robot.model.nq), -np.eye(robot.model.nq)))
    h = np.hstack((q_dot_max, -q_dot_min))


    # Solve the QP to find the next joint velocities
    q_dot = quadprog_optimization(H, c, G, h)
    # Update q current to the next position

    qnext = pin.integrate(robot.model, qcurrent, q_dot * time_step)

    # If the joint positions is not respecting the joint limits, project it back
    if jointlimitsviolated(robot, qnext):
        qnext = projecttojointlimits(robot, qnext)

    # Check for convergence using the norm of x_dot
    if norm(x_dot) < EPSILON:
        cube_reached = True

    return qnext, cube_reached

# Solves inverse kinematics using pseudo-inverse and nullspace projection
def inverse_kinematics_analytic_step(robot, qcurrent, cube, time_step):

    cube_reached = False
    jacobian = get_hand_jacobian(robot, qcurrent)
    jacobian_rh, jacobian_lh = jacobian[6:, :], jacobian[:6, :]

    x_dot = get_hand_cube_errors(robot, qcurrent, cube)
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

    # check for convergence to target
    if norm(q_dot_rh) < EPSILON and norm(q_dot_lh) < EPSILON:
        cube_reached = True

    return qnext, cube_reached


def inverse_kinematics(robot, q, cube, time_step, method='quadprog', interpolation_check=False, viz=None):

    print("Computing grasp pose using", method)
    cube_reached = False
    # for 10 seconds
    for i in range(1000):

        if method == 'quadprog':
            q, cube_reached = inverse_kinematics_quadprog_step(robot, q, cube, time_step)

        elif method == 'analytic':
            q, cube_reached = inverse_kinematics_analytic_step(robot, q, cube, time_step)

        # we can project to fix joint limits, but we can't project to fix collisions - GAME OVER
        if interpolation_check:
            if collision(robot, q):
                return robot.q0, False

        if viz:
            viz.display(q)
            time.sleep(time_step)

        if cube_reached and not collision(robot, q):
            return q, True

    return robot.q0, False


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    # Time step for the inverse kinematics
    time_step = 0.1
    # Method: quadprog or analytic
    # method = "quadprog"
    method = "analytic"
    # Calculate the grasp pose, and the success flag
    qnext, success = inverse_kinematics(robot, qcurrent, cube, time_step, method, viz=viz)

    return qnext, success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    print(successinit)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    print(successend)
    updatevisuals(viz, robot, cube, q0)
    
    
    
