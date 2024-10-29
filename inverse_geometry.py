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

    H = np.ascontiguousarray(H)
    c = np.ascontiguousarray(c)
    G = np.ascontiguousarray(G)
    h = np.ascontiguousarray(h)
    A = np.ascontiguousarray(A)
    b = np.ascontiguousarray(b)

    # Debugging prints to check shapes
    print(f"H shape: {H.shape}")
    print(f"c shape: {c.shape}")
    print(f"G shape: {G.shape}")
    print(f"h shape: {h.shape}")
    print(f"A shape: {A.shape}")
    print(f"b shape: {b.shape}")

    # Solve the QP problem using quadprog
    x_opt = quadprog.solve_qp(H, c, G, h, A, b)[0]  # [0] extracts the solution

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
    print(f"x_dot shape in get_hand_cube_errors: {x_dot.shape}")
    return x_dot

# Solves the inverse kinematic problem over a single time step
def inverse_kinematics_step(robot, qcurrent, cube, time_step=0.1, viz=None):

    """
    Input:

    robot: the robot model
    qcurrent: the current configuration of the robot


    Returns:

    qnext: the next configuration of the robot

    """
    pin.framesForwardKinematics(robot.model, robot.data, qcurrent)
    pin.computeJointJacobians(robot.model, robot.data, qcurrent)

    # Compute the Jacobians for the left and right hand
    rh_frame_id = robot.model.getFrameId(RIGHT_HAND)
    jacobian_rh = pin.computeFrameJacobian(robot.model, robot.data, qcurrent, rh_frame_id, pin.LOCAL)

    lh_frame_id = robot.model.getFrameId(LEFT_HAND)
    jacobian_lh = pin.computeFrameJacobian(robot.model, robot.data, qcurrent, lh_frame_id, pin.LOCAL)

    # Overall Jacobian for both hands
    jacobian = np.vstack((jacobian_lh, jacobian_rh))

    # Compute the error between the current and target configuration
    x_dot = get_hand_cube_errors(robot, qcurrent, cube).flatten()

    # Compute the Hessian matrix H and vector c for the QP objective function
    H = jacobian.T @ jacobian
    c = -jacobian.T @ x_dot  # Linear term to drive the error to zero

    # Set up joint velocity constraints
    q_dot_min = (robot.model.lowerPositionLimit - qcurrent) / time_step
    q_dot_max = (robot.model.upperPositionLimit - qcurrent) / time_step
    h = np.hstack((q_dot_max, -q_dot_min))  # Constraints for velocity bounds

    # Generate G matrix for the constraints
    G = np.vstack((np.eye(robot.model.nq), -np.eye(robot.model.nq))).T

    # Solve the QP to find the next joint velocities
    q_dot = quadprog_optimization(H, c, G, h)

    # Update qcurrent to the next position
    qnext = qcurrent + q_dot * time_step

    if viz is not None:
        viz.display(qnext)
        time.sleep(time_step)

    return qnext

def inverse_kinematics_analytic_step(robot, qcurrent, cube, time_step=0.1, viz=None):

    # Run the algorithms that outputs values in robot.data
    pin.framesForwardKinematics(robot.model, robot.data, q)
    pin.computeJointJacobians(robot.model, robot.data, q)

    """ Right Hand Task """

    # Compute the Jacobians for the left and right hand
    rh_frame_id = robot.model.getFrameId(RIGHT_HAND)
    jacobian_rh = pin.computeFrameJacobian(robot.model, robot.data, qcurrent, rh_frame_id, pin.LOCAL)

    lh_frame_id = robot.model.getFrameId(LEFT_HAND)
    jacobian_lh = pin.computeFrameJacobian(robot.model, robot.data, qcurrent, lh_frame_id, pin.LOCAL)

    # Get the error between the robot hands and the cube
    x_dot = get_hand_cube_errors(robot, qcurrent, cube).flatten()
    x_dot_lh, x_dot_rh = x_dot[:3], x_dot[3:]

    vq_right = pinv(jacobian_rh) @ x_dot_rh

    # Computing the Projector Matrix
    projector_rh = np.eye(robot.nv) - pinv(jacobian_rh) @ jacobian_rh

    # Computing the Left Hand Task
    vq_left = vq_right + pinv(jacobian_lh @ projector_rh) @ (x_dot_lh - jacobian_lh @ vq_right)

    vq = vq_left + vq_right

    qnext = pin.integrate(robot.model, qcurrent, vq * time_step)

    # If the joint positions is not respecting the joint limits, project it back
    if jointlimitsviolated(robot, qnext):
        qnext = projecttojointlimits(robot, qnext)

    if viz:
        viz.display(qnext)
        time.sleep(time_step)

    return qnext



# Solves the inverse kinematics problem using the quadprog optimization until convergence
def inverse_kinematics_quadprog(robot, qcurrent, cube, cubetarget, viz=None):

    while True:
        qnext = inverse_kinematics_step(robot, qcurrent, cube, viz=viz)
        # check if the joint limits are violated
        if jointlimitsviolated(robot, qnext):
            qnext = projecttojointlimits(robot, qnext)

        # check if we have reached the target configuration
        x_dot = get_hand_cube_errors(robot, qnext, cube)
        x_lh, x_rh = x_dot[:3], x_dot[3:]
        if np.linalg.norm(x_lh) < EPSILON and np.linalg.norm(x_rh) < EPSILON:
            break

    # check if the configuration is in collision
    if collision(robot, qnext):
        return qcurrent, False

    return qnext, True


def inverse_kinematics_analytic(robot, qcurrent, cube, cubetarget, viz=None):
    setcubeplacement(robot, cube, cubetarget)

    q = qcurrent.copy()

    cube_reached = False
    for i in range(500):
        q = inverse_kinematics_analytic_step(robot, q, cube, time_step=0.01, viz=viz)
        x_dot = get_hand_cube_errors(robot, q, cubetarget)
        x_lh, x_rh = x_dot[:3], x_dot[3:]
        # check if the position of the cube is reached
        if norm(x_rh) < EPSILON and norm(x_lh) < EPSILON:
            # if the final joint position is not in collision, return success
            if not collision(robot, q):
                return q, True

    return qcurrent, False

def inverse_kinematics(robot, qcurrent, cube, cubetarget, viz=None, interpolation_check=False):
    """
    Returns a collision free configuration grasping a cube at a specific location and a success flag

    interpolation check: bool - > if set to true, every intermediate configuration is checked for collision

    """
    setcubeplacement(robot, cube, cubetarget)

    q = qcurrent.copy()
    DT = 1e-1

    cube_reached = False

    # for 60 seconds try to find
    for i in range(500):

        # Run the algorithms that outputs values in robot.data
        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        """ Right Hand Task """

        rh_frameid = robot.model.getFrameId(RIGHT_HAND)
        oMrh = robot.data.oMf[rh_frameid]
        oMcubeR = getcubeplacement(cube, RIGHT_HOOK)
        rhMcubeR = oMrh.inverse()*oMcubeR
        # 6D error between LH frame and cube
        rh_nu = pin.log(rhMcubeR).vector
        # Compute the Jacobian of the right hand
        Jrh = pin.computeFrameJacobian(robot.model,robot.data,q,rh_frameid,pin.LOCAL)
        # Control Law Using Inverse Differential Kinematics


        """ Left Hand Task """

        lh_frameid = robot.model.getFrameId(LEFT_HAND)
        oMlh = robot.data.oMf[lh_frameid]
        oMcubeL = getcubeplacement(cube, LEFT_HOOK)
        lhMcubeL = oMlh.inverse()*oMcubeL
        # 6D error between LH frame and cube
        lh_nu = pin.log(lhMcubeL).vector
        # Compute the Jacobian of the left hand
        Jlh = pin.computeFrameJacobian(robot.model,robot.data,q,lh_frameid,pin.LOCAL)

        vq_right = pinv(Jrh) @ rh_nu

        # Computing the Projector Matrix
        Prh = np.eye(robot.nv) - pinv(Jrh) @ Jrh

        # Computing the Left Hand Task
        vq_left = vq_right + pinv(Jlh @ Prh) @ (lh_nu - Jlh @ vq_right)

        vq = vq_left + vq_right

        q = pin.integrate(robot.model,q, vq * DT)

        # If the joint positions is not respecting the joint limits, project it back
        if jointlimitsviolated(robot, q):
            q = projecttojointlimits(robot, q)

        # we can project to fix joint limits, but we can't project to fix collisions - GAME OVER
        if interpolation_check:
            if collision(robot, q):
                return robot.q0, False

        if viz:
            viz.display(q)
            time.sleep(DT)

        # check if the position of the cube is reached
        if norm(rh_nu) < EPSILON and norm(lh_nu) < EPSILON:
            cube_reached = True
            break

    # if the final joint position is in collision, return failure
    if collision(robot, q):
        return robot.q0, False

    # if the final joint position does not reach the cube, return failure
    if not cube_reached:
        return robot.q0, False

    # otherwise, the joint position respects the constraints and the cube is reached
    return q, True

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    #TODO implement
    print ("TODO: implement me")

    # Option 1: Inverse Kinematics via Quadprog Optimization
    #q_current = qcurrent.copy()
    #qnext, success = inverse_kinematics_quadprog(robot, q_current, cube, cubetarget, viz)

    # Option 2: Inverse Kinematics via Analytic Solution
    #q_current = qcurrent.copy()
    #qnext, success = inverse_kinematics_analytic(robot, q_current, cube, cubetarget, viz)

    # Option 3: Inverse Kinematics via Analytic Solution
    q_current = qcurrent.copy()
    qnext, success = inverse_kinematics(robot, q_current, cube, cubetarget, viz)

    if success:
        return qnext, True

    return robot.q0, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, q0)
    
    
    
