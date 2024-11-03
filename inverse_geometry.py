#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from pinocchio.pinocchio_pywrap.rpy import rotate

from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, EPSILON, \
    ANOTHER_CUBE_PLACEMENT
from tools import setcubeplacement, setupwithmeshcat, jointlimitsviolated, projecttojointlimits
from setup_meshcat import updatevisuals
import time

from tools import setcubeplacement
import quadprog

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
def quadprog_optimization(H, c, G = None, h = None, A = None, b = None):

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
    x_opt = quadprog.solve_qp(H, c, G, h)[0] # [0] extracts the solution

    return x_opt

# A class for solving the inverse kinematics problem
class InverseKinematicsSolver:

    def __init__(self, robot, q_current, cube, cube_target, time_step, viz = None):
        self.robot = robot
        self.q_current = q_current
        self.cube = cube
        self.cube_target = cube_target
        self.viz = viz
        self.time_step = time_step
        self.method = 'quadprog' # 'quadprog' or 'analytic'

    def get_hand_cube_errors(self, q_current):
        """
        Input:

        robot: the robot model
        q: the current configuration of the robot
        cube: the cube object

        Returns:

        x_dot: the error between the robot hands and the cube

        """
        pin.framesForwardKinematics(self.robot.model, self.robot.data, q_current)

        # 6D error between right hand and target placement on the cube
        rh_frame_id = self.robot.model.getFrameId(RIGHT_HAND)
        oMrh = self.robot.data.oMf[rh_frame_id]
        oMcubeR = getcubeplacement(self.cube, RIGHT_HOOK)
        rhMcubeR = oMrh.inverse() * oMcubeR
        x_dot_rh = pin.log(rhMcubeR).vector

        # 6D error between right hand and target placement on the cube
        lh_frameid = self.robot.model.getFrameId(LEFT_HAND)
        oMlh = self.robot.data.oMf[lh_frameid]
        oMcubeL = getcubeplacement(self.cube, LEFT_HOOK)
        lhMcubeL = oMlh.inverse() * oMcubeL
        x_dot_lh = pin.log(lhMcubeL).vector
        x_dot = np.hstack((x_dot_lh, x_dot_rh))
        return x_dot

    def get_hand_jacobian(self, q_current):
        """
        Input:

        robot: the robot model
        q: the current configuration of the robot

        Returns:

        jacobian (J = [J_lh, J_rh]): the jacobian for the left and right hand

        """
        pin.computeJointJacobians(self.robot.model, self.robot.data, q_current)
        rh_frameid = self.robot.model.getFrameId(RIGHT_HAND)
        lh_frameid = self.robot.model.getFrameId(LEFT_HAND)
        jacobian_rh = pin.computeFrameJacobian(self.robot.model, self.robot.data, q_current, rh_frameid, pin.LOCAL)
        jacobian_lh = pin.computeFrameJacobian(self.robot.model, self.robot.data, q_current, lh_frameid, pin.LOCAL)
        jacobian = np.vstack((jacobian_lh, jacobian_rh))
        return jacobian

    def inverse_kinematics_quadprog_step(self, q_current):
        """
        Input:

        robot: the robot model
        q_current: the current configuration of the robot
        cube: the cube object
        time_step: the time step for integration

        Returns:

        q_next: the next configuration of the robot
        cube_reached: a flag indicating whether the cube has been reached

        """

        cube_reached = False
        x_dot = self.get_hand_cube_errors(q_current).flatten()
        jacobian = self.get_hand_jacobian(q_current)

        # Compute the Hessian matrix H and vector c for the QP objective function
        H = jacobian.T @ jacobian
        c = jacobian.T @ x_dot

        # Compute the joint velocity constraints
        q_dot_min = (self.robot.model.lowerPositionLimit - q_current) / self.time_step
        q_dot_max = (q_current - self.robot.model.upperPositionLimit) / self.time_step
        G = np.vstack((np.eye(self.robot.model.nq), np.eye(self.robot.model.nq)))
        h = np.hstack((q_dot_max, q_dot_min))

        # Get the next joint velocities using quadprog
        q_dot = quadprog_optimization(H, c, G, h)
        q_next = pin.integrate(self.robot.model, q_current, q_dot * self.time_step)

        if jointlimitsviolated(self.robot, q_next):
            q_next = projecttojointlimits(self.robot, q_next)

        if norm(x_dot) < EPSILON:
            cube_reached = True
        return q_next, cube_reached

    def inverse_kinematics_analytic_step(self, q_current):
        """
        Input:

        robot: the robot model
        q_current: the current configuration of the robot
        cube: the cube object
        time_step: the time step for integration

        Returns:

        q_next: the next configuration of the robot
        cube_reached: a flag indicating whether the cube has been reached

        """

        cube_reached = False

        jacobian = self.get_hand_jacobian(q_current)
        jacobian_rh, jacobian_lh = jacobian[6:, :], jacobian[:6, :]
        x_dot = self.get_hand_cube_errors(q_current)
        x_dot_lh, x_dot_rh = x_dot[:6], x_dot[6:]

        # Use Nullspace Projection to compute the joint velocities
        q_dot_rh = pinv(jacobian_rh) @ x_dot_rh

        # Computing the Projector Matrix
        projector_rh = np.eye(self.robot.nv) - pinv(jacobian_rh) @ jacobian_rh

        # Computing the Left Hand Task
        q_dot_lh = q_dot_rh + pinv(jacobian_lh @ projector_rh) @ (x_dot_lh - jacobian_lh @ q_dot_rh)
        q_dot = q_dot_lh + q_dot_rh

        # Integrate the joint velocities to get the next joint position
        q_next = pin.integrate(self.robot.model, q_current, q_dot * self.time_step)
        if jointlimitsviolated(self.robot, q_next):
            q_next = projecttojointlimits(self.robot, q_next)

        if norm(q_dot_rh) < EPSILON and norm(q_dot_lh) < EPSILON:
            cube_reached = True

        return q_next, cube_reached

    def inverse_kinematics(self):
        """
        Input:

        robot: the robot model
        q: the current configuration of the robot
        cube: the cube object
        time_step: the time step for integration
        method: the method to use for solving the inverse kinematics problem
        viz: the visualizer object

        Returns:

        q: the next configuration of the robot
        cube_reached: a flag indicating whether the cube has been reached

        """

        cube_reached = False

        for i in range(1000):
            if self.method == 'quadprog':
                self.q_current, cube_reached = self.inverse_kinematics_quadprog_step(self.q_current)
            elif self.method == 'analytic':
                self.q_current, cube_reached = self.inverse_kinematics_analytic_step(self.q_current)
            if self.viz:
                self.viz.display(self.q_current)
                time.sleep(self.time_step)
            if cube_reached and not collision(self.robot, self.q_current):
                return self.q_current, True

        return self.robot.q0, False



"""
Code Given in the Lab
"""

# Return a collision free configuration grasping a cube at a specific location and a success flag
def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    setcubeplacement(robot, cube, cubetarget)
    solver = InverseKinematicsSolver(robot, qcurrent, cube, cubetarget, time_step=0.1,  viz=viz)
    qnext, success = solver.inverse_kinematics()
    return qnext, success
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()

    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    print(successinit)

    qe, successend = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET,  viz)
    print(successend)
    updatevisuals(viz, robot, cube, q0)
    
    
    
