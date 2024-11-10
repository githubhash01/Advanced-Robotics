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

from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, EPSILON
from tools import setcubeplacement, setupwithmeshcat, jointlimitsviolated, projecttojointlimits, distanceToObstacle
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

class InverseKinematicsSolver:

    DT = 0.1

    def __init__(self, robot_object, cube_object, viz_object=None):
        self.robot = robot_object
        self.q_current = None
        self.cube = cube_object
        self.viz = viz_object

    def find_cube_from_configuration(self):

        rh_frame_id = self.robot.model.getFrameId(RIGHT_HAND)
        oMrh = self.robot.data.oMf[rh_frame_id]

        lh_frame_id = self.robot.model.getFrameId(LEFT_HAND)
        oMlh = self.robot.data.oMf[lh_frame_id]

        # get the average of the two translations
        average_translation = (oMrh.translation + oMlh.translation) / 2

        # we assume cube is never rotated as there is no use for this
        cube_placement = pin.SE3(rotate('z', 0.), average_translation)
        return cube_placement

    def get_hand_cube_errors(self):
        """
        Gets the 6D spatial vector error for LH -> Left Hook of Cube and RH -> Right Hook of Cube
        """
        pin.framesForwardKinematics(self.robot.model, self.robot.data, self.q_current)

        # 6D error between right hand and target placement on the cube
        rh_frame_id = self.robot.model.getFrameId(RIGHT_HAND)
        oMrh = self.robot.data.oMf[rh_frame_id]
        oMcubeR = getcubeplacement(self.cube, RIGHT_HOOK)
        rhMcubeR = oMrh.inverse() * oMcubeR

        x_dot_rh = pin.log(rhMcubeR).vector

        # 6D error between LH frame and cube
        lh_frameid = self.robot.model.getFrameId(LEFT_HAND)
        oMlh = self.robot.data.oMf[lh_frameid]
        oMcubeL = getcubeplacement(self.cube, LEFT_HOOK)
        lhMcubeL = oMlh.inverse() * oMcubeL

        x_dot_lh = pin.log(lhMcubeL).vector

        # overall error used for optimisation x_dot = [x_dot_lh, x_dot_rh]
        x_dot = np.hstack((x_dot_lh, x_dot_rh))
        return x_dot

    def get_hand_jacobian(self):
        """
        Gets the jacobian, of the end effector of the robot
        """

        pin.computeJointJacobians(self.robot.model, self.robot.data, self.q_current)
        rh_frameid = self.robot.model.getFrameId(RIGHT_HAND)
        lh_frameid = self.robot.model.getFrameId(LEFT_HAND)

        jacobian_rh = pin.computeFrameJacobian(self.robot.model, self.robot.data, self.q_current, rh_frameid, pin.LOCAL)
        jacobian_lh = pin.computeFrameJacobian(self.robot.model, self.robot.data, self.q_current, lh_frameid, pin.LOCAL)

        jacobian = np.vstack((jacobian_lh, jacobian_rh))
        return jacobian

    def inverse_kinematics_quadprog_step(self):
        x_dot = self.get_hand_cube_errors().flatten()
        jacobian = self.get_hand_jacobian()

        # Compute the Hessian matrix H and vector c for the QP objective function
        H = jacobian.T @ jacobian
        H = H + (1e-6) * np.eye(H.shape[0])  # Ensures positive semi-definiteness
        c = jacobian.T @ x_dot

        # TODO - extend for joint constraints implicitly defined here

        q_dot_optimal = quadprog.solve_qp(H, c)[0]
        return q_dot_optimal

    def inverse_kinematics(self, q_start, cube_target):

        self.q_current = q_start
        setcubeplacement(self.robot, self.cube, cube_target)

        for t in range(1000):
            q_dot = self.inverse_kinematics_quadprog_step()
            # integrate the joint velocities to get the next joint position
            q_next = pin.integrate(self.robot.model, self.q_current, q_dot * self.DT)
            if jointlimitsviolated(self.robot, q_next):
                q_next = projecttojointlimits(self.robot, q_next)

            self.q_current = q_next

            # visualise the configuration
            if self.viz:
                self.viz.display(self.q_current)
                time.sleep(self.DT)

            # check for if we have reached the cube
            if norm(q_dot) < EPSILON and not collision(self.robot, q_next):
                return self.q_current

        return None


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
    x_opt = quadprog.solve_qp(H, c)[0]  # [0] extracts the solution

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

def inverse_kinematics(robot, q, cube, time_step, viz):
    cube_reached = False

    for i in range(1000):

        q, cube_reached = inverse_kinematics_quadprog_step(robot, q, cube, time_step)
        # check the distance to objects

        if viz:
            viz.display(q)
            time.sleep(0.1)

        if cube_reached and not collision(robot, q):
            return q, True

    return robot.q0, False

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

def compute_grasp_pose_constrained(robot, q_start, cube, cube_target, max_distance, viz=None):
    """
    Go

    """
    setcubeplacement(robot, cube, cube_target)

    cube_original = find_cube_from_configuration(robot)
    q_current = q_start.copy()

    for i in range(1000):

        q_next, cube_reached = inverse_kinematics_quadprog_step(robot, q_current, cube, time_step=0.1)

        # Constraint: Robot is not in collision
        if collision(robot, q_next):
            return q_current, False

        #pin.framesForwardKinematics(robot.model, robot.data, q_current)
        #pin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q_current)

        #if distanceToObstacle(robot, q_next) < EPSILON:
        #    return q_current, False

        ##
        #spin.updateGeometryPlacements(robot.model, robot.data, robot.collision_model, robot.collision_data, q_current)
        #if distanceToObstacle(robot, q_next) < EPSILON:
        #    return q_current, False
        #print(distanceToObstacle(robot, q_next))

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
            time.sleep(0.01)

        if cube_reached:
            return q_current, True

    return q_current, False

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)
    # Calculate the grasp pose, and the success flag
    #qnext, success = inverse_kinematics(robot, qcurrent, cube, time_step=0.01, viz=viz)

    ikSolver = InverseKinematicsSolver(robot, cube, viz)
    qnext = ikSolver.inverse_kinematics(qcurrent, cubetarget)
    if qnext is None:
        return qcurrent, False

    return qnext, True


if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    NOTHER_CUBE_PLACEMENT = pin.SE3(rotate('z', 0.), np.array([0.5, -0.35, 0.93]))
    q0, successinit = computeqgrasppose(robot, q, cube, NOTHER_CUBE_PLACEMENT, viz)
    print(successinit)

    #NOTHER_CUBE_PLACEMENT = pin.SE3(rotate('z', 0.), np.array([0.5, -0.4, 1.05]))
    qe, successend = computeqgrasppose(robot, q0, cube, CUBE_PLACEMENT_TARGET, viz=viz)
    print(successend)
    updatevisuals(viz, robot, cube, q0)


