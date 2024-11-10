#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from numpy.linalg import pinv
from pinocchio.utils import rotate
from config import LEFT_HAND, RIGHT_HAND, EPSILON
import time
import random

# import the inverse kinematics class from the inverse_geometry.py file
from inverse_geometry import compute_grasp_pose_constrained, find_cube_from_configuration, InverseKinematicsSolver

from tools import setupwithmeshcat, setcubeplacement, collision, distanceToObstacle, getcubeplacement
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
from inverse_geometry import computeqgrasppose

"""
Helper Functions
"""

def lerp(cube1, cube2, t):

    # Interpolate positions
    position1 = cube1.translation
    position2 = cube2.translation
    position_interpolated = position1 * (1 - t) + position2 * t

    return pin.SE3(rotate('z', 0.), position_interpolated)


class Node:
    def __init__(self, parent, configuration, cube_placement):
        self.parent = parent
        self.configuration = configuration
        self.cube_placement = cube_placement


class PathFinder2:

    def __init__(self, robot_object, cube_object, viz_object=None):
        self.robot = robot_object
        self.cube = cube_object
        self.viz = viz_object
        self.visualise_process = False

        self.InverseKinematicsSolver = InverseKinematicsSolver(self.robot, self.cube)

        self.tree = []
        self.path_found = False
        self.configuration_path = []
        self.node_path = []

    def generate_random_cube_placement(self):

        while True:

            x_min = min(CUBE_PLACEMENT.translation[0], CUBE_PLACEMENT_TARGET.translation[0]) - 0.2
            x_max = max(CUBE_PLACEMENT.translation[0], CUBE_PLACEMENT_TARGET.translation[0]) + 0.2
            y_min = min(CUBE_PLACEMENT.translation[1], CUBE_PLACEMENT_TARGET.translation[1]) - 0.2
            y_max = max(CUBE_PLACEMENT.translation[1], CUBE_PLACEMENT_TARGET.translation[1]) + 0.2
            z_min = CUBE_PLACEMENT.translation[2]
            z_max = z_min + 0.4

            random_x = np.random.uniform(x_min, x_max, 1)[0]
            random_y = np.random.uniform(y_min, y_max, 1)[0]
            random_z = np.random.uniform(z_min, z_max, 1)[0]

            random_cube_placement = pin.SE3(rotate('z', 0.), np.array([random_x, random_y, random_z]))

            # get the original placement of the cube
            cube_placement = getcubeplacement(self.cube)
            # set the new placement of the cube to check for collisions
            setcubeplacement(self.robot, self.cube, random_cube_placement)
            # check for collisions
            cube_collision = pin.computeCollisions(self.cube.collision_model, self.cube.collision_data, False)
            # reset the placement of the cube
            setcubeplacement(self.robot, self.cube, cube_placement)
            if not cube_collision:
                break

        return random_cube_placement

    def LERP(self, cube1, cube2, t):
        position1 = cube1.translation
        position2 = cube2.translation
        position_interpolated = position1 * (1 - t) + position2 * t

        return pin.SE3(rotate('z', 0.), position_interpolated)

    def find_closest_node(self, cube_placement):
        closest_node = None
        minimum_distance = float('inf')

        for node in self.tree:
            distance = np.linalg.norm(node.cube_placement.translation - cube_placement.translation)
            if distance < minimum_distance:
                closest_node = node
                minimum_distance = distance

        return closest_node

    def generate_path(self):

        if not self.path_found:
            raise ValueError("Path not found")

        goal_node = self.tree[-1]

        while goal_node is not None:
            self.node_path.append(goal_node)
            self.configuration_path.append(goal_node.configuration)
            goal_node = goal_node.parent

        self.configuration_path = self.configuration_path[::-1]
        self.node_path = self.node_path[::-1]


    def NEW_CONF(self, near_node, random_cube_placement, discrete_steps, delta_cube):
        # get the current cube location
        cube_location = near_node.cube_placement
        # get the distance between the near node and the random cube placement
        distance = np.linalg.norm(cube_location.translation - random_cube_placement.translation)
        # if the distance is greater than the delta_cube, we need to interpolate
        if distance > delta_cube:
            # compute the configuration that corresponds to a path of length delta_cube
            cube_location = self.LERP(near_node.cube_placement, random_cube_placement, delta_cube / distance)
            # now distance == delta_cube

        dt = 1 / discrete_steps

        cube_current, q_current = near_node.cube_placement, near_node.configuration

        for i in range(1, discrete_steps):
            cube_target = self.LERP(near_node.cube_placement, cube_location, dt * i)

            # Constraint 1: Cube should not collide with the environment
            setcubeplacement(self.robot, self.cube, cube_target)
            cube_collision = pin.computeCollisions(self.cube.collision_model, self.cube.collision_data, False)
            if cube_collision:
                break
            setcubeplacement(self.robot, self.cube, cube_current)

            # Constraint 2: The robot should not collide with the environment
            q_next = self.InverseKinematicsSolver.inverse_kinematics(q_start=q_current, cube_target=cube_target)
            # by default inverse kinematics returns the None if it fails due to collision or other
            if q_next is None:
                break

            cube_current = cube_target
            q_current = q_next

            # now we can set the cube placement and the configuration
            setcubeplacement(self.robot, self.cube, cube_current)

            # visualise
            if self.viz and self.visualise_process:
                self.viz.display(q_current)
                time.sleep(0.05)

        next_node = Node(near_node, q_current, cube_current)
        return next_node

    def VALID_EDGE(self, near_node, cube_placement_goal, delta_cube):
        if np.linalg.norm(near_node.cube_placement.translation - cube_placement_goal.translation) < delta_cube:
            return True
        #next_node = self.NEW_CONF(near_node, cube_placement_goal, discrete_steps, delta_cube)
        #distance = np.linalg.norm(next_node.cube_placement.translation - cube_placement_goal.translation)
        #return distance < 1e-3

    def build_RRT(self, q_init, q_goal, cube_placement_init, cube_placement_goal):
        start_node = Node(None, q_init, cube_placement_init)
        goal_node = Node(None, q_goal, cube_placement_goal)

        self.tree = [start_node]

        for iteration in range(3000):
            if iteration % 8 == 0:
                random_cube_placement = cube_placement_goal
            else:
                random_cube_placement = self.generate_random_cube_placement()

            closest_node = self.find_closest_node(random_cube_placement)

            next_node = self.NEW_CONF(closest_node, random_cube_placement, 4, 0.06)

            # if the next node is too close to the current node, we skip this iteration
            if np.linalg.norm(next_node.cube_placement.translation - closest_node.cube_placement.translation) < EPSILON:
                continue
            # if the next node is too close to an obstacle, we skip this iteration
            if distanceToObstacle(self.robot, next_node.configuration) < 25 * EPSILON:
                continue

            self.tree.append(next_node)

            if self.VALID_EDGE(next_node, cube_placement_goal, 0.08):
                goal_node.parent = next_node
                self.tree.append(goal_node)
                self.path_found = True
                self.generate_path()
                break


    def display_node_path(self, dt):
        for node in self.node_path:
            setcubeplacement(self.robot, self.cube, node.cube_placement)
            if self.viz is not None:
                self.viz.display(node.configuration)
                time.sleep(dt)


class PathFinder:

    def __init__(self, robot, cube, viz):
        self.robot = robot
        self.cube = cube
        self.viz = viz


        self.tree = []
        self.node_path = [] # path of nodes (including the cube placement)
        self.path = [] # the configuration path
        self.path_found = False

    def generate_random_cube_placement(self):

        while True:
            # TODO - pick these values more deliberately
            x_min = min(CUBE_PLACEMENT.translation[0], CUBE_PLACEMENT_TARGET.translation[0]) - 0.2
            x_max = max(CUBE_PLACEMENT.translation[0], CUBE_PLACEMENT_TARGET.translation[0]) + 0.2
            y_min = min(CUBE_PLACEMENT.translation[1], CUBE_PLACEMENT_TARGET.translation[1]) - 0.2
            y_max = max(CUBE_PLACEMENT.translation[1], CUBE_PLACEMENT_TARGET.translation[1]) + 0.2
            z_min = CUBE_PLACEMENT.translation[2]
            z_max = z_min + 0.4

            random_x = np.random.uniform(x_min, x_max, 1)[0]
            random_y = np.random.uniform(y_min, y_max, 1)[0]
            random_z = np.random.uniform(z_min, z_max, 1)[0]

            random_cube_placement = pin.SE3(rotate('z', 0.), np.array([random_x, random_y, random_z]))
            setcubeplacement(self.robot, self.cube, random_cube_placement)
            collisions = pin.computeCollisions(self.cube.collision_model, self.cube.collision_data, False)
            if not collisions:
                break

        return random_cube_placement

    def find_closest_node(self, cube_placement):

        closest_node = None
        minimum_distance = float('inf')

        # get the
        for node in self.tree:
            distance = np.linalg.norm(node.cube_placement.translation - cube_placement.translation)
            if distance < minimum_distance:
                closest_node = node
                minimum_distance = distance

        return closest_node


    def build_RRT(self, q_init, q_goal, cube_placement_init, cube_placement_goal):
        start_node = Node(None, q_init, cube_placement_init)
        goal_node = Node(None, q_goal, cube_placement_goal)

        self.tree = [start_node]

        for iteration in range(3000):

            # 1. Generate a random cube placement
            if iteration % 8 == 0:
                random_cube_placement = cube_placement_goal
            else:
                random_cube_placement = self.generate_random_cube_placement()

            # 2. Find the closest node
            closest_node = self.find_closest_node(random_cube_placement)
            q_near, cube_placement_near = closest_node.configuration, closest_node.cube_placement

            # 3. Compute the next configuration
            q_next, _ = compute_grasp_pose_constrained(self.robot, closest_node.configuration, self.cube, random_cube_placement, 0.06)
            cube_next = find_cube_from_configuration(self.robot)

            step_size = np.linalg.norm(cube_placement_near.translation - cube_next.translation)

            # if the q_next is too close to the q_near, we skip this iteration
            if step_size < EPSILON:
                continue

            if distanceToObstacle(self.robot, q_next) < 25 * EPSILON:
                continue

            setcubeplacement(self.robot, self.cube, cube_next)

            # visualise
            if self.viz is not None:
                self.viz.display(q_next)
                time.sleep(0.1)

            # 5. Create a new node and add it to the tree
            new_node = Node(closest_node, q_next, cube_next)
            self.tree.append(new_node)

            # Check if we can go to the goal
            if self.VALID_EDGE(q_next, cube_next, cube_placement_goal, 0.08):
                goal_node.parent = new_node
                self.tree.append(goal_node)
                self.extract_node_path()
                self.interpolate_path()
                self.path_found = True
                break

            setcubeplacement(self.robot, self.cube, cube_next)


    def VALID_EDGE(self, q_near, cube_near, cube_goal, max_step):
        q_next, _ = computeqgrasppose(self.robot, q_near, self.cube, cube_goal)
        if np.linalg.norm(cube_near.translation - cube_goal.translation) < max_step:
            return True
        else:
            return False

    def extract_node_path(self):

        node = self.tree[-1]
        while node is not None:
            self.node_path.append(node)
            node = node.parent

        self.node_path = self.node_path[::-1]
        # now build the actual path
        self.path = [node.configuration for node in self.node_path]


    def interpolate_path(self):

        new_path = []
        # goes through all the nodes in the node path, and does inverse kinematics from one node to the next adding each waypoint configuration to the path
        for i in range(len(self.node_path) - 1):
            node1 = self.node_path[i]
            node2 = self.node_path[i + 1]

            # interpolate between the two cube placements
            for t in np.linspace(0, 1, 5):
                cube_placement = lerp(node1.cube_placement, node2.cube_placement, t)
                q, _ = computeqgrasppose(self.robot, node1.configuration, self.cube, cube_placement)
                new_path.append(q)

        self.path = new_path

    # additional helper function to display the path with the cube as well
    def display_node_path(self, dt):
        for node in self.node_path:
            setcubeplacement(self.robot, self.cube, node.cube_placement)
            if self.viz is not None:
                self.viz.display(node.configuration)
                time.sleep(dt)



# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, viz=None):

    """    # create a pathfinder object
    pathfinder = PathFinder(robot, cube, viz)

    start_time = time.time()
    print("Computing path...")
    for attempt in range(100):
        pathfinder.build_RRT(qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
        if pathfinder.path_found:
            print("Found path in: ", round(time.time() - start_time), "seconds")
            pathfinder.display_node_path(0.5)
            #print(pathfinder.path)
            return pathfinder.path

    print("Failed to find path")
    return [qinit, qgoal]
    """

    pathfinder = PathFinder2(robot, cube, viz)

    start_time = time.time()
    print("Computing path...")
    for attempt in range(10):
        pathfinder.build_RRT(qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
        if pathfinder.path_found:
            print("Found path in: ", round(time.time() - start_time), "seconds")
            pathfinder.display_node_path(0.5)
            return pathfinder.configuration_path


def displaypath(robot, path, dt, viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":


    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, None)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, viz=viz)

