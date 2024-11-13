#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:44:32 2023

@author: stonneau
"""

import pinocchio as pin
import numpy as np
from pinocchio.utils import rotate
import time
from scipy.spatial import KDTree

from tools import setupwithmeshcat, setcubeplacement, distanceToObstacle
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, EPSILON, DT
from inverse_geometry import computeqgrasppose, compute_grasp_pose_constrained, find_cube_from_configuration

"""

!!! BONUS TASK - Enhanced RRT !!! 
!!! BONUS TASK - KD Tree !!!

"""


# Does linear interpolation between two cube placements
def lerp(cube1, cube2, t):

    position1 = cube1.translation
    position2 = cube2.translation
    position_interpolated = position1 * (1 - t) + position2 * t

    return pin.SE3(rotate('z', 0.), position_interpolated)


# Node class for the RRT
class Node:
    def __init__(self, parent, configuration, cube_placement):
        self.parent = parent
        self.configuration = configuration
        self.cube_placement = cube_placement

# Main class for the path finding algorithm
class PathFinder:

    def __init__(self, robot, cube, viz):
        self.robot = robot
        self.cube = cube
        self.viz = viz
        self.tree = []
        self.node_path = [] # path of nodes (including the cube placement)
        self.path = [] # the configuration path
        self.path_found = False

        """
        !!! BONUS TASK - KD Tree !!!
        KD Tree for finding the closest node
        """
        self.kd_tree = None
        self.cube_placements = []

        # just a flag for debugging the RRT visually
        self.visualise_process = False


    """
    !!! BONUS TASK - KD Tree !!!
    """
    # Updates the KD-Tree with the current cube placements
    def update_kd_tree(self):
        self.kd_tree = KDTree(self.cube_placements)

    # Generates a random cube placement within the bounds of the two cube placements with a margin for exploration
    def generate_random_cube_placement(self):
        """
        Inputs: None

        Outputs: random_cube_placement (pin.SE3) - a random cube placement
        """
        while True:
            x_min = min(CUBE_PLACEMENT.translation[0], CUBE_PLACEMENT_TARGET.translation[0]) - 0.2
            x_max = max(CUBE_PLACEMENT.translation[0], CUBE_PLACEMENT_TARGET.translation[0]) + 0.2
            y_min = min(CUBE_PLACEMENT.translation[1], CUBE_PLACEMENT_TARGET.translation[1]) - 0.2
            y_max = max(CUBE_PLACEMENT.translation[1], CUBE_PLACEMENT_TARGET.translation[1]) + 0.2


            z_min = min(CUBE_PLACEMENT.translation[2], CUBE_PLACEMENT_TARGET.translation[2])
            z_max = max(CUBE_PLACEMENT.translation[2], CUBE_PLACEMENT_TARGET.translation[2]) + 0.4

            random_x = np.random.uniform(x_min, x_max, 1)[0]
            random_y = np.random.uniform(y_min, y_max, 1)[0]
            random_z = np.random.uniform(z_min, z_max, 1)[0]

            random_cube_placement = pin.SE3(rotate('z', 0.), np.array([random_x, random_y, random_z]))
            setcubeplacement(self.robot, self.cube, random_cube_placement)
            collisions = pin.computeCollisions(self.cube.collision_model, self.cube.collision_data, False)

            # if the cube placement is not in collision, we can break and return this
            if not collisions:
                break

        return random_cube_placement

    # DEPRECATED: Brute force method for finding the closest node
    def find_closest_node(self, cube_placement):
        """
        Inputs: cube_placement (pin.SE3) - the cube placement to find the closest node to

        Returns: closest_node (Node) - the closest node to the cube placement
        """
        closest_node = None
        minimum_distance = float('inf')

        # get the
        for node in self.tree:
            distance = np.linalg.norm(node.cube_placement.translation - cube_placement.translation)
            if distance < minimum_distance:
                closest_node = node
                minimum_distance = distance

        return closest_node

    """
    !!! EXTENSION METHOD !!!
    
    KD Tree method for finding the closest node
    """
    # KD-Tree method for finding the closest node
    def find_closest_node_kd(self, cube_placement):
        """
        Inputs: cube_placement (pin.SE3) - the cube placement to find the closest node to

        Returns: closest_node (Node) - the closest node to the cube placement
        """
        _, index = self.kd_tree.query(cube_placement.translation)
        return self.tree[index]

    # Main function for building the RRT and finding the path
    def build_RRT(self, q_init, q_goal, cube_placement_init, cube_placement_goal):
        """
        Inputs:

        q_init (np.array) - the initial configuration
        q_goal (np.array) - the goal configuration
        cube_placement_init (pin.SE3) - the initial cube placement
        cube_placement_goal (pin.SE3) - the goal cube placement

        Outputs: None

        Notes:



        """

        start_node = Node(None, q_init, cube_placement_init)
        goal_node = Node(None, q_goal, cube_placement_goal)

        self.tree = [start_node]
        self.cube_placements = [cube_placement_init.translation]
        self.update_kd_tree()

        for iteration in range(3000):

            random_coefficient = (iteration // 1000) + 1
            # 1. Generate a random cube placement
            if (np.random.random() * random_coefficient) > 0.875:
                random_cube_placement = cube_placement_goal
            else:
                random_cube_placement = self.generate_random_cube_placement()

            # 2. Find the closest node
            #closest_node = self.find_closest_node(random_cube_placement) # DEPRECATED
            closest_node = self.find_closest_node_kd(random_cube_placement)
            q_near, cube_placement_near = closest_node.configuration, closest_node.cube_placement

            # 3. Compute the next configuration
            q_next, _ = compute_grasp_pose_constrained(self.robot, closest_node.configuration, self.cube, random_cube_placement, 0.08)
            cube_next = find_cube_from_configuration(self.robot)

            step_size = np.linalg.norm(cube_placement_near.translation - cube_next.translation)

            # if the q_next is too close to the q_near, we skip the iteration
            if step_size < EPSILON:
                continue

            # ensure the robot does not go too close to the obstacle, or else small errors in control can cause the robot to collide
            if distanceToObstacle(self.robot, q_next) < 30 * EPSILON: # used to be 30
                continue


            setcubeplacement(self.robot, self.cube, cube_next)

            # Visualise the process if the flag is set
            if self.viz is not None and self.visualise_process:
                self.viz.display(q_next)
                time.sleep(0.001)

            # 5. Create a new node and add it to the tree
            new_node = Node(closest_node, q_next, cube_next)
            self.tree.append(new_node)
            self.cube_placements.append(cube_next.translation)
            self.update_kd_tree()

            # Check if we can go to the goal
            if self.reachable_goal(q_next, cube_next, cube_placement_goal, 0.1):
                goal_node.parent = new_node
                self.tree.append(goal_node)
                self.extract_node_path()
                self.interpolate_path()
                self.path_found = True
                break

            setcubeplacement(self.robot, self.cube, cube_next)

    # Checks if the next node is close enough to the goal to be considered a valid edge
    def reachable_goal(self, q_near, cube_near, cube_goal, max_step):
        distance_to_goal = np.linalg.norm(cube_near.translation - cube_goal.translation)
        if distance_to_goal < max_step:
            if self.VALID_EDGE(q_near, cube_near, cube_goal):
                return True

        return False

    # Extracts the node path from the tree starting from the goal node and going back to the start node
    def extract_node_path(self):

        node = self.tree[-1]
        while node is not None:
            self.node_path.append(node)
            node = node.parent

        self.node_path = self.node_path[::-1]
        # now build the actual configuration path
        self.path = [node.configuration for node in self.node_path]

    # Fills in the path for smooth trajectory
    def interpolate_path(self):

        new_path = []

        # goes through all the nodes in the node path, and does inverse kinematics from one node to the next adding each waypoint configuration to the path
        for i in range(len(self.node_path) - 1):
            node1 = self.node_path[i]
            node2 = self.node_path[i + 1]

            # find the distance between nodes
            distance = np.linalg.norm(node1.cube_placement.translation - node2.cube_placement.translation)

            # interpolate such that each step is roughly 0.01 distance
            num_steps = int(distance / 0.01) # at least 2 steps

            # interpolate between the two cube placements
            for t in np.linspace(0, 1, num_steps):
                cube_placement = lerp(node1.cube_placement, node2.cube_placement, t)
                q, _ = computeqgrasppose(self.robot, node1.configuration, self.cube, cube_placement)
                new_path.append(q)

        # only get 100 waypoints
        if len(new_path) > 100:
            new_path = new_path[::len(new_path) // 100]

        self.path = new_path

    def VALID_EDGE(self, q1, cube1, cube2):
        # interpolate between cube1 and cube2, and try to do inverse kinematics for each step
        # if by the end you have reached cube2, then the edge is valid
        success = False
        distance = np.linalg.norm(cube1.translation - cube2.translation)
        num_steps = int(distance / 0.01)  # check every 0.01 distance
        for t in np.linspace(0, 1, num_steps):
            cube = lerp(cube1, cube2, t)
            q, success = computeqgrasppose(self.robot, q1, self.cube, cube)

        # by the end the success should be true and t should be 1
        success = success and (t == 1)
        print("Success: ", success)
        return success

    # additional helper function to display the path with the cube as well
    def display_node_path(self, dt):
        for node in self.node_path:
            setcubeplacement(self.robot, self.cube, node.cube_placement)
            if self.viz is not None:
                self.viz.display(node.configuration)
                time.sleep(dt)



# returns a collision free path from qinit to qgoal under grasping constraints
def computepath(robot, cube, qinit, qgoal, cubeplacementq0, cubeplacementqgoal, viz=None):

    pathfinder = PathFinder(robot, cube, viz)

    start_time = time.time()
    print("Computing path...")
    for attempt in range(100):
        pathfinder.build_RRT(qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
        if pathfinder.path_found:
            print("Found path in: ", round(time.time() - start_time), "seconds")
            #pathfinder.display_node_path(0.05)
            return pathfinder.path

    print("Failed to find path")
    return [qinit, qgoal]


# displays the configuration path with a delay of dt
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
        raise ValueError("Invalid initial or end configuration")

    path = computepath(robot, cube, q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, viz=viz)
    displaypath(robot, path, 0.001, viz)