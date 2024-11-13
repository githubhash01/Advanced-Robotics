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
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET, EPSILON
from inverse_geometry import computeqgrasppose, compute_grasp_pose_constrained, find_cube_from_configuration, inverse_kinematics_interpolated, inverse_kinematics
import quadprog

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

class PathFinder:

    def __init__(self, robot, cube, viz):
        self.robot = robot
        self.cube = cube
        self.viz = viz
        self.tree = []
        self.node_path = [] # path of nodes (including the cube placement)
        self.path = [] # the configuration path
        self.path_found = False

        self.kd_tree = None
        self.cube_placements = []

        # just a flag for debugging visually
        self.visualise_process = False

    def update_kd_tree(self):
        """Update KD-Tree with current positions."""
        self.kd_tree = KDTree(self.cube_placements)

    def generate_random_cube_placement(self):

        while True:
            x_min = min(CUBE_PLACEMENT.translation[0], CUBE_PLACEMENT_TARGET.translation[0]) - 0.2
            x_max = max(CUBE_PLACEMENT.translation[0], CUBE_PLACEMENT_TARGET.translation[0]) + 0.2
            y_min = min(CUBE_PLACEMENT.translation[1], CUBE_PLACEMENT_TARGET.translation[1]) - 0.2
            y_max = max(CUBE_PLACEMENT.translation[1], CUBE_PLACEMENT_TARGET.translation[1]) + 0.2


            z_min = min(CUBE_PLACEMENT.translation[2], CUBE_PLACEMENT_TARGET.translation[2])
            z_max = max(CUBE_PLACEMENT.translation[2], CUBE_PLACEMENT_TARGET.translation[2]) + 0.4

            #z_min = 0.93
            #z_max = 1.3

            random_x = np.random.uniform(x_min, x_max, 1)[0]
            random_y = np.random.uniform(y_min, y_max, 1)[0]
            random_z = np.random.uniform(z_min, z_max, 1)[0]

            random_cube_placement = pin.SE3(rotate('z', 0.), np.array([random_x, random_y, random_z]))
            setcubeplacement(self.robot, self.cube, random_cube_placement)
            collisions = pin.computeCollisions(self.cube.collision_model, self.cube.collision_data, False)
            if not collisions:
                break

        return random_cube_placement

    # Brute force method for finding the closest node
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

    # KD-Tree method for finding the closest node
    def find_closest_node_kd(self, cube_placement):
        _, index = self.kd_tree.query(cube_placement.translation)
        return self.tree[index]


    def build_RRT(self, q_init, q_goal, cube_placement_init, cube_placement_goal):

        # for RRT to work, we need to make sure that the cubes are not rotated
        #cube_placement_init.rotation = rotate('z', 0)
        #cube_placement_goal.rotation = rotate('z', 0)
        # then resolve the grasppose for q_init and q_goal
        #q_init, _ = computeqgrasppose(self.robot, q_init, self.cube, cube_placement_init)
        #q_goal, _ = computeqgrasppose(self.robot, q_goal, self.cube, cube_placement_goal)

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
            #closest_node = self.find_closest_node(random_cube_placement) # Old Method
            closest_node = self.find_closest_node_kd(random_cube_placement)
            q_near, cube_placement_near = closest_node.configuration, closest_node.cube_placement

            # 3. Compute the next configuration
            q_next, _ = compute_grasp_pose_constrained(self.robot, closest_node.configuration, self.cube, random_cube_placement, 0.08)
            cube_next = find_cube_from_configuration(self.robot)

            step_size = np.linalg.norm(cube_placement_near.translation - cube_next.translation)

            # if the q_next is too close to the q_near, we skip this iteration
            if step_size < EPSILON:
                continue

            #print(cube_next)

            if distanceToObstacle(self.robot, q_next) < 30 * EPSILON: # used to be 30
                continue

            #print(len(self.tree))

            setcubeplacement(self.robot, self.cube, cube_next)
            # visualise
            if self.viz is not None and self.visualise_process:
                self.viz.display(q_next)
                time.sleep(0.01)

            # 5. Create a new node and add it to the tree
            new_node = Node(closest_node, q_next, cube_next)
            self.tree.append(new_node)
            self.cube_placements.append(cube_next.translation)
            self.update_kd_tree()

            # Check if we can go to the goal
            if self.VALID_EDGE(q_next, cube_next, cube_placement_goal, 0.1):
                goal_node.parent = new_node
                self.tree.append(goal_node)
                self.extract_node_path()
                self.interpolate_path()
                print("Path found now!")
                #self.interpolate_path_new()
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

            # find the distance between nodes
            distance = np.linalg.norm(node1.cube_placement.translation - node2.cube_placement.translation)

            # interpolate such that each step is roughly 0.01 distance
            num_steps = int(distance / 0.005) # at least 2 steps

            # interpolate between the two cube placements
            for t in np.linspace(0, 1, num_steps):
                cube_placement = lerp(node1.cube_placement, node2.cube_placement, t)
                q, _ = computeqgrasppose(self.robot, node1.configuration, self.cube, cube_placement)
                new_path.append(q)


        # only get 100 waypoints
        if len(new_path) > 100:
            new_path = new_path[::len(new_path) // 100]

        self.path = new_path

    def interpolate_path_new(self):

        """
        Interpolates a path by doing IK between each configuration in the path and getting the value along the way
        """

        interpolated_path = []

        for i in range(len(self.node_path) - 1):
            node1 = self.node_path[i]
            node2 = self.node_path[i + 1]

            # interpolate between the two cube placements using robot, robot, q, cubetarget, cube, time_step
            setcubeplacement(self.robot, self.cube, node2.cube_placement)
            waypoints = inverse_kinematics_interpolated(self.robot, node1.configuration, node2.cube_placement, self.cube, 0.01)
            # add the waypoints to the path
            interpolated_path.extend(waypoints)

        # scale down to 100 waypoints
        if len(interpolated_path) > 100:
            interpolated_path = interpolated_path[::len(interpolated_path) // 100]

        self.path = interpolated_path

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

    # create a pathfinder object
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
    #displaypath(robot, path, 0.001, viz)
    print(len(path))