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

MAX_DISTANCE = 0.2 # TODO - determine a good value for this and clean up the code
INTERPOLATION_DISTANCE = 0.05
NR_INTERPOLATIONS = int(MAX_DISTANCE // INTERPOLATION_DISTANCE) + 1
NR_INTERPOLATIONS = 1 # TODO - justify using this by leveraging enhanced quadprog

"""
Helper Functions
"""

class ConfigurationNode:
    def __init__(self, parent, configuration, cube_placement):
        self.parent = parent
        self.configuration = configuration
        self.cube_placement = cube_placement


class PathFinder:

    def __init__(self, qinit, qgoal, cubeplacementq0, cubeplacementqgoal):
        self.qinit = qinit
        self.qgoal = qgoal
        self.cubeplacementq0 = cubeplacementq0
        self.cubeplacementqgoal = cubeplacementqgoal
        self.start_node = ConfigurationNode(None, qinit, cubeplacementq0)
        self.goal_node = ConfigurationNode(None, qgoal, cubeplacementqgoal)
        self.tree = [self.start_node]
        self.path_found = False
        self.node_path = []

    def distance_cube_to_cube(self, cube_placement_a, cube_placement_b):
        return np.linalg.norm(cube_placement_a.translation - cube_placement_b.translation)

    def generate_random_cube_placement(self):
        while True:
            random_z = np.random.uniform(0.93, 1.25, 1)[0]
            random_x = np.random.uniform(0.2, 0.5, 1)[0]
            random_y = np.random.uniform(-0.5, 0.5, 1)[0]
            random_cube_placement = pin.SE3(rotate('z', 0.), np.array([random_x, random_y, random_z]))
            setcubeplacement(robot, cube, random_cube_placement)
            collisions = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
            if not collisions:
                break
        return random_cube_placement

    def get_nearest_node(self, cube_placement):
        nearest_node = None
        minimum_distance = float('inf')
        for node in self.tree:
            if self.distance_cube_to_cube(node.cube_placement, cube_placement) < minimum_distance:
                nearest_node = node
                minimum_distance = self.distance_cube_to_cube(node.cube_placement, cube_placement)
        return nearest_node

    def linear_interpolation(self, cube_placement_a, cube_placement_b , t):

        pos_a = cube_placement_a.translation
        pos_b = cube_placement_b.translation
        new_pos = pos_a + t * (pos_b - pos_a)
        interpolated_cube_placement = pin.SE3(cube_placement_a.rotation, new_pos)
        return interpolated_cube_placement

    def get_next_node(self, nearest_node, random_cube_placement, interpolations, max_distance):

        current_distance = self.distance_cube_to_cube(nearest_node.cube_placement, random_cube_placement)

        if current_distance > max_distance:
            random_cube_placement = self.linear_interpolation(nearest_node.cube_placement, random_cube_placement, max_distance / current_distance)
            current_distance = max_distance

        current_node = ConfigurationNode(nearest_node, nearest_node.configuration, nearest_node.cube_placement)
        dt = current_distance / interpolations

        for i in range(interpolations):
            cube_next = self.linear_interpolation(nearest_node.cube_placement, random_cube_placement, ((i + 1) * dt) / current_distance)
            setcubeplacement(robot, cube, cube_next)
            if pin.computeCollisions(cube.collision_model, cube.collision_data, False):
                return current_node
            q_next, success = computeqgrasppose(robot, nearest_node.configuration, cube, cube_next, viz=None)
            if not success:
                return current_node
            current_node = ConfigurationNode(nearest_node, q_next, cube_next)
        return current_node

    def valid_edge(self, nearest_node, goal_node, interpolations, max_distance):
        next_node = self.get_next_node(nearest_node, goal_node.cube_placement, interpolations, max_distance)
        cube_distance = self.distance_cube_to_cube(next_node.cube_placement, goal_node.cube_placement)
        return cube_distance <= EPSILON

    def build_RRT(self):

        for iteration in range(2000):

            random_cube_placement = self.generate_random_cube_placement()
            nearest_node = self.get_nearest_node(random_cube_placement)
            new_node = self.get_next_node(nearest_node, random_cube_placement, NR_INTERPOLATIONS, MAX_DISTANCE)
            self.tree.append(new_node)
            q_rand = new_node.configuration

            if self.valid_edge(new_node, self.goal_node, NR_INTERPOLATIONS, MAX_DISTANCE):
                q_goal, success = computeqgrasppose(robot, q_rand, cube, self.goal_node.cube_placement, viz=None)

                if success:

                    self.goal_node = ConfigurationNode(new_node, q_goal, self.goal_node.cube_placement)
                    self.tree.append(self.goal_node)
                    self.path_found = True
                    break

    def build_node_path(self):

        current_node = self.tree[-1]

        while current_node is not None:
            self.node_path.append(current_node)
            current_node = current_node.parent

        self.node_path = self.node_path[::-1]

    def find_path(self):
        self.build_RRT()

        if self.path_found:
            self.build_node_path()
            return self.node_path


        return [self.qinit, self.qgoal]



# Returns the distance between two cube placements
def distance_cube_to_cube(cube_placement_a, cube_placement_b):
    return np.linalg.norm(cube_placement_a.translation - cube_placement_b.translation)

# Generate a random cube placement not in collision
def generate_random_cube_placement():
    # Generate a random cube placement not in collision

    while True:
        # TODO -  avoid hardcdoing these values
        random_z = np.random.uniform(0.93, 1.25, 1)[0]
        random_x = np.random.uniform(0.2, 0.5, 1)[0]
        random_y = np.random.uniform(-0.5, 0.5, 1)[0]

        CUBE_PLACEMENT = pin.SE3(rotate('z', 0.), np.array([random_x, random_y, random_z]))
        setcubeplacement(robot, cube, CUBE_PLACEMENT)

        # Check cube is not in collision with obstacles or table
        collisions = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
        if not collisions:
            break

    return CUBE_PLACEMENT

# Returns the nearest node in the tree to the given cube placement
def get_nearest_node(tree, cube_placement):
    nearest_node = None
    minimum_distance = float('inf')
    for node in tree:
        if distance_cube_to_cube(node.cube_placement, cube_placement) < minimum_distance:
            nearest_node = node
            minimum_distance = distance_cube_to_cube(node.cube_placement, cube_placement)

    return nearest_node

# Returns a new cube placement that is t fraction of the way from cube_placement_a to cube_placement_b
def linear_interpolation(cube_placement_a, cube_placement_b , t):

    # Extract the translation vectors (positions) of cube_placement_a and cube_placement_b
    pos_a = cube_placement_a.translation  # Position of cube_placement_a
    pos_b = cube_placement_b.translation  # Position of cube_placement_b

    # Compute the new position for the interpolated cube placement
    new_pos = pos_a + t * (pos_b - pos_a)

    # Create a new SE3 matrix for the interpolated cube placement, keeping same orientation as cube_placement_a
    interpolated_cube_placement = pin.SE3(cube_placement_a.rotation, new_pos)

    return interpolated_cube_placement

# Returns a new node that is as close as possible to the random_cube_placement while avoiding collisions
def get_next_node(nearest_node, random_cube_placement, interpolations, max_distance):

    current_distance = distance_cube_to_cube(nearest_node.cube_placement, random_cube_placement)

    if current_distance > max_distance:
        random_cube_placement = linear_interpolation(nearest_node.cube_placement, random_cube_placement, max_distance / current_distance)
        current_distance = max_distance

    current_node = ConfigurationNode(nearest_node, nearest_node.configuration, nearest_node.cube_placement)

    dt = current_distance / interpolations

    for i in range(interpolations):

        cube_next = linear_interpolation(nearest_node.cube_placement, random_cube_placement, ((i + 1) * dt) / current_distance)
        setcubeplacement(robot, cube, cube_next)

        if pin.computeCollisions(cube.collision_model, cube.collision_data, False):
            return current_node

        q_next, success = computeqgrasppose(robot, nearest_node.configuration, cube, cube_next, viz=None)

        if not success:
            return current_node

        current_node = ConfigurationNode(nearest_node, q_next, cube_next)

    return current_node

def valid_edge(nearest_node, goal_node, interpolations, max_distance):

    next_node = get_next_node(nearest_node, goal_node.cube_placement, interpolations, max_distance)

    if abs(distance_cube_to_cube(next_node.cube_placement, goal_node.cube_placement)) > EPSILON:
        return False
    else:
        return True


# TODO - use shortcuts
def build_node_path(tree):
    node_path = []
    current_node = tree[-1]
    while current_node is not None:
        node_path.append(current_node)
        current_node = current_node.parent

    node_path = node_path[::-1]

    return node_path

def shortcut(node_path):
    for i, node in enumerate(node_path):
        for j in reversed(range(i+1,len(node_path))):
            node_2 = node_path[j]
            new_node = get_next_node(node, node_2.cube_placement, interpolations=NR_INTERPOLATIONS, max_distance=MAX_DISTANCE)
            if valid_edge(node, new_node, interpolations=NR_INTERPOLATIONS, max_distance=MAX_DISTANCE):
                node_path = node_path[:i+1] + [new_node] + node_path[j:]
                return node_path

    return node_path

# Does RRT to find a path from start_node to goal_node and a success flag path_found
def build_RRT(start_node, goal_node, max_iterations):
    tree = [start_node]
    path_found = False

    for iteration in range(max_iterations):

        random_cube_placement = generate_random_cube_placement()
        nearest_node = get_nearest_node(tree, random_cube_placement)


        new_node = get_next_node(nearest_node, random_cube_placement, interpolations=NR_INTERPOLATIONS, max_distance=MAX_DISTANCE)

        tree.append(new_node)

        q_rand = new_node.configuration

        # if the distance between target cube and current cube is less than max_distance, then we can go to the goal configuration
        if valid_edge(new_node, goal_node, interpolations=NR_INTERPOLATIONS, max_distance=MAX_DISTANCE):
            q_goal, success = computeqgrasppose(robot, q_rand, cube, goal_node.cube_placement, viz=None)
            if success:
                goal_node = ConfigurationNode(new_node, q_goal, goal_node.cube_placement)
                tree.append(goal_node)
                path_found = True
                break

    return tree, path_found

# additional helper function to display the path with the cube as well
def display_node_path(robot, node_path, dt, viz):
    for node in node_path:
        setcubeplacement(robot, cube, node.cube_placement)
        viz.display(node.configuration)
        time.sleep(dt )

"""
Code given in Lab 
"""

# returns a collision free path from qinit to qgoal under grasping constraints
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):

    path_finder = PathFinder(qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
    node_path = path_finder.find_path()

    if path_finder.path_found:
        print("Path found")
    else:
        print("Path not found")

    return node_path


def displaypath(robot,path,dt,viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat, setcubeplacement
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose
    
    robot, cube, viz = setupwithmeshcat()
    
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, None)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  None)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")

    print("Computing path")
    node_path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    #displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt TODO - change this back to displaypath
    display_node_path(robot, node_path, dt=0.5, viz=viz)
    
