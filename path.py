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
from config import LEFT_HAND, RIGHT_HAND
import time



"""
Helper Functions
"""

MAX_DISTANCE = 0.8 # TODO - determine a good value for this and clean up the code

# Class for robot grasp configuration used in RRT motion planning
class ConfigurationNode:

    def __init__(self, grasp_pose, cube_placement, parent_node = None):
        self.grasp_pose = grasp_pose
        self.parent_node = parent_node
        self.cube_placement = cube_placement

# Returns a cube placement that is not in collision
def generate_random_cube_placement():
    while True:
        # Generate a random point for the cube above the start cube position and in the vicinity of both the start and target cube positions
        # TODO - use config file to set the range of the random values
        random_z = np.random.uniform(0.93, 1.25, 1)[0]
        random_x = np.random.uniform(0.2, 0.5, 1)[0]
        random_y = np.random.uniform(-0.5, 0.5, 1)[0]

        CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([random_x, random_y, random_z]))

        # TODO - not sure when to use setcubeplacement
        setcubeplacement(robot, cube, CUBE_PLACEMENT)

        # Ensure the cube placement is not in collision
        collisions = pin.computeCollisions(cube.collision_model,cube.collision_data,False)
        if not collisions:
            break

    return CUBE_PLACEMENT

# Returns the node with the closest cube placement to the given cube placement
def get_nearest_node(node_list, cube_placement):
    min_distance = float('inf')
    nearest_configuration_node = None
    for node in node_list:
        distance = np.linalg.norm(node.cube_placement.translation - cube_placement.translation)
        if distance < min_distance:
            min_distance = distance
            nearest_configuration_node = node
    return nearest_configuration_node

# Clips the cube placement given a distance from the closest node's cube placement
# TODO - determine a good value for max_distance
def clip_cube_placement(nearest_config_node, next_cube_placement, max_distance = MAX_DISTANCE):

    nearest_cube_placement = nearest_config_node.cube_placement

    # Extract the translation vectors (positions) of cube1 and cube2
    nearest_cube_pos = nearest_cube_placement.translation  # Position of cube1
    next_cube_pos = next_cube_placement.translation  # Position of cube2

    # Compute the Euclidean distance between cube1 and cube2
    current_distance = np.linalg.norm(next_cube_pos - nearest_cube_pos)

    # If the distance is within the allowed delta, return the original mat2
    if current_distance <= max_distance:
        return next_cube_placement

    # Otherwise, adjust cube2's position to be at most delta distance from cube1
    # Compute the direction vector from cube1 to cube2
    direction = (next_cube_pos - nearest_cube_pos) / current_distance
    # Calculate the new position for cube2 at distance delta
    new_position = nearest_cube_pos + direction * max_distance
    # Create a new SE3 matrix for cube2 with the adjusted position
    next_cube_pos = pin.SE3(next_cube_pos.rotation, new_position)

    return next_cube_pos

# Does linear interpolation and returns the configuration that is as far along as valid
def interpolate_configuration(robot, q1, q2, max_distance):
    # TODO - implement this function
    pass

def valid_edge_exists(current_node, goal_node, max_distance=MAX_DISTANCE):
    # TODO - implement this function

    # Checks 2 constraints: 1) distance <= max_distance, 2) no collisions through linear interpolation

    # get the distance between cubes
    distance = np.linalg.norm(current_node.cube_placement.translation - goal_node.cube_placement.translation)

    if distance > max_distance:
        return False

    return True # TODO - implement collision check


def do_RRT(tree, max_iterations, goal_node):

    success_flag = False

    for k in range(max_iterations):

        cube_random_placement = generate_random_cube_placement()
        nearest_node = get_nearest_node(tree, cube_random_placement)

        cube_near, q_near = nearest_node.cube_placement, nearest_node.grasp_pose

        cube_new = clip_cube_placement(nearest_node, cube_random_placement)

        q_proposed, valid_pose = computeqgrasppose(robot, q_near, cube, cube_new)

        if not valid_pose:
            continue

        # TODO - use linear interpolation to get the configuration that is as far along as valid
        q_new = q_proposed

        new_node = ConfigurationNode(q_new, cube_new, nearest_node)
        tree.append(new_node)

        if valid_edge_exists(new_node, goal_node):
            # change the parent of the goal node to the new node
            goal_node.parent_node = new_node
            tree.append(goal_node)
            success_flag = True
            break

    return tree, success_flag



#returns a collision free path from qinit to qgoal under grasping constraints
#the path is expressed as a list of configurations
def computepath(qinit,qgoal,cubeplacementq0, cubeplacementqgoal):

    # Initialise the path and the tree
    start_node = ConfigurationNode(None, cubeplacementq0, qinit)
    end_node = ConfigurationNode(None, cubeplacementqgoal, qgoal)
    tree = [start_node]

    # Do RRT
    tree, success = do_RRT(tree, max_iterations=1000, goal_node=end_node)

    if not success:
        print("Failed to find a path")
        return [qinit, qgoal]

    final_path = []
    # get the final path
    final_node = tree[-1]
    while final_node is not None:
        final_path.append(final_node)
        final_node = final_node.parent_node

    final_path.reverse()
    return final_path


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
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    if not(successinit and successend):
        print ("error: invalid initial or end configuration")
    
    path = computepath(q0,qe,CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)
    
    displaypath(robot,path,dt=0.5,viz=viz) #you ll probably want to lower dt
    
