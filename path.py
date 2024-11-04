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
from inverse_geometry import compute_grasp_pose_constrained, find_cube_from_configuration

"""
Helper Functions
"""

class Node:
    def __init__(self, parent, configuration, cube_placement):
        self.parent = parent
        self.configuration = configuration
        self.cube_placement = cube_placement


def generate_random_cube_placement():

    while True:
        CUBE_PLACEMENT = pin.SE3(rotate('z', 0.), np.array([0.33, -0.3, 0.93]))
        CUBE_PLACEMENT_TARGET = pin.SE3(rotate('z', 0), np.array([0.4, 0.11, 0.93]))

        # TODO - pick these values more deliberately
        random_z = np.random.uniform(0.93, 1.25, 1)[0]
        random_x = np.random.uniform(0.2, 0.5, 1)[0]
        random_y = np.random.uniform(-0.5, 0.5, 1)[0]

        random_cube_placement = pin.SE3(rotate('z', 0.), np.array([random_x, random_y, random_z]))
        setcubeplacement(robot, cube, random_cube_placement)
        collisions = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
        if not collisions:
            break
    return random_cube_placement

def find_closest_node(tree, cube_placement):

    closest_node = None
    minimum_distance = float('inf')

    # get the
    for node in tree:
        distance = np.linalg.norm(node.cube_placement.translation - cube_placement.translation)
        if distance < minimum_distance:
            closest_node = node
            minimum_distance = distance

    return closest_node


def build_RRT(q_init, q_goal, cube_placement_init, cube_placement_goal):
    start_node = Node(None, q_init, cube_placement_init)
    goal_node = Node(None, q_goal, cube_placement_goal)

    tree = [start_node]
    path_found = False

    iteration = 0
    start_time = time.time()
    while not path_found:

        if time.time() - start_time > 5:
            print("No path found - timeout")
            break

        iteration += 1
        path_found = False

        # 1. Generate a random cube placement
        if iteration % 8 == 0:
            random_cube_placement = cube_placement_goal
        else:
            random_cube_placement = generate_random_cube_placement()

        # 2. Find the closest node
        closest_node = find_closest_node(tree, random_cube_placement)
        q_near, cube_placement_near = closest_node.configuration, closest_node.cube_placement

        # 3. Compute the next configuration
        q_next, _ = compute_grasp_pose_constrained(robot, closest_node.configuration, cube, random_cube_placement, 0.06)
        cube_next = find_cube_from_configuration(robot)

        # if the q_next is too close to the q_near, we skip this iteration
        if np.linalg.norm(cube_placement_near.translation - cube_next.translation) < EPSILON:
            #iteration -= 1 # skip this iteration
            continue

        setcubeplacement(robot, cube, cube_next)
        # visualise
        #viz.display(q_next)
        #time.sleep(0.05)

        # 5. Create a new node and add it to the tree
        new_node = Node(closest_node, q_next, cube_next)
        tree.append(new_node)

        # Check if we can go to the goal
        if VALID_EDGE_(robot, q_next, cube_next, cube_placement_goal, 0.08):
            goal_node.parent = new_node
            tree.append(goal_node)
            path_found = True
            print(f"Path found in after {time.time() - start_time} seconds")
            return tree, path_found

        setcubeplacement(robot, cube, cube_next)

    return tree, path_found

def VALID_EDGE_(robot, q_near, cube_near, cube_goal, max_step):
    q_next, _ = computeqgrasppose(robot, q_near, cube, cube_goal)
    if np.linalg.norm(cube_near.translation - cube_goal.translation) < max_step:
        return True
    else:
        return False


def rrt(q_init, q_goal, cube_placement_init, cube_placement_goal):

    start_node = Node(None, q_init, cube_placement_init)
    goal_node = Node(None, q_goal, cube_placement_goal)

    tree = [start_node]
    path_found = False

    for iteration in range(1000):

        # Every 10 iterations, we try to reach the goal
        if iteration % 10 == 0:
            cube_rand = cube_placement_goal
        else:
            cube_rand = generate_random_cube_placement()

        closest_node = find_closest_node(tree, cube_rand)
        q_near, cube_near = closest_node.configuration, closest_node.cube_placement
        q_new, cube_new = NEW_CONF(robot, q_near, cube_near, cube_rand, 4, 0.2)

        # visualise the new configuration
        setcubeplacement(robot, cube, cube_new)
        #viz.display(q_new)
        #time.sleep(0.05)

        tree.append(Node(closest_node, q_new, cube_new))

        if VALID_EDGE(robot, q_new, q_goal, cube_new, cube_placement_goal, 4, 0.2):
            print("Path found")
            tree.append(Node(closest_node, q_goal, cube_placement_goal))
            path_found = True
            break

    return tree, path_found


def lerp(cube1, cube2, t):

    # Interpolate positions
    position1 = cube1.translation
    position2 = cube2.translation
    position_interpolated = position1 * (1 - t) + position2 * t

    return pin.SE3(rotate('z', 0.), position_interpolated)

def NEW_CONF(robot, q_near, cube_near, cube_rand, discretisation_steps, delta_cube):

    q_current = q_near.copy()
    cube_current = cube_near.copy()
    cube_end = cube_rand.copy()

    # Calculate the total distance between cube_near and cube_rand
    distance = np.linalg.norm(cube_rand.translation - cube_near.translation)

    # Adjust cube_end if it is beyond delta_cube distance
    if distance > delta_cube:
        cube_end = lerp(cube_near, cube_rand, delta_cube / distance)

    # Use incremental interpolation factor for each step
    for i in range(discretisation_steps + 1):
        t = i / discretisation_steps
        cube_next = lerp(cube_near, cube_end, t)
        setcubeplacement(robot, cube, cube_next)

        # Check for collisions with the environment
        collisions = pin.computeCollisions(cube.collision_model, cube.collision_data, False)
        if collisions:
            return q_current, cube_current

        # Attempt to compute the next grasp configuration
        q_next, success = computeqgrasppose(robot, q_current, cube, cube_next)
        if not success or collision(robot, q_next):
            return q_current, cube_current

        # Update to the next valid configuration if no collision
        q_current = q_next
        cube_current = cube_next

    return q_current, cube_current


def VALID_EDGE(robot, q_near, q_goal, cube_near, cube_goal, discretisation_steps, delta_cube):

    q_next, cube_next = NEW_CONF(robot, q_near, cube_near, cube_goal, discretisation_steps, delta_cube)

    if np.linalg.norm(cube_next.translation - cube_goal.translation) < delta_cube:
        return True
    else:
        return False


def extract_path(tree):

    path = []
    node = tree[-1]
    while node is not None:
        path.append(node)
        node = node.parent

    return path[::-1]


# additional helper function to display the path with the cube as well
def display_node_path(robot, node_path, dt, viz):
    for node in node_path:
        setcubeplacement(robot, cube, node.cube_placement)
        viz.display(node.configuration)
        time.sleep(dt)


# returns a collision free path from qinit to qgoal under grasping constraints
# the path is expressed as a list of configurations
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):

    #tree, path_found = rrt(qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
    for attempt in range(5):
        tree, path_found = build_RRT(qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
        if path_found:
            path = extract_path(tree)
            return path

    return [qinit, qgoal]

def displaypath(robot, path, dt, viz):
    for q in path:
        viz.display(q)
        time.sleep(dt)


if __name__ == "__main__":
    from tools import setupwithmeshcat, setcubeplacement, collision
    from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET
    from inverse_geometry import computeqgrasppose

    robot, cube, viz = setupwithmeshcat()

    q = robot.q0.copy()
    q0, successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, None)
    qe, successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET, None)

    if not (successinit and successend):
        print("error: invalid initial or end configuration")

    path = computepath(q0, qe, CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET)

    #displaypath(robot, path, dt=0.5, viz=viz)  # you ll probably want to lower dt
    display_node_path(robot, path, dt=0.5, viz=viz)  # you ll probably want to lower dt

