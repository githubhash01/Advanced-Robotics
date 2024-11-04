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
from inverse_geometry import InverseKinematicsSolver

"""
Helper Functions
"""


# additional helper function to display the path with the cube as well
def display_node_path(robot, node_path, dt, viz):
    for node in node_path:
        setcubeplacement(robot, cube, node.cube_placement)
        viz.display(node.configuration)
        time.sleep(dt )

class ConfigurationNode:
    def __init__(self, parent, configuration, cube_placement):
        self.parent = parent
        self.configuration = configuration
        self.cube_placement = cube_placement


class PathFinder:

    # Static constants

    MAX_DISTANCE = 0.2 # TODO - determine a good value for this and clean up the code
    INTERPOLATION_DISTANCE = 0.05
    NR_INTERPOLATIONS = int(MAX_DISTANCE // INTERPOLATION_DISTANCE) + 1


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

    @staticmethod
    def distance_cube_to_cube(cube_placement_a, cube_placement_b):
        return np.linalg.norm(cube_placement_a.translation - cube_placement_b.translation)

    @staticmethod
    def generate_random_cube_placement():
        while True:
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

    @staticmethod
    def linear_interpolation(cube_placement_a, cube_placement_b, t):

        pos_a = cube_placement_a.translation
        pos_b = cube_placement_b.translation
        new_pos = pos_a + t * (pos_b - pos_a)
        interpolated_cube_placement = pin.SE3(cube_placement_a.rotation, new_pos)
        return interpolated_cube_placement

    def get_nearest_node(self, cube_placement):
        nearest_node = None
        minimum_distance = float('inf')
        for node in self.tree:
            if self.distance_cube_to_cube(node.cube_placement, cube_placement) < minimum_distance:
                nearest_node = node
                minimum_distance = self.distance_cube_to_cube(node.cube_placement, cube_placement)
        return nearest_node

    # TODO - refactor this to be more readable
    # This uses linear interpolation - perhaps less favorable than enhanced quadprog
    def get_next_node(self, nearest_node, random_cube_placement):

        current_distance = self.distance_cube_to_cube(nearest_node.cube_placement, random_cube_placement)

        if current_distance > PathFinder.MAX_DISTANCE:
            t = PathFinder.MAX_DISTANCE / current_distance
            random_cube_placement = self.linear_interpolation(nearest_node.cube_placement, random_cube_placement, t)
            current_distance = PathFinder.MAX_DISTANCE

        current_node = ConfigurationNode(nearest_node, nearest_node.configuration, nearest_node.cube_placement)
        dt = current_distance / PathFinder.NR_INTERPOLATIONS

        for i in range(PathFinder.NR_INTERPOLATIONS):
            t = ((i + 1) * dt) / current_distance
            cube_next = self.linear_interpolation(nearest_node.cube_placement, random_cube_placement, t)
            setcubeplacement(robot, cube, cube_next)
            if pin.computeCollisions(cube.collision_model, cube.collision_data, False):
                return current_node
            q_next, success = computeqgrasppose(robot, nearest_node.configuration, cube, cube_next, viz=None)
            if not success:
                return current_node
            current_node = ConfigurationNode(nearest_node, q_next, cube_next)

        return current_node

    # An enhanced quadprog approach to getting the next node
    def get_next_node_enhanced(self, nearest_node, random_cube_placement):
        """
        The original idea behind get_next_node using linear interpolation, was to ensure that there exists
        a collision free path between the nearest node and the random cube placement. However, this approach
        is inefficient as it requires doing inverse kinematics repeatedly. A more efficient approach is to check
        for collisions, whilst doing the inverse kinematics.

        The issue with this idea originally, was ensuring that the robot's end effector pose was such that it was
        constantly gripping the cube. However, with an enhanced numerical technique using quadprog, we can ensure the
        end effectors are always gripping the cube (within some tolerance) and check for collision on the fly.

        """

        # Clip the random cube placement to be within the maximum distance
        current_distance = self.distance_cube_to_cube(nearest_node.cube_placement, random_cube_placement)
        if current_distance > PathFinder.MAX_DISTANCE:
            t = PathFinder.MAX_DISTANCE / current_distance
            random_cube_placement = self.linear_interpolation(nearest_node.cube_placement, random_cube_placement, t)



        # Get the InverseKinematicsSolver
        ik_solver = InverseKinematicsSolver(robot, nearest_node.configuration, cube, random_cube_placement, time_step=0.2)
        q_final, success = ik_solver.inverse_kinematics(method='quadprog', maintain_grip=True, interpolated=True)

        # find the middle point between the left hand and right hand and set that as the cube placement
        left_hand = robot.framePlacement(q_final, robot.model.getFrameId(LEFT_HAND))
        right_hand = robot.framePlacement(q_final, robot.model.getFrameId(RIGHT_HAND))
        middle_point = (left_hand.translation + right_hand.translation) / 2

        final_cube_placement = pin.SE3(rotate('z', 0.), middle_point)

        return ConfigurationNode(nearest_node, q_final, final_cube_placement)



    def valid_edge(self, nearest_node, goal_node):
        next_node = self.get_next_node(nearest_node, goal_node.cube_placement)
        cube_distance = self.distance_cube_to_cube(next_node.cube_placement, goal_node.cube_placement)
        return cube_distance <= EPSILON

    def build_RRT(self):

        # TODO - add a termination condition and check this
        for iteration in range(2000):

            if random.random() < 0.2:  # Goal-biasing with 20% probability
                random_cube_placement = self.goal_node.cube_placement
            else:
                random_cube_placement = self.generate_random_cube_placement()

            #random_cube_placement = self.generate_random_cube_placement()
            nearest_node = self.get_nearest_node(random_cube_placement)

            #new_node = self.get_next_node(nearest_node, random_cube_placement)
            #self.tree.append(new_node)

            # enhanced quadprog approach
            new_node = self.get_next_node_enhanced(nearest_node, random_cube_placement)
            self.tree.append(new_node)


            q_rand = new_node.configuration

            if self.valid_edge(new_node, self.goal_node):
                q_goal, success = computeqgrasppose(robot, q_rand, cube, self.goal_node.cube_placement, viz=None)

                if success:
                    self.goal_node = ConfigurationNode(new_node, q_goal, self.goal_node.cube_placement)
                    self.tree.append(self.goal_node)
                    self.path_found = True
                    print("Path found")
                    break

    def build_node_path(self):

        current_node = self.tree[-1]

        while current_node is not None:
            self.node_path.append(current_node)
            current_node = current_node.parent

        self.node_path = reversed(self.node_path)

    def find_path(self):
        self.build_RRT()

        if self.path_found:
            self.build_node_path()
            return self.node_path


        return [self.qinit, self.qgoal]



"""
Code given in Lab 
"""

# returns a collision free path from qinit to qgoal under grasping constraints
def computepath(qinit, qgoal, cubeplacementq0, cubeplacementqgoal):

    path_finder = PathFinder(qinit, qgoal, cubeplacementq0, cubeplacementqgoal)
    node_path =  path_finder.find_path()

    # TODO - ensure we actually return a configuration path, not a node path
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
    display_node_path(robot, node_path, dt=1, viz=viz)
    
