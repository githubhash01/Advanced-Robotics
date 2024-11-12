#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 11:43:26 2023

@author: stonneau
"""

from os.path import dirname, join, abspath
import numpy as np
import pinocchio as pin #the pinocchio library
from pinocchio.utils import rotate

#These parameters can be edited
USE_MESHCAT = True # part 1 uses meshcat
USE_PYBULLET = True # the second part of the lab will use pybullet
MESHCAT_URL ="tcp://127.0.0.1:6002"
USE_PYBULLET_GUI = USE_PYBULLET and True
USE_PYBULLET_REALTIME = USE_PYBULLET and False

DT = 1e-3 #simulation tick time (s)
EPSILON = 1e-3 #almost 0

#the remaining variables should not be edited in theory
LEFT_HAND  = 'LARM_EFF'
RIGHT_HAND = 'RARM_EFF'

LEFT_HOOK = "LARM_HOOK"
RIGHT_HOOK = "RARM_HOOK"

    
#scene placements
ROBOT_PLACEMENT= pin.XYZQUATToSE3(np.array([0.,0.,0.85,0.,0.,0.,1.]))
TABLE_PLACEMENT= pin.SE3(rotate('z',-np.pi/2),np.array([0.8,0.,0.]))

"""
Testing for robustness by changing the cube and potentially, the obstacle placement

Test Cases: 

1. Standard
2. Reversed
3. Far cubes
4. Close cubes
5. Tall obstacle
6. Long obstacle 
"""

# Standard case
#TEST_CASE = "Standard"
# Distance of cube related tests:


#TEST_CASE = "Reversed"
TEST_CASE = "Distant Start"
#TEST_CASE = "Distant End"
#TEST_CASE = "Distant to Distant"

# Obstacle related tests:

#TEST_CASE = "Forward Obstacle"
#TEST_CASE = "Long Obstacle"


if TEST_CASE == "Standard":

    OBSTACLE_PLACEMENT = pin.SE3(rotate('z', 0), np.array([0.43, -0.1, 0.94]))
    CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.3, 0.93]))
    CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.4, 0.11, 0.93]))

elif TEST_CASE == "Reversed":
    OBSTACLE_PLACEMENT = pin.SE3(rotate('z', 0), np.array([0.43, -0.1, 0.94]))
    CUBE_PLACEMENT = pin.SE3(rotate('z', 0), np.array([0.4, 0.11, 0.93]))
    CUBE_PLACEMENT_TARGET = pin.SE3(rotate('z', 0.), np.array([0.33, -0.3, 0.93]))

elif TEST_CASE == "Distant Start":
    OBSTACLE_PLACEMENT = pin.SE3(rotate('z', 0), np.array([0.43, -0.1, 0.94]))
    CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.48, 0.93]))
    CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.4, 0.11, 0.93]))

elif TEST_CASE == "Distant End":
    OBSTACLE_PLACEMENT = pin.SE3(rotate('z', 0), np.array([0.43, -0.1, 0.94]))
    CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.3, 0.93]))
    CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.6, 0.2, 0.93]))

elif TEST_CASE == "Distant to Distant":
    OBSTACLE_PLACEMENT = pin.SE3(rotate('z', 0), np.array([0.43, -0.1, 0.94]))
    CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.48, 0.93]))
    CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.6, 0.2, 0.93]))

elif TEST_CASE == "Forward Obstacle":
    OBSTACLE_PLACEMENT = pin.SE3(rotate('z', 0), np.array([0.6, -0.1, 0.94]))
    CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.48, 0.93]))
    CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.6, 0.2, 0.93]))

elif TEST_CASE == "Long Obstacle":
    OBSTACLE_PLACEMENT = pin.SE3(rotate('z', 0), np.array([0.2, -0.1, 0.94]))
    CUBE_PLACEMENT = pin.SE3(rotate('z', 0.),np.array([0.33, -0.48, 0.93]))
    CUBE_PLACEMENT_TARGET= pin.SE3(rotate('z', 0),np.array([0.6, 0.2, 0.93]))

#do not edit this part unless you know what you are doing
MODELS_PATH = join(dirname(str(abspath(__file__))), "models") 
MESH_DIR = MODELS_PATH 
NEXTAGE_URDF_PATH = MODELS_PATH + '/nextagea_description/urdf/'
NEXTAGE_URDF = NEXTAGE_URDF_PATH + 'NextageaOpen.urdf'
NEXTAGE_SRDF = NEXTAGE_URDF_PATH + 'NextageAOpen.srdf'
TABLE_URDF = MODELS_PATH + '/table/table_tallerscaled.urdf'
TABLE_MESH = MODELS_PATH +  "/table/" 
OBSTACLE_URDF = MODELS_PATH + '/cubes/obstacle.urdf'
OBSTACLE_MESH = MODELS_PATH + '/cubes/'
CUBE_URDF = MODELS_PATH + '/cubes/cube_small.urdf'
CUBE_MESH = MODELS_PATH + '/cubes/'
