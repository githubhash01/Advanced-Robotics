# Advanced Robotics Software Lab

To find out more about this capstone project - please read the report at https://drive.google.com/file/d/1aNYC6ZS18eOJSxQXAYNwjjc-i1WPBy87/view?usp=sharing

## Overview: 

This project explores how a dual-arm humanoid robot can grasp and place a cube at a desired location using a combination of inverse geometry and motion planning. Utilizing a modified RRT algorithm, trajectory generation, and optimization with Bézier curves, the robot navigates a static scene with a table, an obstacle, and a cube. The method begins with Inverse Geometry to determine joint configurations for initial and target positions, followed by finding a collision-free path. The trajectory includes joint positions, velocities, and accelerations at each time-step, with a controller computing the necessary torques. Extensive testing demonstrated robust performance, achieving a 100% task completion rate for randomized object placements, with trajectory calculations significantly optimized to 2 seconds.
    
https://github.com/user-attachments/assets/ee9dd42f-7ddf-4ea3-b969-0853cbef060a

