# TODO: reached if q is within epsilon away from cube. assume that if it collides eveyrthing is messed up.
## TODO: plto path for each forward kinematics
# TODO: change gains over a range
# TODO: with start position and end postioin, run 10 times and how many times correctly.
import numpy as np
from matplotlib import pyplot as plt

import bezier
import importlib
import inverse_geometry
import path
import setup_meshcat
import setup_pybullet
import time

import config
import control
from config import TEST_CASES
# test 1: Get the success rate of moving the cube from the start to the end position.
from control import control_main


def plot_graphs(predicted_path, actual_path, qs_actual, qs_reference, qs_diff, test_name):
    # Plot the predicted and actual pathh in 3d: three graphs, predicted, actual, and both on the same
    predicted_path = np.array(predicted_path)
    actual_path = np.array(actual_path)

    # Plot traj
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(actual_path[:, 0], actual_path[:, 1], actual_path[:, 2], label="Actual path")
    ax.set_title("Reference and actual path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot(predicted_path[:, 0], predicted_path[:, 1], predicted_path[:, 2], label="Reference path")
    ax.legend()
    plt.savefig(f"images/{test_name}_path_visual.pdf")

    # Plot the joint angles, one line for each joint for the actual and reference on the same plot,
    # with the reference in dotted and the actual in solid
    qs_actual = np.array(qs_actual)
    qs_reference = np.array(qs_reference)
    fig, axs = plt.subplots(1, 1)
    for i in range(qs_actual.shape[1]):
        color = next(axs._get_lines.prop_cycler)['color']
        axs.plot(qs_actual[:, i], label=f"Joint {i} actual", color=color)
        axs.plot(qs_reference[:, i], label=f"Joint {i} reference", linestyle="--", color=color)
    axs.set_title("Actual and reference joint angles")
    axs.set_xlabel("Time")
    axs.set_ylabel("Joint angle")
    plt.savefig(f"images/{test_name}_joint_angles.pdf")

    # Plot the difference between the actual and reference joint angles over time
    qs_diff = np.array(qs_diff)
    fig, axs = plt.subplots(1, 1)
    for i in range(qs_diff.shape[1]):
        axs.plot(qs_diff[:, i], label=f"Joint {i}")
    axs.set_title("Difference between actual and reference joint angles")
    axs.set_xlabel("Time")
    axs.set_ylabel("Difference")
    plt.savefig(f"images/{test_name}_joint_angles_diff.pdf")

    # Plot the distance of the position in task space of the right hand to the reference position
    # over time
    diff_ref_actual = np.linalg.norm(predicted_path - actual_path, axis=1)
    fig, axs = plt.subplots(1, 1)
    axs.plot(diff_ref_actual)
    axs.set_title("Distance between actual and reference position")
    axs.set_xlabel("Time")
    axs.set_ylabel("Distance")
    plt.savefig(f"images/{test_name}_distance_pos.pdf")


if __name__=="__main__":
    CONTROL_TEST = "control"
    INVERSE_GEOMETRY_TEST = "inverse_geometry"
    CONTROL_VARY_KP = "control_vary_kp"
    VALUES_OF_KP = [150, 200, 300, 400, 500, 750, 1000, 2000]
    CONTROL_VARY_BEZIER = "control_vary_bezier"
    VALUES_OF_N_BEZIER = [None, 3, 5, 10, 15]

    TEST_TO_RUN = CONTROL_TEST ### !!!! TODO CHANGE THIS TO RUN DIFFERENT TESTS !!!! ####
    TEST_CASES_TO_RUN = [config.TEST_CASE]  # To run it on all test cases, use: TEST_CASES

    if TEST_TO_RUN == INVERSE_GEOMETRY_TEST:
        print("!!!!!REMINDER!!!!! Please start meshcat server by running 'meshcat-server' in a terminal")

    successes = {}
    for test_case in TEST_CASES_TO_RUN:
        print(f"Running test case: {test_case}")

        test_case_successes = []

        # run the test case 10 times and get the success rate
        NUM_RUNS = 10
        for i in range(NUM_RUNS):

            if TEST_TO_RUN == CONTROL_TEST:
                current_time = time.time()
                reached, dist_to_goal, predicted_path, actual_path, qs_actual, qs_reference, qs_diff = control_main(render=True)
                time_taken = time.time() - current_time
                plot_graphs(predicted_path, actual_path, qs_actual, qs_reference, qs_diff, test_case)
                test_case_successes.append((reached, dist_to_goal, time_taken))
            elif TEST_TO_RUN == INVERSE_GEOMETRY_TEST:
                current_time = time.time()
                successinit, successend, error = inverse_geometry.inverse_geometry_main(viz_on=True)
                time_taken = time.time() - current_time
                test_case_successes.append((successinit, successend, error, time_taken))
            elif TEST_TO_RUN == CONTROL_VARY_KP:
                # Update the value of Kp in control
                assert len(VALUES_OF_KP) == NUM_RUNS, "Number of runs should match the number of values of Kp"
                control.Kp = VALUES_OF_KP[i]
                importlib.reload(control)
                current_time = time.time()
                reached, dist_to_goal, predicted_path, actual_path, qs_actual, qs_reference, qs_diff = control_main(render=False)
                time_taken = time.time() - current_time
                test_case_successes.append((reached, predicted_path, actual_path, time_taken))
            elif TEST_TO_RUN == CONTROL_VARY_BEZIER:
                # Update the value of N in bezier
                assert len(VALUES_OF_N_BEZIER) == NUM_RUNS, "Number of runs should match the number of values of N"
                control.N_BEZIERS = VALUES_OF_N_BEZIER[i]
                importlib.reload(control)
                current_time = time.time()
                reached, dist_to_goal, predicted_path, actual_path, qs_actual, qs_reference, qs_diff = control_main(render=False)
                time_taken = time.time() - current_time
                test_case_successes.append((reached, predicted_path, actual_path, time_taken))
            else:
                raise ValueError("Invalid test case")

        successes[test_case] = test_case_successes

    if TEST_TO_RUN == INVERSE_GEOMETRY_TEST:
        # summarise the number of successes for each test case and the final error. average over the runs for the error
        print("\n\n##############\n\n")
        print("Test Name & Success Rate & Error & Time (s) \\\\ \\hline")

        # Iterate over each test case in the 'successes' dictionary
        for test_case in successes:
            # Calculate the total success for init and end
            successinit = sum([success[0] for success in successes[test_case]])
            successend = sum([success[1] for success in successes[test_case]])

            # Calculate the mean error and time
            error = np.mean([success[2] for success in successes[test_case]])
            time = np.mean([success[3] for success in successes[test_case]])/2 # as we have 2 runs in one test case

            # Combine the success rates
            total_tests = len(successes[test_case])
            success_rate = f"{successinit + successend} / {(2 * total_tests)}"

            # Print the result for this test case, rounding to 3 significant digits
            print(f"{test_case} & {success_rate} & {error:.6f} & {time:.3f} \\\\ \\hline")
    elif TEST_TO_RUN == CONTROL_VARY_KP:
        for test_case in successes:
            # plot the average distance between the actual and reference path for each value of Kp along the time it took
            distance_actual_reference = []
            time_taken = []
            for success in successes[test_case]:
                actual_path = np.array(success[2])
                predicted_path = np.array(success[1])
                distance_per_timestep = np.linalg.norm(actual_path - predicted_path, axis=1)
                mean_distance = np.mean(distance_per_timestep, axis=0)
                distance_actual_reference.append(mean_distance)
                time_taken.append(success[3])
            error_path = np.array(distance_actual_reference)
            time_taken = np.array(time_taken)

            # Plot Error and time vs Kp
            fig, axs = plt.subplots(1, 1)
            axs.plot(VALUES_OF_KP, error_path, label="Error")
            axs.set_title("Average error trajectory vs Kp")
            axs.set_xlabel("Kp")
            axs.set_ylabel("Error")
            plt.savefig(f"images/{test_case}_vary_kp_error.pdf")

            fig, axs = plt.subplots(1, 1)
            axs.plot(VALUES_OF_KP, time_taken, label="Time")
            axs.set_title("Time vs Kp")
            axs.set_xlabel("Kp")
            axs.set_ylabel("Time")
            plt.savefig(f"images/{test_case}_vary_kp_time.pdf")
    elif TEST_TO_RUN == CONTROL_VARY_BEZIER:
        for test_case in successes:
            # plot the average distance between the actual and reference path for each value of N along the time it took
            distance_actual_reference = []
            time_taken = []
            for success in successes[test_case]:
                actual_path = np.array(success[2])
                predicted_path = np.array(success[1])
                distance_per_timestep = np.linalg.norm(actual_path - predicted_path, axis=1)
                mean_distance = np.mean(distance_per_timestep, axis=0)
                distance_actual_reference.append(mean_distance)
                time_taken.append(success[3])
            error_path = np.array(distance_actual_reference)
            time_taken = np.array(time_taken)

            # Plot Error and time vs N
            fig, axs = plt.subplots(1, 1)
            axs.plot(VALUES_OF_N_BEZIER, error_path, label="Error")
            axs.set_title("Average error trajectory vs Number of Bezier curves")
            axs.set_xlabel("N")
            axs.set_ylabel("Error")
            plt.savefig(f"images/{test_case}_vary_bezier_error.pdf")

            fig, axs = plt.subplots(1, 1)
            axs.plot(VALUES_OF_N_BEZIER, time_taken, label="Time")
            axs.set_title("Time vs N")
            axs.set_xlabel("N")
            axs.set_ylabel("Time")
            plt.savefig(f"images/{test_case}_vary_bezier_time.pdf")

    print(successes)

