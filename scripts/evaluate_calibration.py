import camera_joint_calibration_0_eval
import camera_joint_calibration_1_eval
import calibration_set_merger
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from pathlib import Path

def generate_calibrations_0():
    for x in range(10):
        calibration_set_merger.random_split(x)
        camera_joint_calibration_0_eval.main()

def generate_calibrations_1():
    for x in range(10):
        calibration_set_merger.random_split(x)
        camera_joint_calibration_1_eval.main()

def load_json(base_path, dir_name, file_name = "camera_calibration_result.yaml"):
    path = Path(os.path.join(base_path, dir_name))/file_name
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data

def plot_results(result_key, list_of_result_dicts):
    plt.figure(result_key)
    for idx, result in enumerate(list_of_result_dicts):
        if isinstance(result[result_key], float):
            list_to_plot = [result[result_key]]
        elif "data" in result[result_key]:
            list_to_plot = result[result_key]["data"]
        else:
            list_to_plot = result[result_key]
        plt.plot(np.arange(0, len(list_to_plot), 1), list_to_plot, 'x')
        plt.title(result_key)
        plt.xlabel('-')
        plt.ylabel('-')
    plt.show()

def plot_results_std(result_key, result_std_key, list_of_result_dicts, axis):
    for idx, result in enumerate(list_of_result_dicts):
        list_to_plot = [result[result_key]]
        list_to_plot_std = [result[result_std_key]]
        axis.errorbar([idx], list_to_plot, list_to_plot_std, linestyle='None', marker='x')
        axis.set_title(result_key)
        axis.set_xlabel('datasets')
        axis.set_ylabel('pix')

def plot_values(list_of_values, title):
    plt.figure(title)
    for idx, result in enumerate(list_of_values):
        list_to_plot = result[list_of_values]
        plt.plot(np.arange(0, len(list_to_plot), 1), list_to_plot, 'x')
        plt.title(title)
        plt.xlabel('-')
        plt.ylabel('-')
    plt.show()

def se3_avg(T_lst):
    aa_arr = np.array([T.rotation for T in T_lst])
    t_arr = np.array([T.translation for T in T_lst])
    aa_mean = aa_arr.mean(axis=0)
    U, S, Vh = np.linalg.svd(aa_mean)
    R_mean_proj = U @ Vh
    t_mean = t_arr.mean(axis=0)
    return pin.SE3(R_mean_proj, t_mean)

def plot_result_transformation(result_key, list_of_result_dicts, ):
    figure, axis = plt.subplots(1, 2)
    Tf_lst = []
    for idx, result in enumerate(list_of_result_dicts):
        if "data" in result[result_key]:
            list_to_plot = result[result_key]["data"]
        else:
            list_to_plot = result[result_key]
        Tf = pin.SE3(np.reshape(np.array(list_to_plot), (-1, 4)))
        Tf_lst.append(Tf)
    Tf_mean = se3_avg(Tf_lst)
    for idx, result in enumerate(list_of_result_dicts):
        if "data" in result[result_key]:
            list_to_plot = result[result_key]["data"]
        else:
            list_to_plot = result[result_key]
        Tf = pin.SE3(np.reshape(np.array(list_to_plot), (-1, 4)))
        axis[0].plot(np.arange(0, 3, 1), np.rad2deg(pin.log3(Tf_mean.rotation.T@Tf.rotation)), 'x')
        axis[1].plot(np.arange(0, 3, 1), (Tf.translation-Tf_mean.translation)*1000, 'x')

    axis[0].set_title(result_key + " rot")
    axis[1].set_title(result_key + " trans")
    axis[0].set_ylabel("deg")
    axis[1].set_ylabel("mm")
    plt.show()


def main():
    BASE_PATH = "/home/bagr/ws_moveit/src/calibration_eyeinhand_ros_pkg/calibration_results"
    datasets0 = ["28-08-2023-14:37:24",
    "28-08-2023-14:40:19",
    "28-08-2023-14:43:03",
    "28-08-2023-14:45:18",
    "28-08-2023-14:47:17",
    "28-08-2023-14:49:28",
    "28-08-2023-14:51:19",
    "28-08-2023-14:53:26",
    "28-08-2023-14:55:55",
    "28-08-2023-14:57:58"]

    datasets1 = ["28-08-2023-15:00:21",
"28-08-2023-15:00:53",
"28-08-2023-15:01:29",
"28-08-2023-15:02:02",
"28-08-2023-15:02:37",
"28-08-2023-15:03:10",
"28-08-2023-15:03:43",
"28-08-2023-15:04:16",
"28-08-2023-15:04:52",
"28-08-2023-15:05:24"]

    # print(load_json(BASE_PATH, datasets[0])["panda_joint_offsets(deg)"])
    # plot_results("panda_joint_offsets(deg)", results)
    # plot_result_transformation("bt_transform", results)
    # plot_result_transformation("gc_transform", results)
    #plot_results_std("all_images_max", error_results)
    figure, axis = plt.subplots(1, 2)
    results = []
    for dataset in datasets0:
        results.append(load_json(BASE_PATH, dataset, "camera_calibration_result.yaml"))
    error_results = []
    for dataset in datasets0:
        error_results.append(load_json(BASE_PATH, dataset + "/eye_in_hand", "reprojection_errors.yaml"))
    plot_results_std("all_images_mean", "all_images_std", error_results, axis[0])
    results = []
    for dataset in datasets1:
        results.append(load_json(BASE_PATH, dataset, "camera_calibration_result.yaml"))
    error_results = []
    for dataset in datasets1:
        error_results.append(load_json(BASE_PATH, dataset + "/eye_in_hand", "reprojection_errors.yaml"))
    plot_results_std("all_images_mean", "all_images_std", error_results, axis[1])
    plt.show()
    # plot_results("bt_transform", results)
    # plot_results("gc_transform", results)
    # plot_results("projection_matrix", results)



if __name__ == "__main__":
    # generate_calibrations_0()
    # generate_calibrations_1()
    main()