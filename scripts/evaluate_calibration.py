import camera_joint_calibration_0_eval
import camera_joint_calibration_1_eval
import camera_joint_calibration_efforts_eval
import calibration_set_merger
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from pathlib import Path

def generate_calibrations_0():
    for x in range(20, 35):
        calibration_set_merger.random_split(x)
        camera_joint_calibration_0_eval.main()

def generate_calibrations_1():
    for x in range(5):
        calibration_set_merger.random_split(x)
        camera_joint_calibration_1_eval.main()

def generate_calibrations_3():
    for x in range(5):
        calibration_set_merger.random_split(x)
        camera_joint_calibration_efforts_eval.main()
def load_json(base_path, dir_name, file_name = "camera_calibration_result.yaml"):
    path = Path(os.path.join(base_path, dir_name))/file_name
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data

def plot_results(result_key, list_of_result_dicts, axis):
    plt.figure(result_key)
    for idx, result in enumerate(list_of_result_dicts):
        if isinstance(result[result_key], float):
            list_to_plot = [result[result_key]]
        elif "data" in result[result_key]:
            list_to_plot = result[result_key]["data"]
        else:
            list_to_plot = result[result_key]
        axis.plot(np.arange(0, len(list_to_plot), 1), list_to_plot, 'x')
        axis.set_title(result_key)
        axis.set_xlabel('-')
        axis.set_ylabel('-')
    # plt.show()

def plot_results_std(result_key, result_std_key, list_of_result_dicts, axis, title = "-"):
    for idx, result in enumerate(list_of_result_dicts):
        list_to_plot = [result[result_key]]
        list_to_plot_std = [result[result_std_key]]
        axis.errorbar([idx], list_to_plot, list_to_plot_std, linestyle='None', marker='x')
        axis.set_title(title)
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
    # plt.show()

def se3_avg(T_lst):
    aa_arr = np.array([T.rotation for T in T_lst])
    t_arr = np.array([T.translation for T in T_lst])
    aa_mean = aa_arr.mean(axis=0)
    U, S, Vh = np.linalg.svd(aa_mean)
    R_mean_proj = U @ Vh
    t_mean = t_arr.mean(axis=0)
    return pin.SE3(R_mean_proj, t_mean)

def plot_result_transformation(result_key, list_of_result_dicts, axis1, axis2):
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
        axis1.plot(np.arange(0, 3, 1), np.rad2deg(pin.log3(Tf_mean.rotation.T@Tf.rotation)), 'x')
        axis2.plot(np.arange(0, 3, 1), (Tf.translation-Tf_mean.translation)*1000, 'x')

    axis1.set_title(result_key + " rot")
    axis2.set_title(result_key + " trans")
    axis1.set_ylabel("deg")
    axis2.set_ylabel("mm")
    # plt.show()

def get_avg_tf(result_key, list_of_result_dicts):
    Tf_lst = []
    for idx, result in enumerate(list_of_result_dicts):
        if "data" in result[result_key]:
            list_to_plot = result[result_key]["data"]
        else:
            list_to_plot = result[result_key]
        Tf = pin.SE3(np.reshape(np.array(list_to_plot), (-1, 4)))
        Tf_lst.append(Tf)
    Tf_mean = se3_avg(Tf_lst)
    return Tf_mean

def get_avg_K(result_key, list_of_result_dicts):
    K_lst = []
    K_sum = np.zeros((4, 4))
    count = 0
    for idx, result in enumerate(list_of_result_dicts):
        if "data" in result[result_key]:
            list_to_plot = result[result_key]["data"]
        else:
            list_to_plot = result[result_key]
        K = np.reshape(np.array(list_to_plot), (-1, 4))
        K_sum += K
        count += 1
        K_lst.append(K)

    K_mean = K_sum/count
    # Tf_mean = se3_avg(Tf_lst)
    return K_mean

def get_avg_joint_offsets(result_key, list_of_result_dicts):
    joint_sum = np.zeros((9))
    count = 0
    for idx, result in enumerate(list_of_result_dicts):
        joint_sum += result[result_key]
        count += 1
    return joint_sum/count

def main():
    # BASE_PATH = "/home/bagr/ws_moveit/src/calibration_eyeinhand_ros_pkg/calibration_results"
    BASE_PATH = "C:\\Users\\Vojta\\PycharmProjects\\calibration_eyeinhand_ros_pkg\\calibration_results"
    datasets0 = ["30_08_2023_20_11_26",
"30_08_2023_20_13_17",
"30_08_2023_20_14_55",
"30_08_2023_20_16_33",]

    datasets1 = ["30_08_2023_20_19_48",
"30_08_2023_20_21_31",
"30_08_2023_20_23_07",
"30_08_2023_20_24_38",]

    datasets2 = ["30_08_2023_20_27_59",
"30_08_2023_20_29_25",
"30_08_2023_20_30_44",
"30_08_2023_20_32_03",]

    dataset_final = ["31_08_2023_15_22_58",
"31_08_2023_15_24_32",
"31_08_2023_15_26_06",
"31_08_2023_15_27_38",
"31_08_2023_15_29_22",
"31_08_2023_15_30_58",
"31_08_2023_15_32_47",
"31_08_2023_15_34_38",
"31_08_2023_15_36_17",
"31_08_2023_15_38_04",
"31_08_2023_15_39_45",
"31_08_2023_15_41_32",
"31_08_2023_15_43_06",
"31_08_2023_15_44_55",
"31_08_2023_15_47_07"
    ]

    # print(load_json(BASE_PATH, datasets[0])["panda_joint_offsets(deg)"])
    figure, axis = plt.subplots(3, 2)
    results = []
    for dataset in dataset_final:
        results.append(load_json(BASE_PATH, dataset, "camera_calibration_result.yaml"))
    print(get_avg_joint_offsets("panda_joint_offsets(deg)", results))
    print(get_avg_tf("bt_transform", results))
    print(get_avg_tf("gc_transform", results))
    print(get_avg_K("projection_matrix", results))
    # plot_results("panda_joint_offsets(deg)", results, axis[0][0])
    # plot_result_transformation("bt_transform", results, axis[1][0], axis[1][1])
    # plot_result_transformation("gc_transform", results, axis[2][0], axis[2][1])
    # plt.subplots_adjust(bottom=0.05, left=0.125, right=0.9, top=0.950, wspace=0.7, hspace=0.4)
    # axis[0][0].set_ylabel("deg")
    # axis[0][0].set_xlabel("joint")
    #
    # axis[1][0].set_title("bt_rot (error from mean)")
    # axis[1][1].set_title("bt_trans (error from mean)")
    #
    # axis[2][0].set_title("gc_rot (error from mean)")
    # axis[2][1].set_title("gc_trans (error from mean)")
    # plt.show()

    #plot_results_std("all_images_max", error_results)

    # figure, axis = plt.subplots(1, 3)
    # results = []
    # for dataset in datasets0:
    #     results.append(load_json(BASE_PATH, dataset, "camera_calibration_result.yaml"))
    # error_results = []
    # for dataset in datasets0:
    #     error_results.append(load_json(BASE_PATH, dataset + "/eye_in_hand", "reprojection_errors.yaml"))
    # plot_results_std("all_images_mean", "all_images_std", error_results, axis[0], "joint offsets + torque calib")
    #
    # results = []
    # for dataset in datasets1:
    #     results.append(load_json(BASE_PATH, dataset, "camera_calibration_result.yaml"))
    # error_results = []
    # for dataset in datasets1:
    #     error_results.append(load_json(BASE_PATH, dataset + "/eye_in_hand", "reprojection_errors.yaml"))
    # plot_results_std("all_images_mean", "all_images_std", error_results, axis[1], "joint offsets calib")
    #
    # results = []
    # for dataset in datasets2:
    #     results.append(load_json(BASE_PATH, dataset, "camera_calibration_result.yaml"))
    # error_results = []
    # for dataset in datasets2:
    #     error_results.append(load_json(BASE_PATH, dataset + "/eye_in_hand", "reprojection_errors.yaml"))
    # plot_results_std("all_images_mean", "all_images_std", error_results, axis[2], "only opencv calib")

    # plt.show()
    # plot_results("bt_transform", results)
    # plot_results("gc_transform", results)
    # plot_results("projection_matrix", results)



if __name__ == "__main__":
    # generate_calibrations_3()
    # generate_calibrations_0()
    # generate_calibrations_1()

    main()