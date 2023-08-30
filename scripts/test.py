import camera_calibration as cc
import os
from os.path import join
import cv2
import numpy as np
from utils import get_images_paths, read_yaml, save_yaml

def main():
    data = read_yaml('../config/test_config.yaml')
    board, aruco_dict = cc.get_aruco_board_and_dict(data)
    squares_scale = data['board_scaling']
    img_dir_path = data['directory']
    calibration_path = data['calibration_data_directory']
    test_directory = data['test_directory']
    calibraton_name = data['calibration_name']

    images = get_images_paths(img_dir_path)

    data = read_yaml(join(calibration_path, calibraton_name))
    Tcamera_gripper = data['camera_gripper_transform']
    Tcamera_gripper = np.array(Tcamera_gripper['data']).reshape(Tcamera_gripper['rows'],Tcamera_gripper['cols'])
    Tbase_target = data['base_target_transform']
    Tbase_target = np.array(Tbase_target['data']).reshape(Tbase_target['rows'],Tbase_target['cols'])
    cam_mtx = data['camera_matrix']
    cam_mtx = np.array(cam_mtx['data']).reshape(cam_mtx['rows'],cam_mtx['cols'])
    dist_coef = data['distortion_coefficients']
    dist_coef = np.array(dist_coef['data']).reshape(dist_coef['rows'],dist_coef['cols'])

    log={}
    log['used_calibration_data'] = calibration_path
    log['used_testing_data'] = img_dir_path


    tvecs = []
    rvecs = []
    for i in range(len(images)):
        yml = str(os.path.splitext(images[i])[0]) + '.yaml'

        data = read_yaml(join(img_dir_path,yml))
        # ??
        Tbase_gripper = np.delete(data['r_mtx'],3,axis=1)
        Tbase_gripper = np.hstack(( Tbase_gripper, np.vstack((np.reshape(data['t_vec'],(3,1)),[1]) )))
        T = cc.get_Ttarget2camera(Tbase_target, Tbase_gripper, Tcamera_gripper)
        tvecs.append(np.matmul(T,[[0],[0],[0],[1]])[0:3,:])
        rvecs.append(cv2.Rodrigues(T[0:3,0:3])[0])

    os.makedirs(test_directory, exist_ok=True)
    result_directory = join(test_directory, cc.get_time_str("%d-%m-%Y-%H:%M:%S"))
    os.mkdir(result_directory)
    save_yaml(join(result_directory,'test_info.yaml'), log)
    eye_result_directory = join(result_directory,'eye_in_hand')

    os.mkdir(eye_result_directory)
    eye_reprojection_errors = cc.reprojection_error(images, img_dir_path, tvecs, rvecs, board, aruco_dict, cam_mtx, dist_coef, eye_result_directory, squares_scale, False)
    print("Eye-in-hand reprojection mean error: ", np.mean(eye_reprojection_errors), " px.")
    print("Eye-in-hand reprojection min error: ", min(eye_reprojection_errors), " px.")
    print("Eye-in-hand reprojection max error: ", max(eye_reprojection_errors), " px.")




if __name__ == '__main__':
    main()
