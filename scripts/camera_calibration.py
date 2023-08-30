
"""
IMPORTANT NOTE: 
Naming convention for transformations follows opencv. For instance:
T_g2b brings a vector expressed in gripper frame to the base frame like
base_v = T_g2b @ gripper_v

Chain rule notation T_base_gripper or T_bg for short would be easier to read.

In calibration_imagex.yaml files are stored translation ('t_vec') and rotation ('r_mtx') corresponding to the transformation:
T_base_eef aka. T_eef2base

LIST OF ALL frames and short names:
b = "base" -> root of the robot kinematic chain
g = "gripper" -> end effector of the robot
c = "camera" -> camera frame, centered at its optical frame
t = "target" -> frame attached to the board used for calibration 
"""

from pathlib import Path
import copy
import numpy as np
import numpy.matlib  # necessary to use np.matlib.repmat
import matplotlib.pyplot as plt
import cv2
from cv2 import aruco

# To deal with SE3
import pinocchio as pin  

#import tf.transformations as tft

from utils import read_yaml, save_yaml, get_time_str, load_images

from eye_in_hand_calibration import calibrate_eye_in_hand, refine_eye_in_hand_calibration


def calibrate(board, hand_eye_method, paths, image_names_ignored=[], use_eye_in_hand_refinement=False):

    # load images for once all images in grayscale
    img_file_map = load_images(paths['data_dir'])  # dict of {img_name (without suffix): gray_image}

    # image_names_ignored = []
    # image_names_ignored = ['calibration_image_017']
    # image_names_ignored = ['calibration_image_014', 'calibration_image_018']
    for name_ign in image_names_ignored:
        if name_ign in img_file_map:
            img_file_map.pop(name_ign)

    # Detect markers and charuco board once and for all
    all_ch_points, all_ch_corners, all_ch_ids, all_ar_corners, all_ar_ids, unused_image_names = detect_corners_all_imgs(img_file_map, board)
    
    # Filter out images with too few marker detections
    for img_name in unused_image_names:
        img_file_map.pop(img_name)

    ################################################
    ################# INTRINSICS ###################
    ################################################

    # Intrinsics calibration
    imsize = next(iter(img_file_map.values())).shape  # First get any of the images size (the first will do)
    img_height, img_width = imsize[0], imsize[1]
    # rvecs_ct, tvecs_ct correspond to target2camera transformation in opencv notation convention 
    cam_mtx, dist_coef, rvecs_ct, tvecs_ct = calibrate_camera(all_ch_points, all_ch_corners, imsize)

    # Draw and save charuco detections for all images
    for i, (img_name, img) in enumerate(img_file_map.items()):
        save_charuco_detections(paths['camera_calibration_dir'], img_name, img, 
                                all_ar_corners[i], all_ar_ids[i], all_ch_corners[i], all_ch_ids[i], rvecs_ct[i], tvecs_ct[i], cam_mtx, dist_coef)

    save_print_reprojection_errors(img_file_map, tvecs_ct, rvecs_ct, cam_mtx, dist_coef, 
                                   all_ch_points, all_ch_corners, all_ch_ids, paths['camera_calibration_dir'])

    # Store camera intrinsics calibration data
    # TOREMOVE IF NOT USED
    intrinsics_calibration_logs = {
        'camera_matrix': cam_mtx.tolist(),
        'distortion_coefficients': dist_coef.tolist()
    }
    for img_name, rvec_ct, tvec_ct in zip(img_file_map, rvecs_ct, tvecs_ct):
        intrinsics_calibration_logs[img_name] = {}
        intrinsics_calibration_logs[img_name]['rvec_ct'] = rvec_ct.tolist()
        intrinsics_calibration_logs[img_name]['t_ct'] = tvec_ct.tolist()
    save_yaml(paths['results_file_path'], intrinsics_calibration_logs)


    ################################################
    ################# EYE IN HAND ##################
    ################################################
    # Recover robot kinematics data
    t_bg_lst, R_bg_lst, T_bg_lst = [], [], []
    for img_name in img_file_map:
        # Each image comes with accompanying yaml file with current kinematics information
        moveit_img_info_path = (paths['data_dir'] / img_name).with_suffix('.yaml')
        moveit_data = read_yaml(moveit_img_info_path)

        t_bg = np.array(moveit_data['t_vec'])
        R_bg = np.array(moveit_data['r_mtx'] )[0:3,0:3]
        t_bg_lst.append(t_bg)
        R_bg_lst.append(R_bg)
        T_bg_lst.append(pin.SE3(R_bg, t_bg))
    
    image_names = list(img_file_map.keys())
    T_gc, T_bt = calibrate_eye_in_hand(rvecs_ct, tvecs_ct, R_bg_lst, t_bg_lst, image_names, paths['eye_in_hand_dir'], hand_eye_method)

    if use_eye_in_hand_refinement:
        print('Use refinement step for eye in hand')
        T_gc, T_bt = refine_eye_in_hand_calibration(T_gc, T_bt, T_bg_lst, cam_mtx, dist_coef, all_ch_points, all_ch_corners)

    # Recover camera/target transformation from calibration and kinematics
    rvecs_ct, tvecs_ct = [], []
    T_cg = T_gc.inverse()
    for T_bg in T_bg_lst:
        T_gb = T_bg.inverse()

        T_ct = T_cg * T_gb * T_bt

        tvecs_ct.append(T_ct.translation)
        rvecs_ct.append(pin.log3(T_ct.rotation))


    save_print_reprojection_errors(img_file_map, tvecs_ct, rvecs_ct, cam_mtx, dist_coef, 
                                   all_ch_points, all_ch_corners, all_ch_ids, paths['eye_in_hand_dir'])


    # Save all possibly usefull data in ROS compatible format
    data_to_save = {}
    data_to_save['image_width'] = img_width
    data_to_save['image_height'] = img_height
    data_to_save['camera_name'] = 'camera'
    data_to_save['distortion_model'] = 'plumb_bob'
    data_to_save['camera_matrix'] = {'rows':3,'cols':3, 'data':cam_mtx.reshape(1,9)[0].tolist()}
    data_to_save['distortion_coefficients'] = {'rows':1,'cols':5, 'data':dist_coef[0].tolist()}
    data_to_save['rectification_matrix'] = {'rows':3,'cols':3, 'data':np.eye(3,3).reshape(1,9)[0].tolist()}
    data_to_save['projection_matrix'] = {'rows':3,'cols':4, 'data':np.hstack((cam_mtx,[[0],[0],[0]])).reshape(1,12)[0].tolist()}
    data_to_save['gc_transform'] = {'rows':4,'cols':4, 'data': T_gc.homogeneous.reshape(1,16)[0].tolist()}
    data_to_save['bt_transform'] = {'rows':4,'cols':4, 'data': T_bt.homogeneous.reshape(1,16)[0].tolist()}
    data_to_save['base_frame'] = moveit_data['base_frame']
    data_to_save['end_effector_frame'] = moveit_data['eef_frame']

    # Save Camera calibration
    camera_intrinsic = {}
    camera_intrinsic['image_width'] = img_width
    camera_intrinsic['image_height'] = img_height
    camera_intrinsic['camera_name'] = 'camera'
    camera_intrinsic['distortion_model'] = 'plumb_bob'
    camera_intrinsic['camera_matrix'] = {'rows':3,'cols':3, 'data':cam_mtx.reshape(1,9)[0].tolist()}
    camera_intrinsic['distortion_coefficients'] = {'rows':1,'cols':5, 'data':dist_coef[0].tolist()}
    camera_intrinsic['rectification_matrix'] = {'rows':3,'cols':3, 'data':np.eye(3,3).reshape(1,9)[0].tolist()}
    camera_intrinsic['projection_matrix'] = {'rows':3,'cols':4, 'data':np.hstack((cam_mtx,[[0],[0],[0]])).reshape(1,12)[0].tolist()}
    save_yaml(paths['camera_calibration_dir'] / 'camera.yaml', camera_intrinsic)

    calib_info = {}
    calib_info['time_of_calibration'] = get_time_str("%d-%m-%Y-%H:%M:%S")
    calib_info['used_calibration_data'] = paths['data_dir']
    calib_info['note'] = ''
    save_yaml(paths['expe_dir'] / 'calibration_info.yaml', calib_info)
    save_yaml(paths['expe_dir'] / 'camera_calibration_result.yaml', data_to_save)

    # Return esential data
    data_to_return = {}
    data_to_return['camera_matrix'] = cam_mtx
    data_to_return['distortion_coef'] = dist_coef
    data_to_return['gc_transform'] = T_gc.homogeneous
    data_to_return['bt_transform'] = T_bt.homogeneous

    return data_to_return


def detect_corners_all_imgs(img_file_map, board):

    all_ch_points = []
    all_ar_corners = []
    all_ar_ids = []
    all_ch_corners = []
    all_ch_ids = []
    unused_images = []
    
    for img_name, img in img_file_map.items():
        visible_ch_pts, ar_corners, ar_ids, ch_corners, ch_ids, good_img = detect_charuco_markers(img, board)

        if not good_img:
            unused_images.append(img_name)
            continue
        
        # Store detected chessboard corners for successful imgs
        all_ch_points.append(visible_ch_pts)
        all_ar_corners.append(ar_corners)
        all_ar_ids.append(ar_ids)
        all_ch_corners.append(ch_corners)
        all_ch_ids.append(ch_ids)

    if len(all_ch_points) < 4:
        raise ValueError("ERROR: Not enough good images. Calibration cancelled!")


    return all_ch_points, all_ch_corners, all_ch_ids, all_ar_corners, all_ar_ids, unused_images


def calibrate_camera(all_points, all_corners, imsize):
    """
    Perform intrinsics calibration of the camera.

    all_points and all_corners are list of lists such that:
    all_points[i][j] all_corners[i][j] -> i-th image, j-th corner detection
    all_points and all_corners have same sizes and
    len(all_points[i]) may vary depending on i (number of detected/visible corners depends on the taken image for a charuco board)

    all_points: 3D points coordinates of a pattern (e.g. charuco board)
    all_corners: 2D pixel detections corresponding to all_points
    imsize: tuple (image_width, image_height), only used as initial guess for intrinsics matrix
    """

    # Call opencv calibration routine
    flags = cv2.CALIB_FIX_ASPECT_RATIO
    # flags += cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
    # flags += cv2.CALIB_FIX_TANGENT_DIST
    camera_mtx_init = np.eye(3)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    criteria_intrinsics_solver = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-9)
    # Recover transformations target2camera -> ct 
    ret, cam_mtx, dist_coef, rvecs_ct, tvecs_ct = cv2.calibrateCamera(
                                                      objectPoints=all_points,
                                                      imagePoints=all_corners,
                                                      imageSize=imsize,
                                                      cameraMatrix=camera_mtx_init,
                                                      distCoeffs=None,
                                                      flags=flags,
                                                      criteria=criteria_intrinsics_solver)


    return cam_mtx, dist_coef, rvecs_ct, tvecs_ct


def save_print_reprojection_errors(img_file_map, tvecs_ct, rvecs_ct, cam_mtx, dist_coef, 
                                   all_ch_points, all_ch_corners, all_ch_ids, reprojection_dir):

    images_errors = {}
    all_proj_errors = []
    for (img_name, img), tvec_ct, rvec_ct, ch_pts, ch_corners, ch_ids in zip(img_file_map.items(), tvecs_ct, rvecs_ct, all_ch_points, all_ch_corners, all_ch_ids):

        projected_visible_ch_pts, jac = cv2.projectPoints(ch_pts, rvec_ct, tvec_ct, cam_mtx, dist_coef)
        img_errors = (projected_visible_ch_pts - ch_corners).squeeze()
        proj_errors = np.linalg.norm(img_errors, axis=1)  # compute reproje error for each individual corner
        all_proj_errors += proj_errors.tolist()
        # This previously computed "error" is not the mean reprojection error per say -> gives lower measure
        # mean_proj_error = cv2.norm(ch_corners, projected_visible_ch_pts, cv2.NORM_L2)/len(projected_visible_ch_pts)

        images_errors[img_name] = proj_errors.tolist()

        save_corner_reprojections(img_name, img, projected_visible_ch_pts, ch_corners, ch_ids, reprojection_dir)

    # casting to builtin float to be able to save to yaml
    images_errors['all_images_mean'] = float(np.mean(all_proj_errors))
    images_errors['all_images_min'] = min(all_proj_errors)
    images_errors['all_images_max'] = max(all_proj_errors)
    images_errors['all_images_std'] = float(np.std(all_proj_errors))
    save_yaml(reprojection_dir / 'reprojection_errors.yaml', images_errors)

    print(f"{reprojection_dir.stem} reprojection mean error: , {images_errors['all_images_mean']} px.")
    print(f"{reprojection_dir.stem} reprojection std error:  , {images_errors['all_images_std']} px.")
    print(f"{reprojection_dir.stem} reprojection min error:  , {images_errors['all_images_min']} px.")
    print(f"{reprojection_dir.stem} reprojection max error:  , {images_errors['all_images_max']} px.")

    return images_errors


def save_corner_reprojections(img_name, img, projected_visible_ch_pts, ch_corners, ch_ids, reprojection_dir):
    img_rep = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    # BGR
    color_detection = (255, 0, 0)
    color_reproj =    (0, 255, 0)
    aruco.drawDetectedCornersCharuco(img_rep, ch_corners, ch_ids, color_detection)
    aruco.drawDetectedCornersCharuco(img_rep, projected_visible_ch_pts, ch_ids, color_reproj)

    image_with_detections_path = (reprojection_dir / ('reproj_'+img_name)).with_suffix('.jpg')
    cv2.imwrite(image_with_detections_path.as_posix(), img_rep)


def get_hand_eye_calibration_method(data):
    """
    Return id of hand eye calibration method from its string. 

    For available methods https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html, HandEyeCalibrationMethod
    """

    cm_name = data['calibration_method']
    if hasattr(cv2, cm_name):
        return getattr(cv2, cm_name)
    else:
        raise(AttributeError(f'cv2 does not support hand calibration method {cm_name}'))


class ScaledBoard:
    ar_board: cv2.aruco_CharucoBoard
    ar_dict: cv2.aruco_Dictionary
    scaled_corners: np.ndarray

    def __init__(self, board_data):
        squaresX =           board_data['board_squaresX']
        squaresY =           board_data['board_squaresY']
        square_length =      board_data['board_square_size']
        marker_length =      board_data['board_marker_size']
        aruco_dict_name =    board_data['aruco_dictionary']
        self.board_scaling = board_data['board_scaling']
        if hasattr(aruco, aruco_dict_name):
            aruco_dict_id = getattr(aruco, aruco_dict_name)
        else:
            raise(AttributeError(f'aruco lib does not support aruco directory {aruco_dict_name} dict name'))

        self.ar_dict = aruco.Dictionary_get(aruco_dict_id)
        self.ar_board = aruco.CharucoBoard_create(squaresX, squaresY, square_length, marker_length, self.ar_dict)

        # board.chessboardCorners is NOT writable, we cannot override board.chessboardCorners even if it is not useful
        self.scaled_corners = self.scale_board_corners(self.ar_board.chessboardCorners, self.board_scaling)
        # self.ar_board.chessboardCorners = self.scaled_corners  # does not do anything

    @staticmethod
    def scale_board_corners(board_corners, scale):
        """
        Mitigate the fact that printed boards may not have the same scale in x and y axis.  
        """
        assert(len(scale) == 2)
        scale = (scale[0], scale[1], 1)
        scaled_pts = copy.deepcopy(board_corners)
        for i in range(len(board_corners)):
            scaled_pts[i] = np.multiply(board_corners[i], scale )
        return scaled_pts


def detect_charuco_markers(img: np.ndarray, board: ScaledBoard, K=None, dist=None):
    """
    Detect aruco markers corners then interpolate the chessboard corners.

    Interpolation is done using the board default chessboardCorners (the API does not permit to change its value).
    So it seems that using it for camera calibration works better as a result.
    """

    param_detector = aruco.DetectorParameters_create()
    # Default choice -> no refinement, other perform worse
    param_detector.cornerRefinementMethod = aruco.CORNER_REFINE_NONE

    # Detect the individual aruco markers, return their corners (tuple of arrays of size (1,4,2)) and their ids
    ar_corners, ar_ids, rejected_ar_corners = aruco.detectMarkers(img, board.ar_dict, parameters=param_detector)

    # A minimum number of aruco markers need to be detected for the image to be considered valid
    min_aruco_detected = 2
    if len(ar_corners) <= min_aruco_detected:
        return [], [], [], [], [], False
    
    # Takes aruco marker corners and associated ids and return chessboard (charuco) corners and associated ids 
    # Interpolate charuco corners using local homography computed from aruco marker corners
    #   - compute one local homography for each marker
    #   - for each charuco corner, compute its pixel based of the closest aruco marker homographies
    res, ch_corners, ch_ids = aruco.interpolateCornersCharuco(ar_corners, ar_ids, img, board.ar_board)

    # Position of the visble chessboard corners in board frame (pseudo 3D)
    # Idea: using the scaled corners for calibration scripts should get better results. 
    # But actually it does slightly worse since the charuco corners are estimated using the unscaled corners -> incoherence
    visible_ch_pts = board.ar_board.chessboardCorners[ch_ids,:]  # non scaled coordinates
    # visible_ch_pts = board.scaled_corners[ch_ids,:]  # scaled corners 

    return visible_ch_pts, ar_corners, ar_ids, ch_corners, ch_ids, True


def save_charuco_detections(detection_directory, img_name, img, ar_corners, ar_ids, ch_corners, ch_ids, rvec=None, tvec=None, cam_mtx=None, dist_coef=None):
    # Looks nicer with some color
    img_det = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_det = aruco.drawDetectedMarkers(img_det, ar_corners, ar_ids)
    color_detection = (255, 0, 0)  # BGR
    aruco.drawDetectedCornersCharuco(img_det, ch_corners, ch_ids, color_detection)

    if rvec is not None and tvec is not None:
        cv2.drawFrameAxes(img_det, cam_mtx, dist_coef, rvec, tvec, 0.1)
    
    image_with_detections_path = (detection_directory / ('charuco_det_'+img_name)).with_suffix('.jpg')
    cv2.imwrite(image_with_detections_path.as_posix(), img_det)


def create_or_overwrite_dir(dir_path: Path):
    assert(isinstance(dir_path, Path))
    if not dir_path.exists():
        print(f'Creating new dir {dir_path}')
        dir_path.mkdir(parents=True, exist_ok=False)
    else:
        print(dir_path, 'already exists! Not creating a new one')


def plot_reprojection_errors(reprojection_error_path):
    calib_res_path = reprojection_error_path.parent
    calib_type = calib_res_path.stem
    images_errors = read_yaml(reprojection_error_path)

    plt.figure(f'Reproj error {calib_type}')
    for img_name in images_errors:
        if 'all_images' in img_name:
            # Aggregated metrics are also stored in this yaml, ignore them here
            continue
        img_err = images_errors[img_name]
        # Assume name of the image is ending with _011 for instance
        idx = int(img_name.split('_')[-1])
        plt.plot(idx*np.ones(len(img_err)), img_err, 'x')
        plt.title(f'Reprojection error per image {calib_type}')
        plt.xlabel('Image #')
        plt.ylabel('Reprojection error (pix)')
    plt.savefig(calib_res_path / 'reprojection_errors.jpg')


def setup_paths_and_create_directories(config):
    
    # Directories for loading data and storing results of the calibration
    data_dir = Path(config['directory'])
    if not data_dir.exists():
        raise FileNotFoundError(f'Wrong data directory {data_dir}')
    result_dir = Path(config['result_directory'])

    # Sub directories for storing calibration results for each expe
    expe_name = config['expe_name']
    if expe_name == '':
        expe_name = get_time_str("%d-%m-%Y-%H:%M:%S")
    expe_dir = result_dir / Path(expe_name)  # directory containing all results from one experiment 

    # contains both intrinsics and eye in hand results -> split?
    # These information are already stored in camera_calibration and eye_in_hand directories -> REDUNDANT!
    results_file_path = expe_dir / Path(config['result_name'])

    camera_calibration_dir = expe_dir / 'camera_calibration'
    eye_in_hand_dir = expe_dir / 'eye_in_hand'

    create_or_overwrite_dir(result_dir)
    create_or_overwrite_dir(camera_calibration_dir)
    create_or_overwrite_dir(eye_in_hand_dir)

    paths = {
        'data_dir': data_dir,
        'expe_dir': expe_dir,
        'results_file_path': results_file_path,
        'camera_calibration_dir': camera_calibration_dir,
        'eye_in_hand_dir': eye_in_hand_dir,
    }

    return paths



def main():
    config = read_yaml('../config/calibration_config.yaml')

    paths = setup_paths_and_create_directories(config)

    board = ScaledBoard(board_data=config)
    hand_eye_method = get_hand_eye_calibration_method(config)

    
    calibration_data = calibrate(
                        board,
                        hand_eye_method,
                        paths,
                        config['image_names_ignored'],
                        config['use_eye_in_hand_refinement'])

    # scene_calib = calibrate_scene_cam(data['directory'],
    #                                     board,
    #                                     data['board_scaling'],
    #                                     data['result_directory'],
    #                                     data['result_name'],
    #                                     hand_eye_method,
    #                                   calibration_data)

    plot_reprojection_errors(paths['camera_calibration_dir'] / 'reprojection_errors.yaml')
    plot_reprojection_errors(paths['eye_in_hand_dir'] / 'reprojection_errors.yaml')
    plt.show()

    print('Result:')
    print('|-Camera Matrix:')
    for m in calibration_data['camera_matrix']:
        print('|---',end='')
        print(m)
    print('|-Distortion coefficients:')
    for m in calibration_data['distortion_coef']:
        print('|---',end='')
        print(m)
    print('|-Camera Gripper Transform:')
    for m in calibration_data['gc_transform']:
        print('|---',end='')
        print(m)

if __name__ == '__main__':
    main()
