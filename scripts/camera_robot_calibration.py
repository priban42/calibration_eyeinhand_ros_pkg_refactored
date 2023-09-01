from pathlib import Path
import copy
import numpy as np
import numpy.matlib  # necessary to use np.matlib.repmat
import matplotlib.pyplot as plt
import cv2
from cv2 import aruco
import math
from collections import OrderedDict
import cv2
import yaml
import datetime
from pathlib import Path
from scipy.optimize import least_squares

from utils import read_yaml, save_yaml, get_time_str


# To deal with SE3
import pinocchio as pin

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
        self.scaled_corners = self.scale_board_corners(self.ar_board.chessboardCorners, self.board_scaling)

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

class RobotModel():
    def __init__(self, urdf_path = 'panda.urdf', eef_id = 'panda_hand'):
        self.urdf_path = urdf_path
        self.robot_model = pin.buildModelFromUrdf(self.urdf_path)
        self.robot_data = self.robot_model.createData()
        self.robod_eef_id = self.robot_model.getFrameId(eef_id)
        self.values_to_optimize = 7
        self.joint_offsets = np.zeros(7)
        self.joint_offsets_mask = np.array([1, 1, 1, 1, 1, 1, 1])

    def update_model(self, x):
        assert x.size == self.values_to_optimize
        self.joint_offsets = x[0:self.joint_offsets.size]
    def get_eef_pose(self, joint_values):
        corrected_joint_values = joint_values + self.joint_offsets*self.joint_offsets_mask
        expanded_joint_values = np.concatenate((corrected_joint_values, np.array([0, 0])))
        pin.forwardKinematics(self.robot_model, self.robot_data, expanded_joint_values)
        eef_pose = pin.updateFramePlacement(self.robot_model, self.robot_data, self.robod_eef_id)
        return eef_pose


class Calibration():
    def __init__(self, config_path = '../config/calibration_config.yaml'):
        self.config = read_yaml(config_path)
        self.paths = None
        self.board = ScaledBoard(board_data=self.config)

        self.robot_model = RobotModel()


        self.imsize = None
        self.data_dir = None

        # dictionaries with image name as key
        self.img_file_map = None
        self.detections_map = None
        self.transformations_map = None
        self.robot_configuration_map = None

        self.cam_mtx = None
        self.dist_coef = None
        self.T_bt = None
        self.T_gc = None

        self.res_len = None


    @staticmethod
    def create_or_overwrite_dir(dir_path: Path):
        assert (isinstance(dir_path, Path))
        if not dir_path.exists():
            print(f'Creating new dir {dir_path}')
            dir_path.mkdir(parents=True, exist_ok=False)
        else:
            print(dir_path, 'already exists! Not creating a new one')

    def setup_paths(self):

        # Directories for loading data and storing results of the calibration
        data_dir = Path(self.config['directory'])
        if not data_dir.exists():
            raise FileNotFoundError(f'Wrong data directory {data_dir}')
        result_dir = Path(self.config['result_directory'])

        # Sub directories for storing calibration results for each expe
        expe_name = self.config['expe_name']
        if expe_name == '':
            expe_name = get_time_str("%d_%m_%Y_%H_%M_%S")
        expe_dir = result_dir / Path(expe_name)  # directory containing all results from one experiment

        # contains both intrinsics and eye in hand results -> split?
        # These information are already stored in camera_calibration and eye_in_hand directories -> REDUNDANT!
        results_file_path = expe_dir / Path(self.config['result_name'])

        camera_calibration_dir = expe_dir / 'camera_calibration'
        eye_in_hand_dir = expe_dir / 'eye_in_hand'
        robot_calib = expe_dir / 'robot_calib'

        self.paths = {
            'data_dir': data_dir,
            'expe_dir': expe_dir,
            'result_dir': result_dir,
            'results_file_path': results_file_path,
            'camera_calibration_dir': camera_calibration_dir,
            'eye_in_hand_dir': eye_in_hand_dir,
            'robot_calib': robot_calib
        }

    def create_directories(self):
        self.create_or_overwrite_dir(self.paths["result_dir"])
        self.create_or_overwrite_dir(self.paths["camera_calibration_dir"])
        self.create_or_overwrite_dir(self.paths["eye_in_hand_dir"])
        self.create_or_overwrite_dir(self.paths["robot_calib"])
    @staticmethod
    def __get_images_paths(data_dir: Path) -> list:
        """
        @param data_dir: path to a directory with calibration images
        @return: list of paths of calibration images
        """
        return sorted(
            [path for path in data_dir.iterdir() if path.is_file() and path.suffix in {'.jpg', '.jpeg', '.png'}])

    @staticmethod
    def __detect_charuco_markers(img: np.ndarray, board: ScaledBoard, K=None, dist=None):
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
        visible_ch_pts = board.ar_board.chessboardCorners[ch_ids, :]  # non scaled coordinates
        # visible_ch_pts = board.scaled_corners[ch_ids,:]  # scaled corners

        return visible_ch_pts, ar_corners, ar_ids, ch_corners, ch_ids, True

    def __edit_image_on_loading(self, img: np.ndarray) -> np.ndarray:
        """
        This function is called upon loading each image.
        """
        height = img.shape[0]
        width = img.shape[1]
        new_image = cv2.rectangle(img, (width // 2 - 250, 0), (width // 2 + 150, 120),
                                  (100, 100, 100), -1)
        return new_image

    def load_images(self):
        """
        Loads images from self.data_dir into img_file_map.
        """
        image_paths = self.__get_images_paths(self.paths["data_dir"])
        self.img_file_map = OrderedDict()
        self.detections_map = OrderedDict()
        self.transformations_map = OrderedDict()

        for img_path in image_paths:
            img_name = img_path.stem
            img = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
            self.img_file_map[img_name] = self.__edit_image_on_loading(img)
        self.imsize = next(iter(self.img_file_map.values())).shape

    def update_board_detections(self):
        self.detections_map = OrderedDict()
        self.transformations_map = OrderedDict()

        for img_name, img in self.img_file_map.items():
            visible_ch_pts, ar_corners, ar_ids, ch_corners, ch_ids, good_img = self.__detect_charuco_markers(img, self.board)

            if not good_img or len(visible_ch_pts) < 6:
                self.img_file_map.pop(img_name)
                continue
            self.detections_map[img_name] = {"ch_points":visible_ch_pts,
                                               "ar_corners":ar_corners,
                                               "ar_ids":ar_ids,
                                               "ch_corners":ch_corners,
                                               "ch_ids":ch_ids, }

        if len(self.detections_map) < 4:
            raise ValueError("ERROR: Not enough good images. Calibration cancelled!")
        pass

    def calibrate_camera(self):
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

        self.transformations_map = OrderedDict()
        # Call opencv calibration routine
        flags = cv2.CALIB_FIX_ASPECT_RATIO
        # flags += cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3
        # flags += cv2.CALIB_FIX_TANGENT_DIST
        camera_mtx_init = np.eye(3)
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
        criteria_intrinsics_solver = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 100, 1e-9)


        all_points = [self.detections_map[img_name]["ch_points"] for img_name in self.detections_map]
        all_corners = [self.detections_map[img_name]["ch_corners"] for img_name in self.detections_map]
        # Recover transformations target2camera -> ct
        ret, cam_mtx, dist_coef, rvecs_ct, tvecs_ct = cv2.calibrateCamera(
            objectPoints=all_points,
            imagePoints=all_corners,
            imageSize=self.imsize,
            cameraMatrix=camera_mtx_init,
            distCoeffs=None,
            flags=flags,
            criteria=criteria_intrinsics_solver)
        self.cam_mtx = cam_mtx
        self.dist_coef = dist_coef

        for idx, img_name in enumerate(self.detections_map.keys()):
            T_ct = pin.SE3(pin.exp3(rvecs_ct[idx]), tvecs_ct[idx])
            self.transformations_map[img_name] = {"T_ct_camera": T_ct}

    def remove_poor_reprojection_images(self, max_error):
        images_to_remove = []
        for img_name, img in self.img_file_map.items():
            board_det = self.detections_map[img_name]
            tfs = self.transformations_map[img_name]
            T_ct = tfs["T_ct_camera"]
            rvec_ct = T_ct.rotation
            tvec_ct = T_ct.translation
            projected_visible_ch_pts, jac = cv2.projectPoints(board_det["ch_points"], rvec_ct, tvec_ct, self.cam_mtx, self.dist_coef)
            img_errors = (projected_visible_ch_pts - board_det["ch_corners"]).squeeze()
            proj_errors = np.linalg.norm(img_errors, axis=1)
            if max(proj_errors) > max_error:
                images_to_remove.append(img_name)
        for image_to_remove in images_to_remove:
            self.img_file_map.pop(image_to_remove)
            self.detections_map.pop(image_to_remove)
            self.transformations_map.pop(image_to_remove)

    @staticmethod
    def __save_corner_reprojections(img_name, img, projected_visible_ch_pts, ch_corners, ch_ids, reprojection_dir):
        img_rep = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        # BGR
        color_detection = (255, 0, 0)
        color_reproj = (0, 255, 0)
        aruco.drawDetectedCornersCharuco(img_rep, ch_corners, ch_ids, color_detection)
        aruco.drawDetectedCornersCharuco(img_rep, projected_visible_ch_pts, ch_ids, color_reproj)

        image_with_detections_path = (reprojection_dir / ('reproj_' + img_name)).with_suffix('.jpg')
        cv2.imwrite(image_with_detections_path.as_posix(), img_rep)

    def save_print_reprojection_errors(self, path, T_ct_key = "T_ct_camera"):

        images_errors = {}
        all_proj_errors = []
        for (img_name, img) in self.img_file_map.items():
            board_det = self.detections_map[img_name]
            tfs = self.transformations_map[img_name]
            T_ct = tfs[T_ct_key]
            rvec_ct = T_ct.rotation
            tvec_ct = T_ct.translation
            projected_visible_ch_pts, jac = cv2.projectPoints(board_det["ch_points"], rvec_ct, tvec_ct, self.cam_mtx, self.dist_coef)
            img_errors = (projected_visible_ch_pts - board_det["ch_corners"]).squeeze()
            proj_errors = np.linalg.norm(img_errors, axis=1)  # compute reproje error for each individual corner
            all_proj_errors += proj_errors.tolist()
            # This previously computed "error" is not the mean reprojection error per say -> gives lower measure
            # mean_proj_error = cv2.norm(ch_corners, projected_visible_ch_pts, cv2.NORM_L2)/len(projected_visible_ch_pts)

            images_errors[img_name] = proj_errors.tolist()
            self.__save_corner_reprojections(img_name, img, projected_visible_ch_pts, board_det["ch_corners"], board_det["ch_ids"], path)

        # casting to builtin float to be able to save to yaml
        images_errors['all_images_mean'] = float(np.mean(all_proj_errors))
        images_errors['all_images_min'] = min(all_proj_errors)
        images_errors['all_images_max'] = max(all_proj_errors)
        images_errors['all_images_std'] = float(np.std(all_proj_errors))
        save_yaml(path / 'reprojection_errors.yaml', images_errors)

        print(f"{path.stem} reprojection mean error: , {images_errors['all_images_mean']} px.")
        print(f"{path.stem} reprojection std error:  , {images_errors['all_images_std']} px.")
        print(f"{path.stem} reprojection min error:  , {images_errors['all_images_min']} px.")
        print(f"{path.stem} reprojection max error:  , {images_errors['all_images_max']} px.")

        return images_errors

    @staticmethod
    def se3_avg(T_lst):
        aa_arr = np.array([T.rotation for T in T_lst])
        t_arr = np.array([T.translation for T in T_lst])
        aa_mean = aa_arr.mean(axis=0)
        U, S, Vh = np.linalg.svd(aa_mean)
        R_mean_proj = U @ Vh
        t_mean = t_arr.mean(axis=0)
        return pin.SE3(R_mean_proj, t_mean)

    def plot_save_trans_rot_errors(self, T_bt_lst, T_bt_mean, file_name = 'translation_tb.jpg'):

        t_err = np.array([T_bt.translation - T_bt_mean.translation for T_bt in T_bt_lst])
        rot_err = np.array([pin.log3(T_bt_mean.rotation.T @ T_bt.rotation) for T_bt in T_bt_lst])
        t_err_mm = 1000 * t_err
        rot_err_deg = np.rad2deg(rot_err)

        x_labels = np.arange(len(T_bt_lst))
        fig = plt.figure('Translation error T_tb')
        for i in range(3):
            plt.plot(x_labels, t_err_mm[:, i], 'rgb'[i] + 'x', label='err_t_' + 'xyz'[i])
        plt.xlabel('Pose #')
        plt.ylabel('Translation error (mm)')
        plt.savefig(Path(self.paths["expe_dir"]) / file_name)
        plt.close()
    def calibrate_eye_in_hand(self):

        """
        calibration data contains at this point:
        - tvec, rvec: pose of the target in camera frame (or the inverse?) from intrinsics calibration phase
        - t_vec, r_mtx: pose of the gripper in robot base frame
        """

        # takes both rotation matrices or rotation vector (angle-axis) as inputs
        # gripper2base, target2cam notations, cam2gripper -> bg, ct, gc
        cm_name = self.config['calibration_method']
        if hasattr(cv2, cm_name):
            hand_eye_method = getattr(cv2, cm_name)
        else:
            raise (AttributeError(f'cv2 does not support hand calibration method {cm_name}'))

        R_bg_lst = [self.transformations_map[img_name]["T_bg"].rotation for img_name in self.transformations_map]
        t_bg_lst = [self.transformations_map[img_name]["T_bg"].translation for img_name in self.transformations_map]
        rvecs_ct = [self.transformations_map[img_name]["T_ct_camera"].rotation for img_name in self.transformations_map]
        tvecs_ct = [self.transformations_map[img_name]["T_ct_camera"].translation for img_name in self.transformations_map]
        R_gc, t_gc = cv2.calibrateHandEye(R_bg_lst, t_bg_lst, rvecs_ct, tvecs_ct, hand_eye_method)
        self.T_gc = pin.SE3(R_gc, t_gc)

        T_bt_lst = []
        for img_name in self.transformations_map:
            tf_dict = self.transformations_map[img_name]
            T_ct = tf_dict["T_ct_camera"]
            T_bg = tf_dict["T_bg"]
            T_bt = T_bg * self.T_gc * T_ct
            T_bt_lst.append(T_bt)

        self.T_bt = self.se3_avg(T_bt_lst)



    def save_result_transformations(self):
        q_gc = pin.Quaternion(self.T_gc.rotation).coeffs()
        t_gc = self.T_gc.translation
        q_bt = pin.Quaternion(self.T_bt.rotation).coeffs()
        t_bt = self.T_bt.translation
        result_transforms = {
            'T_gc': self.T_gc.homogeneous.tolist(),
            'q_gc': q_gc.tolist(),
            't_gc': t_gc.tolist(),
            'T_bt': self.T_bt.homogeneous.tolist(),
            'q_bt': q_bt.tolist(),
            't_bt': t_bt.tolist(),
        }
        save_yaml(Path(self.paths["expe_dir"]) / 'result_transforms.yaml', result_transforms)

    def update_robot_transformations(self):
        self.robot_configuration_map = OrderedDict()
        for img_name in self.img_file_map:
            image_info = read_yaml((self.paths['data_dir'] / img_name).with_suffix('.yaml'))
            joint_values = image_info["joint_values"][0:7]
            eef_pose = self.robot_model.get_eef_pose(joint_values)
            self.robot_configuration_map[img_name] = {"joint_values": joint_values}
            self.transformations_map[img_name]["T_bg"] = eef_pose

    def update_T_ct_robot_transformation(self):
        T_cg = self.T_gc.inverse()
        for img_name in self.transformations_map:
            T_gb = self.transformations_map[img_name]["T_bg"].inverse()
            T_ct_robot = T_cg * T_gb * self.T_bt
            self.transformations_map[img_name]["T_ct_robot"] = T_ct_robot

    def __get_transforms_from_x(self, x):
        # Optimization is done on SE(3)
        # -> use a minimal representation of decision variables as local increments "nu" on se3
        nu_gc = x[0:6]
        nu_bt = x[6:12]

        # Recover corresponding transformation matrices
        T_gc = self.T_gc * pin.exp6(nu_gc)
        T_bt = self.T_bt * pin.exp6(nu_bt)

        return T_gc, T_bt

    def get_reprj_res(self, x):
        T_gc, T_bt = self.__get_transforms_from_x(x[0:12])
        T_cg = T_gc.inverse()
        res = np.zeros(self.res_len)
        index_ij = 0

        self.robot_model.update_model(x[12:19])
        for img_name in self.robot_configuration_map:
            T_bg = self.robot_model.get_eef_pose(self.robot_configuration_map[img_name]["joint_values"])
            T_gb = T_bg.inverse()
            T_ct = T_cg * T_gb * T_bt

            points = self.detections_map[img_name]["ch_points"]
            rvec = T_ct.rotation
            tvec = T_ct.translation
            cij_proj, jacobian = cv2.projectPoints(points, rvec, tvec, self.cam_mtx, self.dist_coef)
            cij_proj = np.squeeze(cij_proj)
            cij_det = self.detections_map[img_name]["ch_corners"].squeeze()
            repj_error = cij_proj - cij_det
            points_count = len(self.detections_map[img_name]["ch_points"])
            # res[index_ij] = (np.sum(np.sum(repj_error ** 2, axis=-1) ** (0.5))/points_count)
            res[index_ij] = (np.sum(np.sum(repj_error ** 4, axis=-1))/points_count)
            # res[index_ij] = (np.sum(np.sum(repj_error, axis=-1))/self.res_len)
            index_ij += 1
        # penalize joint_offset size
        res = (1 + np.linalg.norm(self.robot_model.joint_offsets)*0.01) * res
        return res

    def calibrate_robot(self):
        x0 = np.concatenate((np.zeros(12), np.array([0, 0, 0, 0, 0, 0, 0] + [0, 0])), axis=0)
        # self.res_len = sum(2 * len(self.detections_map[img_name]["ch_points"]) for img_name in self.detections_map)
        self.res_len = len(self.detections_map)
        result = least_squares(fun=self.get_reprj_res, x0=x0, jac='3-point', method='trf', verbose=2, xtol=5e-4, max_nfev=50)
        self.robot_model.update_model(result.x[12:19])

    def plot_reprojection_errors(self, path, axis):
        calib_res_path = path.parent
        calib_type = path.stem
        images_errors = read_yaml(path / 'reprojection_errors.yaml')
        for img_name in images_errors:
            if 'all_images' in img_name:
                # Aggregated metrics are also stored in this yaml, ignore them here
                continue
            img_err = images_errors[img_name]
            # Assume name of the image is ending with _011 for instance
            idx = int(img_name.split('_')[-1])
            axis.plot(idx * np.ones(len(img_err)), img_err, 'x')
            axis.set_title(f'Reprojection error per image {calib_type}')
            axis.set_xlabel('Image #')
            axis.set_ylabel('Reprojection error (pix)')

    def plot_results(self):
        figure, axis = plt.subplots(2, 2)
        figure.tight_layout(pad=1.0)
        figure.set_figheight(7)
        figure.set_figwidth(15)
        self.plot_reprojection_errors(self.paths["camera_calibration_dir"], axis[0, 0])
        self.plot_reprojection_errors(self.paths["eye_in_hand_dir"], axis[1, 0])
        self.plot_reprojection_errors(self.paths["robot_calib"], axis[0, 1])
        plt.savefig(Path(self.paths["expe_dir"]) / 'reprojection_errors.png', dpi = 200)
        plt.show()

    def calibrate(self):
        self.setup_paths()
        self.create_directories()
        self.load_images()
        self.update_board_detections()
        for x in [40, 10, 5]:
            self.calibrate_camera()
            self.remove_poor_reprojection_images(max_error=x)
        self.save_print_reprojection_errors(path = self.paths["camera_calibration_dir"], T_ct_key="T_ct_camera")

        self.update_robot_transformations()
        self.calibrate_eye_in_hand()
        self.update_T_ct_robot_transformation()

        self.save_print_reprojection_errors(path = self.paths["eye_in_hand_dir"], T_ct_key="T_ct_robot")
        self.save_result_transformations()

        self.calibrate_robot()
        self.update_robot_transformations()
        self.calibrate_eye_in_hand()
        self.update_T_ct_robot_transformation()

        print("joint_offsets:", self.robot_model.joint_offsets * 180 / np.pi)

        self.save_print_reprojection_errors(path = self.paths["robot_calib"], T_ct_key="T_ct_robot")
        self.save_result_transformations()
        self.plot_results()


def main():
    c = Calibration()
    c.calibrate()

if __name__ == "__main__":
    main()







