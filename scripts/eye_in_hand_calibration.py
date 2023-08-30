import cv2
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from utils import read_yaml, save_yaml

from scipy.optimize import least_squares





def calibrate_eye_in_hand(rvecs_ct, tvecs_ct, R_bg_lst, t_bg_lst, img_names, expe_dir, hand_eye_method):

    """
    calibration data contains at this point:
    - tvec, rvec: pose of the target in camera frame (or the inverse?) from intrinsics calibration phase
    - t_vec, r_mtx: pose of the gripper in robot base frame 
    """

    # takes both rotation matrices or rotation vector (angle-axis) as inputs
    # gripper2base, target2cam notations, cam2gripper -> bg, ct, gc
    R_gc, t_gc = cv2.calibrateHandEye(R_bg_lst, t_bg_lst, rvecs_ct, tvecs_ct, hand_eye_method)
    T_gc = pin.SE3(R_gc, t_gc)

    transforms_to_save = {}

    T_bt_lst = []
    for rvec_ct, tvec_ct, R_bg, t_bg, img_name in zip(rvecs_ct, tvecs_ct, R_bg_lst, t_bg_lst, img_names):
        T_ct = pin.SE3(pin.exp3(rvec_ct), tvec_ct)
        T_bg = pin.SE3(R_bg, t_bg)
        
        T_bt = T_bg * T_gc * T_ct
        T_bt_lst.append(T_bt)

        transforms_to_save[img_name] = T_bt.homogeneous.tolist()

    T_bt_mean = se3_avg(T_bt_lst)

    plot_save_trans_rot_errors(T_bt_lst, T_bt_mean, expe_dir)

    save_yaml(expe_dir / 'bt_transform.yaml', transforms_to_save)
    
    q_gc = pin.Quaternion(T_gc.rotation).coeffs() 
    t_gc = T_gc.translation
    q_bt = pin.Quaternion(T_bt_mean.rotation).coeffs()
    t_bt = T_bt_mean.translation
    result_transforms = {
        'T_gc': T_gc.homogeneous.tolist(), 
        'q_gc': q_gc.tolist(),
        't_gc': t_gc.tolist(),
        'T_bt': T_bt_mean.homogeneous.tolist(),
        'q_bt': q_bt.tolist(),
        't_bt': t_bt.tolist(),
    }
    save_yaml(expe_dir / 'result_transforms.yaml', result_transforms)

    return T_gc, T_bt_mean

# works incorrectly
# def se3_avg(T_lst):
#     aa_arr = np.array([pin.log3(T.rotation) for T in T_lst])
#     t_arr = np.array([T.translation for T in T_lst])
#     aa_mean = aa_arr.mean(axis=0)
#     t_mean = t_arr.mean(axis=0)
#     return pin.SE3(pin.exp3(aa_mean), t_mean)

# def se3_avg(T_lst):
#     aa_arr = np.array([T.rotation for T in T_lst])
#     t_arr = np.array([T.translation for T in T_lst])
#     aa_mean = aa_arr.mean(axis=0)
#     U, S, Vh = np.linalg.svd(aa_mean)
#     R_mean_proj = U @ Vh
#     t_mean = t_arr.mean(axis=0)
#     return pin.SE3(R_mean_proj, t_mean)

def se3_avg(T_lst):

    avg_q = pin.Quaternion()
    count = 1
    for t in T_lst:
        q = pin.Quaternion(t.rotation)
        avg_q = avg_q.slerp(1/count, q)
        count += 1
    t_mean = np.mean([t.translation for t in T_lst], axis=0)
    return pin.SE3(avg_q, t_mean)
    # aa_arr = np.array([T.rotation for T in T_lst])
    # t_arr = np.array([T.translation for T in T_lst])
    # aa_mean = aa_arr.mean(axis=0)
    # U, S, Vh = np.linalg.svd(aa_mean)
    # R_mean_proj = U @ Vh
    # t_mean = t_arr.mean(axis=0)
    # return pin.SE3(R_mean_proj, t_mean)

def plot_save_trans_rot_errors(T_bt_lst, T_bt_mean, expe_dir):

    t_err = np.array([T_bt.translation - T_bt_mean.translation for T_bt in T_bt_lst])
    rot_err = np.array([pin.log3(T_bt_mean.rotation.T@T_bt.rotation) for T_bt in T_bt_lst])
    t_err_mm = 1000*t_err
    rot_err_deg = np.rad2deg(rot_err)

    x_labels = np.arange(len(T_bt_lst))
    fig = plt.figure('Translation error T_tb')
    for i in range(3):
        plt.plot(x_labels, t_err_mm[:,i], 'rgb'[i]+'x', label='err_t_'+'xyz'[i])
    plt.xlabel('Pose #')
    plt.ylabel('Translation error (mm)')
    plt.savefig(expe_dir / 'translation_tb.jpg')
    plt.close()


    fig = plt.figure('Rotation error T_tb')
    for i in range(3):
        plt.plot(x_labels, rot_err_deg[:,i], 'rgb'[i]+'x', label='err_rot_'+'xyz'[i])
    plt.xlabel('Pose #')
    plt.ylabel('Rotation error (deg)')
    plt.savefig(expe_dir / 'rotation_tb.jpg')
    plt.close()



class RefinementRobotWorldHandEye:
    """
    Class storing calibration data and providing a residual for scipy least_squares
    """


    def __init__(self, T_gc_init, T_bt_init, T_bg_lst, cam_mtx, dist_coef, all_ch_points, all_ch_corners):
        # Initial guess
        self.T_gc_init = T_gc_init
        self.T_bt_init = T_bt_init

        # Camera calibration (fixed)
        self.K = cam_mtx
        self.dist = dist_coef


        # Data 
        self.T_bg_lst = T_bg_lst
        self.T_gb_lst = [T_bg.inverse() for T_bg in T_bg_lst]
        self.pts = all_ch_points
        self.corners = all_ch_corners
        self.n_poses = len(T_bg_lst)
        assert(self.n_poses == len(all_ch_points) == len(all_ch_corners))

        # Compute size of the residual
        self.Nres = sum(2*len(pts_one_pose) for pts_one_pose in self.pts)

    def func(self, x):
        """
        Computes the vector of residuals according to current guess and recorded dataset.
        -> find optimal T_gc and T_bt

        Problem:
        T_gc_opti, T_bt_opti = argmin \sum_i^{nbposes} \sum_j^{nbcorners} ||res(T_c_o, T_c_b*T_b_o)_ij||^2

        with 
        res(T_gc, T_bt)_ij = proj(T_ct_i * pt_ij, K, dist) - c_ij_detected 
        """

        T_gc, T_bt = self.get_transforms(x)
        T_cg = T_gc.inverse()

        res = np.zeros(self.Nres)

        index_ij = 0
        for i in range(self.n_poses):
            T_ct_i =  T_cg * self.T_gb_lst[i] * T_bt
            n_corner_this_pose = len(self.pts[i]) 
            for j in range(n_corner_this_pose):
                rvec = pin.log3(T_ct_i.rotation)
                tvec = T_ct_i.translation 
                cij_proj = cv2.projectPoints(self.pts[i][j], rvec, tvec, self.K, self.dist)[0].squeeze()

                cij_det = self.corners[i][j].squeeze()
                res[index_ij:index_ij+2] = (cij_proj - cij_det)/np.sqrt(self.Nres)

                index_ij += 2  # increment by one corner length

        # res /= np.sqrt(self.Nres)
        return res
        
    def get_transforms(self, x):
        # Optimization is done on SE(3) 
        # -> use a minimal representation of decision variables as local increments "nu" on se3
        nu_gc = x[:6]
        nu_bt = x[6:]

        # Recover corresponding transformation matrices
        T_gc = self.T_gc_init * pin.exp6(nu_gc)
        T_bt = self.T_bt_init * pin.exp6(nu_bt)

        return T_gc, T_bt


def refine_eye_in_hand_calibration(T_gc_init, T_bt_init, T_bg_lst, cam_mtx, dist_coef, all_ch_points, all_ch_corners):
    pbe = RefinementRobotWorldHandEye(T_gc_init, T_bt_init, T_bg_lst, cam_mtx, dist_coef, all_ch_points, all_ch_corners)
    x0 = np.zeros(12)
    result = least_squares(fun=pbe.func, x0=x0, jac='2-point', method='trf', verbose=2, xtol=1e-10)

    T_gc, T_bt = pbe.get_transforms(result.x)
    return T_gc, T_bt

# def calibrate_RobotWorldHandEye(calibration_data, expe_dir, hand_eye_method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH):
#     """
#     !! Does not seem to work!
#     """
#     R_bg_lst = []
#     t_bg_lst = []
#     R_ct_lst = []
#     t_ct_lst = []
#     R_base2gripper = []
#     t_base2gripper = []
#     R_world2cam = []
#     t_world2cam = []

#     transforms_to_save = {}

#     for img_name in calibration_data:
#         # this is the transform read from tf for the gripper in base
#         R_base2gripper.append(np.array(calibration_data[img_name]['r_mtx'])[:3,:3])
#         t_base2gripper.append(np.array(calibration_data[img_name]['t_vec']))

#         #transform from the target to the camera
#         T_world2cam = tft.translation_matrix(np.array(calibration_data[img_name]['t_ct']).squeeze())
#         T_world2cam[:3, :3] = cv2.Rodrigues(np.array(calibration_data[img_name]['rvec_ct']))[0]
#         T_world2cam = np.linalg.pinv(T_world2cam)

#         R_world2cam.append(T_world2cam[:3, :3])
#         t_world2cam.append(T_world2cam[:3,3])

#         # R_world2cam.append(cv2.Rodrigues(np.array(calibration_data[img_name]['rvec_ct']))[0].T)
#         # t_world2cam.append(np.array(calibration_data[img_name]['t_ct']).squeeze() * (-1))


#     for img_name in calibration_data:
#         rvec = np.array(calibration_data[img_name]['rvec_ct'])
#         tvec = np.array(calibration_data[img_name]['t_ct'])
#         R, J = cv2.Rodrigues(rvec)
#         t_ct_lst.append( copy.deepcopy(tvec) )
#         R_ct_lst.append( copy.deepcopy(R) )

#         R = np.array( calibration_data[img_name]['r_mtx'] )
#         R = R[0:3,0:3]
#         R_bg_lst.append( copy.deepcopy(R) )
#         t_bg_lst.append( copy.deepcopy(np.array(calibration_data[img_name]['t_vec'])) )


#     R_gc, t_gc = cv2.calibrateHandEye(R_bg_lst, t_bg_lst, R_ct_lst, t_ct_lst, hand_eye_method)
#     R_base2world, t_base2world, R_gripper2cam, t_gripper2cam  = cv2.calibrateRobotWorldHandEye(R_world2cam, t_world2cam, R_base2gripper, t_base2gripper,method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)

#     tb = tft.translation_matrix(t_base2world.squeeze())
#     tb[:3,:3] = R_base2world[:,:]

#     gripper2cam = tft.translation_matrix(t_gripper2cam.squeeze())
#     gripper2cam[:3,:3] = R_gripper2cam[:,:]

#     gripper2cam2 = tft.translation_matrix(t_gc.squeeze())
#     gripper2cam2[:3,:3] = R_gc[:,:]
#     gripper2cam2 = np.linalg.pinv(gripper2cam)

#     print(gripper2cam2)

#     print(gripper2cam)


#     # save_yaml(expe_dir / 'bt_transform.yaml', transforms_to_save)
#     # result_transforms = {'T_gc': T_gc.tolist(), 'T_tb':T_tb.tolist()}
#     # save_yaml(expe_dir / 'result_transforms.yaml', result_transforms)

#     return gripper2cam2, tb
