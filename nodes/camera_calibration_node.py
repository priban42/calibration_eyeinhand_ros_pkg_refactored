#!/usr/bin/env python3

import sys
import os
os.chdir(os.path.dirname(__file__))
import copy
import rospy
from math import pi
import numpy as np
import tf.transformations as tft
import tf


import moveit_commander
from moveit_commander.conversions import pose_to_list
from sensor_msgs.msg import Image
import moveit_msgs.msg
from geometry_msgs.msg import Pose

import datetime

import yaml

from cv_bridge import CvBridge
import cv2

import sys

SECONDS_WAITING_BEFORE_STARTING = 1.0
SECONDS_WAITING_BEFORE_TAKING_PICTURE = 3.0


def image_callback(data):
    global last_image
    last_image = copy.deepcopy(data)


def get_time_str(form):
    return str(datetime.datetime.now().strftime(form))


def capture_images():

    pos_to_take = open(positions_file,'r')
    calib_data_dir = dir_path+get_time_str("%d-%m-%Y-%H:%M:%S")
    rospy.loginfo("CALIBRATION: Creating directory " + calib_data_dir)
    os.mkdir(calib_data_dir)

    arm.set_max_velocity_scaling_factor(max_speed)
    arm.set_end_effector_link(eef_frame)

    capture_info = {
        'time_of_start': get_time_str("%d-%m-%Y-%H:%M:%S"),
        'used_positions_file': positions_file,
    }

    ## Move to start position
    start_position = pos_to_take.readline()
    start_position = start_position.split()
    start_position = [float(p) for p in start_position]

    if not 'joint' in positions_file:
        start_pose_goal = Pose()
        start_pose_goal.position.x = start_position[0]
        start_pose_goal.position.y = start_position[1]
        start_pose_goal.position.z = start_position[2]
        start_pose_goal.orientation.x = start_position[3]
        start_pose_goal.orientation.y = start_position[4]
        start_pose_goal.orientation.z = start_position[5]
        start_pose_goal.orientation.w = start_position[6]
        rospy.loginfo("CALIBRATION: Setting starting position " + str(start_position))
        try:
            arm.clear_pose_targets()
            arm.set_pose_target(start_pose_goal)
            arm.go()
        except ValueError as e:
            rospy.logfatal("CALIBRATION: " + e)
            rospy.logfatal("CALIBRATION: Exiting!")
            return False
    else:
        rospy.loginfo("CALIBRATION: Setting joint starting position " + str(start_position))
        try:
            arm.go(start_position)
        except ValueError as e:
            rospy.logfatal("CALIBRATION: " + e)
            rospy.logfatal("CALIBRATION: Exiting!")
            return False

    rospy.sleep(SECONDS_WAITING_BEFORE_STARTING)
    for pose_nb, line in enumerate(pos_to_take):
        ## Set robot to position
        position = line.split()
        position = [float(p) for p in position]
        if not 'joint' in positions_file:
            pose_goal = Pose()
            pose_goal.position.x = position[0]
            pose_goal.position.y = position[1]
            pose_goal.position.z = position[2]
            pose_goal.orientation.x = position[3]
            pose_goal.orientation.y = position[4]
            pose_goal.orientation.z = position[5]
            pose_goal.orientation.w = position[6]

            rospy.loginfo("CALIBRATION: Setting position " + line)
            try:
                arm.clear_pose_targets()
                arm.set_pose_target(pose_goal)
                arm.go()
            except ValueError as e:
                rospy.logfatal("CALIBRATION: " + e)
                rospy.logfatal("CALIBRATION: Exiting!")
                return False
        else:
            rospy.loginfo("CALIBRATION: Setting joint position " + line)
            try:
                arm.go(position)
            except ValueError as e:
                rospy.logfatal("CALIBRATION: " + e)
                rospy.logfatal("CALIBRATION: Exiting!")
                return False

        # Wait before capturing the image, in case the trajectory execution is not completely finished
        print(f'Wait for {SECONDS_WAITING_BEFORE_TAKING_PICTURE} before taking picture')
        rospy.sleep(SECONDS_WAITING_BEFORE_TAKING_PICTURE)

        ## Capture image
        rospy.loginfo("CALIBRATION: Capturing eye in hand image")
        global last_image
        cap_image = bridge.imgmsg_to_cv2(last_image, 'bgr8')

        # ## Save image eye
        date_time = get_time_str("%d-%m-%Y-%H:%M:%S")
        if use_time:
            image_name = image_name_prefix + date_time
        else:
        #     # Add some leading zeros to get lexicographically ordered names
            image_name = f'{image_name_prefix}_{pose_nb:03d}'
        # file_name = f'{calib_data_dir}/{image_name}.jpg'
        # rospy.loginfo("CALIBRATION: Saving file " + file_name)
        # cv2.imwrite(file_name, cap_image)

        # if rospy.get_param('/camera_calibration/use_ext_camera'):
        #     try:
        #         rospy.loginfo("CALIBRATION: Capturing external camera image")
        #         ext_image = rospy.wait_for_message(rospy.get_param('/camera_calibration/ext_cam_topic'), Image, timeout=0.5)
        #         ext_cap_image = bridge.imgmsg_to_cv2(ext_image, 'bgr8')
        #     except:
        #         rospy.logfatal("CALIBRATION ERROR: No image captured. Saving black screen!")
        #         ext_cap_image = np.zeros((100,100))
        #
        #     ## Save image ext
        #     date_time = get_time_str("%d-%m-%Y-%H:%M:%S")
        #     if use_time:
        #         image_name_ext = image_name_prefix + '_ext_' + date_time
        #     else:
        #         image_name_ext = image_name_prefix + '_ext_' + str(pose_nb)
        #     file_name_ext = calib_data_dir + '/' + image_name_ext + '.jpg'
        #     rospy.loginfo("CALIBRATION: Saving file " + file_name)
        #     cv2.imwrite(file_name_ext, ext_cap_image)


        rospy.loginfo("CALIBRATION: Images captured and saved")


        ## Create log file
        joint_values = arm.get_current_joint_values()
        pose = arm.get_current_pose()
        trans, rot = listener.lookupTransform(base_frame, eef_frame, last_image.header.stamp)
        print("base_frame, eef_frame, last_image.header.stamp:", base_frame, eef_frame, last_image.header.stamp)
        log_info = {
            'directory': calib_data_dir,
            'image_name': image_name,
            'time': date_time,
            't_vec': copy.deepcopy(trans),
            'r_mtx': tft.quaternion_matrix(rot).tolist(),
            'joint_values': joint_values,
            'image_shape': "list(cap_image.shape)",
            'base_frame': base_frame,
            'eef_frame': eef_frame,
        }

        file_name = calib_data_dir + '/' + image_name + '.yaml'
        rospy.loginfo("CALIBRATION: Saving file " + file_name)
        with open(file_name, 'w') as f:
            yaml.dump(log_info, f)

    rospy.loginfo("CALIBRATION: All images captured!")
    ## Move to start position
    rospy.loginfo("CALIBRATION: Moving to starting position!")
    if not 'joint' in positions_file:
        arm.clear_pose_targets()
        arm.set_pose_target(start_pose_goal)
        arm.go()
    else:
        arm.go(start_position)

    capture_info['time_of_end'] = get_time_str("%d-%m-%Y-%H:%M:%S")
    capture_info['note'] = ''
    image_info_path = os.path.join(calib_data_dir,'image_capture_info.yaml')
    with open(image_info_path, 'w') as f:
        yaml.dump(capture_info, f)
    rospy.loginfo("CALIBRATION: Exiting!")
    return True


if __name__=="__main__":
    rospy.init_node('camera_calibration_node', anonymous=True)

    dir_path = rospy.get_param('camera_calibration/directory')
    group_name = rospy.get_param('camera_calibration/group')
    use_time = rospy.get_param('camera_calibration/use_time_in_image_names')
    image_name_prefix = rospy.get_param('camera_calibration/image_name_prefix')
    positions_file = rospy.get_param('camera_calibration/positions_file')
    max_speed = rospy.get_param('camera_calibration/max_speed')
    image_topic = rospy.get_param('camera_calibration/image_topic')
    base_frame = rospy.get_param('camera_calibration/base_frame')
    eef_frame = rospy.get_param('camera_calibration/eef_frame')
    moveit_commander.roscpp_initialize(sys.argv)

    listener = tf.TransformListener()
    rospy.sleep(1)

    bridge = CvBridge()


    #last_image = np.zeros((100,100))

    last_image = Image()
    last_image.height = 100
    last_image.width = 100
    last_image.data = np.zeros((100,100))
    colour_image_subs = rospy.Subscriber(image_topic, Image, image_callback)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    arm = moveit_commander.MoveGroupCommander(group_name)

    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)


    capture_images()
