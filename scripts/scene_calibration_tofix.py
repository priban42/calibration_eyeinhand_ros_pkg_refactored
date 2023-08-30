
def calibrate_scene_cam(path, board, result_directory, result_name, hand_eye_method=cv2.CALIB_HAND_EYE_TSAI, animate=False, calib_data=None):
    ext_path = os.path.join(path, 'ext')
    images = get_images_paths(ext_path)

    result_directory = join(result_directory, get_time_str("%d-%m-%Y-%H:%M:%S"))
    os.makedirs(result_directory, exist_ok=True)
    camera_result_directory = join(result_directory, 'ext_camera_calibration')
    # os.mkdir(camera_result_directory, exists_ok=True)
    Path(camera_result_directory).mkdir(parents=True, exist_ok=True)

    # calibrate intrinsics
    cam_mtx, dist_coef, rvecs, tvecs, unused_images = calibrate_camera(images, ext_path, board,
                                                                       markers_directory, animate)
    
    # Store camera calibration
    camera_data = {'camera_matrix': cam_mtx.tolist(),
                   'distortion_coefficients': dist_coef.tolist()}

    i = 0
    for im in images:
        camera_data[os.path.splitext(im)[0]] = {}
        camera_data[os.path.splitext(im)[0]]['rvec'] = rvecs[i].tolist()
        camera_data[os.path.splitext(im)[0]]['tvec'] = tvecs[i].tolist()
        i += 1

    save_yaml(join(camera_result_directory, 'camera_result.yaml'), camera_data)
    
    if cam_mtx is None:
        return None

    for im in unused_images:
        images.remove(im)

    camera_reprojection_errors = reprojection_error(images, ext_path, tvecs, rvecs, board, cam_mtx, dist_coef,
                                                    camera_result_directory, 10, animate)
    print("Camera reprojection mean error: ", np.mean(camera_reprojection_errors), " px.")
    print("Camera reprojection min error: ", min(camera_reprojection_errors), " px.")
    print("Camera reprojection max error: ", max(camera_reprojection_errors), " px.")


    # Prepare data for Eye In Hand calibration
    # calibration_data = {}
    # for i in range(len(images)):
    #     yml = str(os.path.splitext(images[i])[0]) + '.yaml'
    #     f = open(join(ext_path, yml), 'r')
    #     data = yaml.load(f, yaml.SafeLoader)
    #     f.close()
    #     data['rvec'] = rvecs[i].tolist()
    #     data['tvec'] = tvecs[i].tolist()
    #     calibration_data[os.path.splitext(yml)[0]] = data

    rvec = np.mean(rvecs, axis=0)
    tvec = np.mean(tvecs, axis=0)

    cam2board = tft.translation_matrix(tvec.squeeze())
    # rotmat = tft.euler_matrix(*rvec.squeeze())
    rotmat = cv2.Rodrigues(rvec)[0]

    cam2board[:3,:3] = rotmat[:3,:3]

    if calib_data is not None:
        base2board = calib_data['base_target_transform']
        base2cam = base2board @ np.linalg.pinv(cam2board)
        print('still here')

    # Save Camera calibration
    camera_intrinsic = {}
    # TODO: do not assume image shape
    camera_intrinsic['image_width'] = 640  #data['image_shape'][1]
    camera_intrinsic['image_height'] = 480 #data['image_shape'][0]
    camera_intrinsic['camera_name'] = 'camera'
    camera_intrinsic['distortion_model'] = 'plumb_bob'
    camera_intrinsic['camera_matrix'] = {'rows':3,'cols':3, 'data':cam_mtx.reshape(1,9)[0].tolist()}
    camera_intrinsic['distortion_coefficients'] = {'rows':1,'cols':5, 'data':dist_coef[0].tolist()}
    camera_intrinsic['rectification_matrix'] = {'rows':3,'cols':3, 'data':np.eye(3,3).reshape(1,9)[0].tolist()}
    camera_intrinsic['projection_matrix'] = {'rows':3,'cols':4, 'data':np.hstack((cam_mtx,[[0],[0],[0]])).reshape(1,12)[0].tolist()}
    save_yaml(join(camera_result_directory,'camera.yaml'), camera_intrinsic)
