
from camera_intrinsics import K
import os
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import cv2
import logging


class DatasetBuider:
    """ Builds the dataset in kitti format so that our models can do inference on them """

    def __init__(self, out_directory, pcl_format="bin"):
        self.lidar_path = os.path.join(out_directory, "velodyne")
        self.image_path = os.path.join(out_directory, "image_2")
        self.label_path = os.path.join(out_directory, "label_2")
        self.calib_path = os.path.join(out_directory, "calib")
        self.groundplane_path = os.path.join(out_directory, "planes")
        self.out_directory = out_directory
        self.image_file_fmt = "{0:06}.png"
        self.lidar_file_fmt = "{0:06}."+pcl_format
        self.lidar_count = 0
        self.image_count = 0

        self._init_dirs()
        self.K = K

    def _init_dirs(self):
        """ Creates all the directories necessary for the kitti file format if they do not already exist """
        dirs = [self.lidar_path, self.image_path, self.label_path,
                self.calib_path, self.groundplane_path, "velodyne_reduced"]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

    def add_pointcloud(self, pointcloud, pcl_format="bin", skip_nans=True):
        """Save this point-cloud to disk as PLY or bin format.
            In addition, this call triggers the save_calibration_matrices function, since
            we need to store calibration matrices for each frame.
            Pandora by default follows the FLU-system x: forward, left, up (CHECK THIS)
            NOTE: When saving to .bin, we flip the y-axis to get the same coordinate system as KITTI.
        """
        filename = self.lidar_file_fmt.format(self.lidar_count)
        lidar_filepath = os.path.join(self.lidar_path, filename)
        if pcl_format not in ["bin", "ply"]:
            raise ValueError("Format must be of type 'bin' or 'ply'")
        if pcl_format == "bin":
            # Points are of the format
            # x, y, z, intensity, timestamp, ring
            # Flip the points from x, y,z to -y, x, z
            pointcloud = np.array([[-y, x, z, intensity] for x, y, z, intensity,
                                   _, _ in pc2.read_points(pointcloud, skip_nans=skip_nans)])
            pointcloud.astype(np.float32).tofile(lidar_filepath)
        elif pcl_format == "ply":
            points = [[x, y, z] for x, y, z, _, _, _ in pc2.read_points(
                pointcloud, skip_nans=skip_nans)]
            ply = '\n'.join(['{:.2f} {:.2f} {:.2f}'.format(*p)
                             for p in points])
            num_points = len(points)
            # Open the file and save with the specific PLY format.
            with open(filename, 'w+') as ply_file:
                ply_file.write(
                    '\n'.join([self._construct_ply_header(num_points), ply]))
        self._save_calibration_matrices(self.lidar_count)
        self.append_datafile(self.lidar_count)
        self.lidar_count += 1

    def add_image(self, image):
        """ Save a cropped version of the image to dataset in png format 
            The original resolution is 1280x720.
        """
        filename = self.image_file_fmt.format(self.image_count)
        image_filepath = os.path.join(self.image_path, filename)
        RESOLUTION_X, RESOLUTION_Y = 1280, 720
        TARGET_RESOLUTION_X, TARGET_RESOLUTION_Y = 1224, 370
        # Find out how much we need to discard at each side
        CROP_X = (RESOLUTION_X - TARGET_RESOLUTION_X) // 2
        CROP_Y = (RESOLUTION_Y - TARGET_RESOLUTION_Y) // 2
        cropped_img = image[CROP_Y:-CROP_Y, CROP_X:-CROP_X]
        print("Saving image with shape {} to filepath {}".format(
            cropped_img.shape, image_filepath))
        cv2.imwrite(image_filepath, cropped_img)
        self.image_count += 1

    def _save_calibration_matrices(self, frame_no):
        """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                            and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                            from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                        coordinate frame according to:
                                        Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                        imu data.
        """
        filename = os.path.join(self.calib_path, "{0:06}.txt".format(frame_no))
        # KITTI format demands that we flatten in row-major order
        ravel_mode = 'C'
        P0 = self.K
        P0 = np.column_stack((P0, np.array([0, 0, 0])))
        P0 = np.ravel(P0, order=ravel_mode)
        R0 = np.identity(3)
        TR_velodyne = np.array([[0, -1, 0],
                                [0, 0, -1],
                                [1, 0, 0]])
        # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.
        TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))
        TR_imu_to_velo = np.identity(3)
        TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

        def write_flat(f, name, arr):
            f.write("{}: {}\n".format(name, ' '.join(
                map(str, arr.flatten(ravel_mode).squeeze()))))

        # All matrices are written on a line with spacing
        with open(filename, 'w') as f:
            for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
                write_flat(f, "P" + str(i), P0)
            write_flat(f, "R0_rect", R0)
            write_flat(f, "Tr_velo_to_cam", TR_velodyne)
            write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
        logging.info("Wrote all calibration matrices to %s", filename)

    def _construct_ply_header(self, num_points):
        """Generates a PLY header given a total number of 3D points and
        coloring property if specified
        """

        header = ['ply',
                  'format ascii 1.0',
                  'element vertex {}',
                  'property float32 x',
                  'property float32 y',
                  'property float32 z',
                  'property uchar diffuse_red',
                  'property uchar diffuse_green',
                  'property uchar diffuse_blue',
                  'end_header']

        return '\n'.join(header[0:6] + [header[-1]]).format(num_points)

    def append_datafile(self, idx):
        filename = os.path.join(self.out_directory, "test.txt")
        with open(filename, "a") as f:
            f.write("{0:06}".format(idx))
