#!/usr/bin/env python2
import rosbag
import argparse
from cv_bridge import CvBridge, CvBridgeError
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import logging
import roslib
import numpy as np
import os
from camera_intrinsics import K

"""
Reads ros bags from a filepath and publishes the data to a topic.
This is useful when we already have gathered data and want to simulate actual real-time recordings.
The topics of the bag can be listed by typing rosbag info <your bagfile> in cmd.
The command will yield some info about the rosbag, for example length of recording, size of bag etc.
The specific topics for pandora are listed as
types:       rosgraph_msgs/Log       [acffd30cd6b6de30f120938c17c593fb]
             sensor_msgs/Image       [060021388200f6f0f447d0fcd9c64743]
             sensor_msgs/PointCloud2 [1158d486dd51d683ce2f1be655c3c181]
topics:      /pandora/sensor/pandora/camera/back_gray      5709 msgs    : sensor_msgs/Image
             /pandora/sensor/pandora/camera/front_color    5710 msgs    : sensor_msgs/Image
             /pandora/sensor/pandora/camera/front_gray     5710 msgs    : sensor_msgs/Image
             /pandora/sensor/pandora/camera/left_gray      5709 msgs    : sensor_msgs/Image
             /pandora/sensor/pandora/camera/right_gray     5710 msgs    : sensor_msgs/Image
             /pandora/sensor/pandora/hesai40/PointCloud2   5709 msgs    : sensor_msgs/PointCloud2
             /rosout                                         10 msgs    : rosgraph_msgs/Log       (2 connections)


TODO: 
    - Time synchronization between topics (http://wiki.ros.org/message_filters)
    - Subscribing to calibration info from pandora (https://github.com/HesaiTechnology/Pandora_Apollo/blob/master/src/pandora.cc#L263)
    - Writing groundplane information to file
    - Writing camera intrinsic parameters to file

"""

DATA_PATH = "../../../../_out"
IMAGE_TOPIC = "/pandora/sensor/pandora/camera/front_color"
PCL_TOPIC = "/pandora/sensor/pandora/hesai40/PointCloud2"
topics = [IMAGE_TOPIC, PCL_TOPIC]


class DataConverter:
    """ Converts image and point cloud data from rosbags to .png and .bin files
        so that we can run inference on them.
    """

    def __init__(self):
        self.img_count = 0
        self.lidar_count = 0
        self.db_builder = DatasetBuider(out_directory=DATA_PATH)
        # self.image_publisher = rospy.Publisher(
        #    IMAGE_TOPIC, Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_subscriber = rospy.Subscriber(
            IMAGE_TOPIC, Image, self.image_callback)

        self.pcl_subscriber = rospy.Subscriber(
            PCL_TOPIC, PointCloud2, self.pointcloud_callback)

    def image_callback(self, imgmsg):
        """ Convert the image from imgmsg representation to png format,
            then publish it back on topic.
        """
        try:
            image = self.bridge.imgmsg_to_cv2(imgmsg, "bgr8")
            self.db_builder.add_image(image)
        except CvBridgeError as e:
            logging.warning(e)
        # Publish back the image so that we can do inference on it later
        # try:
        #    self.image_publisher.publish(
        #        self.bridge.cv2_to_imgmsg(image, "bgr8"))
        # except CvBridgeError as e:
        #    logging.warning(e)

    def pointcloud_callback(self, pointcloud2_array):
        """
            Convert the pointcloud, which is represented as a PointCloud2 to an xyz array
            that our model uses for inference. Then publish this converted pointcloud back on topic.
        """
        self.db_builder.add_pointcloud(pointcloud2_array)


class DatasetBuider:
    """ Builds the dataset in kitti format so that our models can do inference on them """

    def __init__(self, out_directory, pcl_format="bin"):
        self.lidar_path = os.path.join(out_directory, "velodyne")
        self.image_path = os.path.join(out_directory, "image_2")
        self.label_path = os.path.join(out_directory, "label_2")
        self.calib_path = os.path.join(out_directory, "calib")
        self.groundplane_path = os.path.join(out_directory, "planes")

        self.image_file_fmt = "{}.png"
        self.lidar_file_fmt = "{}."+pcl_format
        self.lidar_count = 0
        self.image_count = 0

        self._init_dirs()

        self.K = K

    def _init_dirs(self):
        """ Creates all the directories necessary for the kitti file format if they do not already exist """
        dirs = [self.lidar_path, self.image_path, self.label_path,
                self.calib_path, self.groundplane_path]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

    def add_pointcloud(self, pointcloud, pcl_format="bin"):
        """Save this point-cloud to disk as PLY or bin format."""
        filename = self.lidar_file_fmt.format(self.lidar_count)
        lidar_filepath = os.path.join(self.lidar_path, filename)
        if pcl_format == "bin":
            # Points are of the format
            # x, y, z, intensity, timestamp, ring
            pointcloud = np.array([[x, y, z, intensity] for x, y, z, intensity,
                                   _, _ in pc2.read_points(pointcloud, skip_nans=True)])
            pointcloud.astype(np.float32).tofile(lidar_filepath)
            self.lidar_count += 1
        elif pcl_format == "ply":
            points = [[x, y, z] for x, y, z, _, _, _ in pc2.read_points(
                pointcloud, skip_nans=True)]
            ply = '\n'.join(['{:.2f} {:.2f} {:.2f}'.format(*p)
                             for p in points])
            num_points = len(points)
            # Open the file and save with the specific PLY format.
            with open(filename, 'w+') as ply_file:
                ply_file.write(
                    '\n'.join([self._construct_ply_header(num_points), ply]))
            self.lidar_count += 1
        else:
            raise ValueError("Format must be of type 'bin' or 'ply'")

    def add_image(self, image):
        """ Save image to dataset in png format """
        filename = self.image_file_fmt.format(self.image_count)
        image_filepath = os.path.join(self.image_path, filename)
        cv2.imwrite(image_filepath, image)
        self.image_count += 1

    def _save_calibration_matrices(self):
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


def read_bag(filepath, topics):
    print("Reading bag with filepath ", filepath)
    image_publisher = rospy.Publisher(
        IMAGE_TOPIC, Image, queue_size=1)
    pcl_publisher = rospy.Publisher(PCL_TOPIC, PointCloud2, queue_size=1)
    bag = rosbag.Bag(filepath)

    c = 0
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == IMAGE_TOPIC:
            print("Publishing topic {} at timestamp {}".format(topic, t))
            image_publisher.publish(msg)
        if topic == PCL_TOPIC:
            print("Publishing topic {} at timestamp {}".format(topic, t))
            pcl_publisher.publish(msg)

    bag.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", help="Filepath to the ros bag",
                        default="/lhome/aasmunhb/Masteroppgave/pandora-kitti-ros/pandora_recording/1-02-2019/2019-02-01-11-18-09.bag")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    topics = ["/pandora/sensor/pandora/camera/front_color",
              "/pandora/sensor/pandora/hesai40/PointCloud2"]
    rospy.init_node('data_converter', anonymous=True)
    dc = DataConverter()
    read_bag(args.filepath, topics=topics)

    try:
        rospy.spin()
    except KeyboardInterrupt as e:
        print(e)
