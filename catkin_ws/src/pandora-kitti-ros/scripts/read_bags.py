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

from datasetbuilder import DatasetBuider
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
    - Find out how high, relative to the ground the camera sensor is. 
    - Find extrinsic transform between camera and lidar

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


def read_bag(filepath, topics):
    print("Reading bag with filepath ", filepath)
    image_publisher = rospy.Publisher(
        IMAGE_TOPIC, Image, queue_size=0)
    pcl_publisher = rospy.Publisher(PCL_TOPIC, PointCloud2, queue_size=0)
    bag = rosbag.Bag(filepath)

    c = 0
    num_images = 0
    num_pcls = 0
    for topic, msg, t in bag.read_messages(topics=topics):
        if num_images > 10 and num_pcls > 10:
            break
        if topic == IMAGE_TOPIC:
            num_images += 1
            #print("Publishing topic {} at timestamp {}".format(topic, t))
            image_publisher.publish(msg)
        if topic == PCL_TOPIC:
            num_pcls += 1
            #print("Publishing topic {} at timestamp {}".format(topic, t))
            pcl_publisher.publish(msg)
        print("Images: {}, PCLS: {}".format(num_images, num_pcls))

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
    # Set up listener for image and pointclouds
    dc = DataConverter()
    # Publish image and pointcloud topics
    read_bag(args.filepath, topics=topics)

    try:
        rospy.spin()
    except KeyboardInterrupt as e:
        print(e)
