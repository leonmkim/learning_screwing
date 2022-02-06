import rosbag
import bagpy
import os

bag_path = '~/datasets/rosbags/input.bag'
bag_path = os.path.expanduser(bag_path)

bag = rosbag.Bag(bag_path)
topics = bag.get_type_and_topic_info()[1].keys()
types = []
for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
    types.append(bag.get_type_and_topic_info()[1].values()[i][0])
print(types)

