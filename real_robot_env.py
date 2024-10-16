import dm_env
from absl import logging

import rclpy 
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Bool

import numpy as np
import threading
import time
from visualize_utils import window
import random


class Z1RobotEnv(dm_env.Environment):
    def __init__(self):
        rclpy.init() # initialize ROS2 node
        self._node = rclpy.create_node('z1_robot_env_node')
        self._subscriber_bringup()

        self.init_tasks()

        self.num_tasks = len(self.task_idxs)

        self.thread_done = False
        self.hz = 20 # control frequency
        
        self.thread = PeriodicThread(1/self.hz, self.timer_callback)
        self.thread.start()

        self.timer_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self.timer_thread.start()

        logging.set_verbosity(logging.INFO)
        logging.info('Z1RobotEnv successfully initialized.')

        # self.window = window('ENV Observation', (640, 480))

    def init_tasks(self):
        self.tasks = [
            'pick up the cup and put it in the basket',
            'pick up the green bowl and put it in the basket',
            'pick up the pink bowl and put it in the basket',
        ]

        self.task_idxs = list(range(len(self.tasks)))
        random.shuffle(self.task_idxs)
        self.start = False
        self.curr_task = 0
        msg = Bool()
        msg.data = False
        self.sync_pub.publish(msg)

    def reset(self):
        while not self.thread_done:
            time.sleep(0.01)
            continue
        self.thread_done = False
        self.init_tasks()
        # print("toc")

        return dm_env.restart(observation=self._observation())

    def step(self, action):
        '''
        Note: [action] is stored by the TFDS backend writer.
        '''
        while not self.thread_done:
            time.sleep(0.01)
            continue
        self.thread_done = False
        # print("toc")

        return dm_env.transition(reward=0.0, observation=self._observation())

    def ros_close(self):
        self.thread.stop()
        self.timer_thread.stop()
        self._node.destroy_node()
        rclpy.shutdown()

    def get_action(self):
        return self.action.copy()

    def _subscriber_bringup(self):
        '''
        Note: This function creates all the subscribers \
              for reading joint and gripper states.
        '''
        ###### Initial Setup ##### 
        self.obs = {} 
        self.action = {} 
        self.sync = False 

        ###### ACTION ######
        # rexel joint commands -> action (7, ) + (7, )
        self._node.create_subscription(Float64MultiArray, '/right/joint_command', self.right_joint_command_callback, 10) # 10 is the queue size(history depth)
        self._node.create_subscription(Float64MultiArray, '/left/joint_command', self.left_joint_command_callback, 10)
        self.action['right_qpos'] = np.zeros(shape=(6,), dtype=np.float64)
        self.action['left_qpos'] = np.zeros(shape=(6,), dtype=np.float64)

        self._node.create_subscription(Float64MultiArray, '/right/rexel/pose_states', self.right_pose_command_callback, 10)
        self._node.create_subscription(Float64MultiArray, '/left/rexel/pose_states', self.left_pose_command_callback, 10)
        self.action['right_pose'] = np.zeros(shape=(6,), dtype=np.float64)
        self.action['left_pose'] = np.zeros(shape=(6,), dtype=np.float64)

        # self._node.create_subscription(JointState, '/right/gripper_command', self.right_gripper_command_callback, 10)
        # self._node.create_subscription(JointState, '/left/gripper_command', self.left_gripper_command_callback, 10)
        self.action['right_gripper_command'] = np.zeros(shape=(1,), dtype=np.float64)
        self.action['left_gripper_command'] = np.zeros(shape=(1,), dtype=np.float64)

        ###### OBSERVATION ######
        # image 
        self._node.create_subscription(Image, '/camera/camera/color/image_rect_raw', self.image_callback, 10)
        self.obs['image'] = np.zeros(shape=(480, 640, 3), dtype=np.uint8)

        # arm joint states 
        self._node.create_subscription(JointState, '/right/arm/joint_states', self.right_joint_states_callback, 10)
        self._node.create_subscription(JointState, '/left/arm/joint_states', self.left_joint_states_callback, 10)
        self.obs['right_qpos'] = np.zeros(shape=(6,), dtype=np.float64)
        self.obs['left_qpos'] = np.zeros(shape=(6,), dtype=np.float64)

        # arm pose states
        self._node.create_subscription(Float64MultiArray, '/right/pose_states', self.right_pose_callback, 10)
        self._node.create_subscription(Float64MultiArray, '/left/pose_states', self.left_pose_callback, 10)
        self.obs['right_pose'] = np.zeros(shape=(6,), dtype=np.float64)
        self.obs['left_pose'] = np.zeros(shape=(6,), dtype=np.float64)

        # gripper joint states
        self._node.create_subscription(JointState, '/right/gripper/joint_states', self.right_gripper_joint_states_callback, 10)
        self._node.create_subscription(JointState, '/left/gripper/joint_states', self.left_gripper_joint_states_callback, 10)
        self.obs['right_gripper_state'] = np.zeros(shape=(1,), dtype=np.float64)
        self.obs['left_gripper_state'] = np.zeros(shape=(1,), dtype=np.float64)

        self.obs['language_instruction'] = ''

        ##### TRIGGER ##### 
        self.sync_pub = self._node.create_publisher(Bool, '/sync', 10)
        self._node.create_subscription(Bool, '/done', self.done_callback, 10)

    # callback functions 
    #### ACTIONS ########
    def right_joint_command_callback(self, msg):
        self.action['right_qpos'] = np.array(msg.data[:6])
        if msg.data[6] >= 1.0:
            grp = 4.0
        else:
            grp = 0.0
        self.action['right_gripper_command'] = np.array([grp])

    def left_joint_command_callback(self, msg):
        self.action['left_qpos'] = np.array(msg.data[:6])
        if msg.data[6] >= 1.0:
            grp = 4.0
        else:
            grp = 0.0
        self.action['left_gripper_command'] = np.array([grp])

    def right_pose_command_callback(self, msg):
        # right_xyz_rpy = np.concatenate([np.array(msg.data[:3]), np.flip(msg.data[3:])])
        self.action['right_pose'] = np.array(msg.data)

    def left_pose_command_callback(self, msg):
        left_xyz_rpy = np.concatenate([np.array(msg.data[:3]), np.flip(msg.data[3:])])
        self.action['left_pose'] = np.array(msg.data)

    #### OBS ###########

    def image_callback(self, msg):
        # self._node.get_logger().info('img cb!@')
        self.obs['image'] = np.reshape(msg.data, (480, 640, 3))

    def right_joint_states_callback(self, msg):
        self.obs['right_qpos'] = np.array(msg.position)

    def left_joint_states_callback(self, msg):
        self.obs['left_qpos'] = np.array(msg.position)

    def right_pose_callback(self, msg):
        # right_xyz_rpy = np.concatenate([np.array(msg.data[:3]), np.flip(msg.data[3:])])
        self.obs['right_pose'] = np.array(msg.data)

    def left_pose_callback(self, msg):
        # left_xyz_rpy = np.concatenate([np.array(msg.data[:3]), np.flip(msg.data[3:])])
        self.obs['left_pose'] = np.array(msg.data)

    def right_gripper_joint_states_callback(self, msg):
        self.obs['right_gripper_state'] = np.array(msg.position)

    def left_gripper_joint_states_callback(self, msg):
        self.obs['left_gripper_state'] = np.array(msg.position)

    def done_callback(self, msg):
        # self.sync = msg.data
        if not self.start:
            self.start = True
            msg = Bool()
            msg.data = True
            self.sync_pub.publish(msg)
        else:
            if self.curr_task == self.num_tasks - 1:
                self.start = False
                msg = Bool()
                msg.data = False
                self.sync_pub.publish(msg)
            else:
                self.curr_task += 1

    def timer_callback(self):
        self.thread_done = True
        if self.tasks:
            self.obs['language_instruction'] = self.tasks[self.task_idxs[self.curr_task]]

    # end of callback functions

    def observation_spec(self):
        observation_spec = {} 
        for key, value in self.obs.items():
            if key == 'language_instruction':
                continue
            observation_spec[key] = dm_env.specs.Array(shape=value.shape, dtype=value.dtype)
        return observation_spec    

    def action_spec(self):
        action_spec = {}
        for key, value in self.action.items():
            action_spec[key] = dm_env.specs.Array(shape=value.shape, dtype=value.dtype)
        return action_spec 

    def print_action_and_obs(self):
        print("Action:", self.action)
        print("Observation:", self.obs)

    def _observation(self): 
        return self.obs.copy()


class PeriodicThread(threading.Thread):
    def __init__(self, interval, function, *args, **kwargs):
        super().__init__()
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.stop_event = threading.Event()
        self._lock = threading.Lock()

    def run(self):
        while not self.stop_event.is_set():
            start_time = time.time()
            self.function(*self.args, **self.kwargs)
            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.interval - elapsed_time)
            time.sleep(sleep_time)

    def stop(self):
        self.stop_event.set()

    def change_period(self, new_interval):
        with self._lock:  
            self.interval = new_interval


if __name__ == '__main__':
    env = Z1RobotEnv()

    while True: 
        continue

    env.close()