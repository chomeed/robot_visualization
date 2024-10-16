import mujoco as mj
from mujoco.glfw import glfw
import h5py
import numpy as np
import time  # Import time module
import OpenGL.GL as GL

from real_robot_env import Z1RobotEnv


class RobotController:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.env = Z1RobotEnv()

        self.current_step = 0
        self.threshold = 0.001

        self.num_joints = 14 # Number of joints to control | previously 6
        self.num_actuators = self.model.nu  # Number of actuators (controls)
        print("Number of joints: ", self.num_joints)
        print("Number of actuators: ", self.num_actuators)

        self.error = []  # Store the error for each step

    # def set_initial_position(self):
    #     # Get the initial desired qpos from the first step
    #     initial_qpos = self.actions[0]

    #     # Ensure that initial_qpos has the correct size
        
    #     # initial_qpos = initial_qpos[7:7+self.num_joints]
    #     initial_qpos = np.delete(initial_qpos, [6, 13])
    #     print("Initial qpos: ", initial_qpos, initial_qpos.shape)

    #     # Set the initial qpos in the simulation
    #     self.data.qpos[:self.num_joints] = initial_qpos

    def control(self):
        # Read the current qpos from the simulation
        current_qpos = self.data.qpos.copy() # (20, ) = 6 + 4 + 6 + 4
        current_qpos = np.delete(current_qpos, [6, 7, 8, 9, 16, 17, 18, 19])  # Shape (12, )
        current_qpos = current_qpos[:12]
        print("Current qpos shape:", current_qpos.shape)

        # Get the desired qpos from the observations
        self.action = np.concatenate([self.env.action['left_qpos'], self.env.action['right_qpos']])
        desired_qpos = self.action
        print("Desired qpos shape:", desired_qpos.shape)

        # Ensure that desired_qpos has the correct size
        # desired_qpos = desired_qpos[7:7+self.num_joints]
        # desired_qpos = np.delete(desired_qpos, [6, 13])  # Shape (12, )
        # print(desired_qpos, current_qpos)

        # Calculate the difference between desired and current qpos
        qpos_error = desired_qpos - current_qpos[:self.num_joints] # NOTE: The current qpos includes the gripper joints!
        # print(qpos_error)
        self.error.append(qpos_error)
        zero_mask = np.abs(qpos_error) < self.threshold

        # Implement a simple proportional controller to compute control inputs
        kp = 0.7  # Proportional gain, adjust as needed for responsiveness
        control_input = kp * qpos_error
        control_input[zero_mask] = 0.0
        print("Control Input:", control_input)
        print("----")

        # Ensure control_input matches the number of actuators
        control_input = control_input

        # add gripper commands 
        if self.env.action['left_gripper_command'] == 4.0:
            control_input = np.insert(control_input, 6, -1)
        else: 
            control_input = np.insert(control_input, 6, 0)

        if self.env.action['right_gripper_command'] == 4.0:
            control_input = np.insert(control_input, 13, -1)
        else: 
            control_input = np.insert(control_input, 13, 0)


        # Update data.ctrl with the computed control inputs
        self.data.ctrl[:] = control_input

        # Move to the next step
        self.current_step += 1


class BaseWindow:
    def __init__(self, xml_path):
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)

        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        # self.model.opt.timestep = 0.002
        print(self.model.opt.timestep)

        glfw.init()
        self.window = glfw.create_window(1200, 900, "Robot Simulation", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(
            self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        self.controller = RobotController(
            self.data, self.model)

    def read_screen(self):
        # Get the viewport size
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        width, height = viewport_width, viewport_height

        # Read pixel data from the framebuffer
        pixel_data = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)

        # Convert pixel data to a NumPy array
        image = np.frombuffer(pixel_data, dtype=np.uint8).reshape(height, width, 3)

        # Flip the image vertically
        image = np.flipud(image)

        return image

    def simulate(self):
        # Get the viewport size
        frames = []  # Store rendered frames
        viewport_width, viewport_height = glfw.get_framebuffer_size(
            self.window)
        # viewport_width, viewport_height = 1200, 900
        # print(640, 480)

        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        while not glfw.window_should_close(self.window):
            simstart = self.data.time
            
            while (self.data.time - simstart < 1.0 / 20):
                mj.mj_step(self.model, self.data)
                
            self.controller.control()

            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            
            mj.mjr_render(viewport, self.scene, self.context)
            frame = self.read_screen()
            print(type(frame), frame.shape)
            # frames.append(frame)
            # time.sleep(0.1)

            # Swap buffers and poll for events
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()
        import matplotlib.pyplot as plt
        error = np.array(self.controller.error)
        print(error.shape)
        plt.plot(error)
        # plt.show()


if __name__ == "__main__":
    xml_path = "/home/univ/workspace/visualization/urdf/robot_dual.xml"
    window = BaseWindow(xml_path)
    window.simulate()
