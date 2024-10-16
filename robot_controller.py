import mujoco as mj
from mujoco.glfw import glfw
import h5py
import numpy as np
import time  # Import time module
import OpenGL.GL as GL


class RobotController:
    def __init__(self, data, model, hdf5_file_path):
        self.data = data
        self.model = model

        if hdf5_file_path is not None:
            self.use_demo = True
            # Load the HDF5 file containing desired qpos observations
            self.hdf5_file = h5py.File(hdf5_file_path, 'r')
            # Actions are stored under 'action/joint_pos'
            self.actions = self.hdf5_file['action']['joint_pos'][2:]
            print(self.actions.shape)
            self.num_steps = self.actions.shape[0]

        else: 
            self.use_demo = False
            self.actions = [[-0.02914563,  0.28071848, -0.68875737,  0.58904862, -0.23316508,  0.08130098, 0, -0.02914563,  0.36201947, -0.76852437,  0.65654378,  0.26691266, -0.17487381, 0]]
            self.num_steps = 800

        self.current_step = 0
        self.threshold = 0.001

        # Get the total number of steps and the number of joints
        self.num_joints = 12  # Number of joints to control | previously 6
        self.num_actuators = self.model.nu  # Number of actuators (controls)
        print("Number of joints: ", self.num_joints)
        print("Number of actuators: ", self.num_actuators)

        # Set initial position as the first step from the HDF5 demo file
        self.set_initial_position()

        self.error = []  # Store the error for each step

    def set_initial_position(self):
        # Get the initial desired qpos from the first step
        initial_qpos = self.actions[0]

        # Ensure that initial_qpos has the correct size
        
        # initial_qpos = initial_qpos[7:7+self.num_joints]
        initial_qpos = np.delete(initial_qpos, [6, 13])
        print("Initial qpos: ", initial_qpos, initial_qpos.shape)

        # Set the initial qpos in the simulation
        self.data.qpos[:self.num_joints] = initial_qpos

    def control(self):
        # Read the current qpos from the simulation
        current_qpos = self.data.qpos.copy() # (20, ) = 6 + 4 + 6 + 4
        current_qpos = np.delete(current_qpos, [6, 7, 8, 9, 16, 17, 18, 19])  # Shape (12, )
        print("Current qpos shape:", current_qpos.shape)

        # Check if there are more desired qpos to follow
        if self.current_step < self.num_steps:
            # Get the desired qpos from the observations
            if not self.use_demo:
                desired_qpos = self.actions[0]
            else: 
                desired_qpos = self.actions[self.current_step]

            # Ensure that desired_qpos has the correct size
            # desired_qpos = desired_qpos[7:7+self.num_joints]
            desired_qpos = np.delete(desired_qpos, [6, 13])  # Shape (12, )
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
            control_input = control_input[:self.num_actuators]

            # add gripper commands 
            if self.use_demo:
                if self.actions[self.current_step, 6] == 4.0:
                    control_input = np.insert(control_input, 6, -1)
                else: 
                    control_input = np.insert(control_input, 6, 0)

                if self.actions[self.current_step, 13] == 4.0:
                    control_input = np.insert(control_input, 13, -1)
                else: 
                    control_input = np.insert(control_input, 13, 0)
            else: 
                control_input = np.insert(control_input, 6, 0)
                control_input = np.insert(control_input, 13, 0)

            # Update data.ctrl with the computed control inputs
            self.data.ctrl[:] = control_input

            # Move to the next step
            self.current_step += 1
        else:
            # No more observations; stop the robot by zeroing control inputs
            self.data.ctrl[:] = 0.0


class BaseWindow:
    def __init__(self, xml_path, hdf5_file_path=None):
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

        # Initialize the RobotController with the simulation data and HDF5 file path
        self.controller = RobotController(
            self.data, self.model, hdf5_file_path)

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
                
            # Update control inputs based on the controller
            # Step the simulation forward
            # Update the scene and render

            if self.controller.current_step < 850:
                self.controller.control()

                mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                mj.mjtCatBit.mjCAT_ALL.value, self.scene)
                
                mj.mjr_render(viewport, self.scene, self.context)
                frame = self.read_screen()
                print(type(frame), frame.shape)
                frames.append(frame)
                time.sleep(0.1)
            else: 
                glfw.swap_buffers(self.window)
                glfw.poll_events()

                # import cv2 
                # import numpy as np 
                # from tqdm import tqdm

                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # # change the output file name 
                # out = cv2.VideoWriter('demo_35_sim.mp4', fourcc, 10.0, (viewport_width, viewport_height))

                # for frame in tqdm(frames): 
                #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #     out.write(frame)  

                # out.release() 
                break   

            

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
    import os 
    xml_path = os.path.join(os.getcwd(), "urdf/robot_dual.xml")
    hdf5_file_path = "./sample_demo.hdf5"  # Replace with the actual path to your HDF5 file
    window = BaseWindow(xml_path, None)
    window.simulate()
