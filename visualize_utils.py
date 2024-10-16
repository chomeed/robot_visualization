import cv2
import pyautogui

screen_width, screen_height = pyautogui.size()

class window:
    def __init__(self, name, size=None):
        self.screen_width, self.screen_height = pyautogui.size()
        if size is not None:
            self.width = size[0]
            self.height = size[1]
        else:
            self.width = self.screen_width
            self.height = self.screen_height

        self.x_pos = (self.screen_width - self.width) // 2
        self.y_pos = (self.screen_height - self.height) // 2
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.width, self.height)
        cv2.moveWindow(self.name, self.x_pos, self.y_pos)

    def show(self, img, text=''):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        font_color = (0, 0, 0)  # Black color
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        image_height, image_width = img.shape[:2]
        x = (image_width - text_width) // 2  # X coordinate
        y = (image_height - text_height)
        cv2.rectangle(img, (0, image_height - int(2.2 * text_height)), (image_width, image_height), (255, 255, 255), -1)
        cv2.putText(img, text, (x, y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)
        img = cv2.resize(img, (self.width, self.height))
        # print(img.shape)
        cv2.imshow(self.name, img)
        cv2.waitKey(1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate rotation matrix from RPY (roll, pitch, yaw)
def rpy_to_rotation_matrix(roll, pitch, yaw):
    # Rotation matrix around X axis (roll)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    # Rotation matrix around Y axis (pitch)
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    # Rotation matrix around Z axis (yaw)
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))  # R = Rz * Ry * Rx
    return R

# Function to visualize the position and orientation
def visualize_xyz_rpy(x, y, z, roll, pitch, yaw):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define object position
    position = np.array([x, y, z])

    # Compute rotation matrix
    R = rpy_to_rotation_matrix(roll, pitch, yaw)

    # Create the local axes (unit vectors)
    x_axis = R[:, 0]  # Local x-axis
    y_axis = R[:, 1]  # Local y-axis
    z_axis = R[:, 2]  # Local z-axis

    # Plot the original position point
    ax.scatter([x], [y], [z], color='k', s=100, label="Position")

    # Plot the rotated axes (X, Y, Z)
    ax.quiver(x, y, z, x_axis[0], x_axis[1], x_axis[2], color='r', length=1.0, normalize=True, label="X-axis (roll)")
    ax.quiver(x, y, z, y_axis[0], y_axis[1], y_axis[2], color='g', length=1.0, normalize=True, label="Y-axis (pitch)")
    ax.quiver(x, y, z, z_axis[0], z_axis[1], z_axis[2], color='b', length=1.0, normalize=True, label="Z-axis (yaw)")

    # Set plot limits for better visualization
    ax.set_xlim([x-2, x+2])
    ax.set_ylim([y-2, y+2])
    ax.set_zlim([z-2, z+2])

    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Position: ({x}, {y}, {z}), RPY: ({np.degrees(roll):.1f}, {np.degrees(pitch):.1f}, {np.degrees(yaw):.1f})')

    # Show legend
    ax.legend()

    # Show the plot
    plt.show()

# Call the visualization function
# visualize_xyz_rpy(x, y, z, roll, pitch, yaw)


        
