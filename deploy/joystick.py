import os
import threading
import time

import numpy as np
from sensor_msgs.msg import JointState


class Joystick:
    def __init__(self, controller_node, debounce_time=1.0):
        self.controller_node = controller_node
        self.debounce_time = debounce_time
        self.last_command_time = {}

        # Previous joystick state, used to trigger commands only on rising edges.
        self.old_buttons = np.zeros(11)
        self.old_axes = np.zeros(8)
        self.gripper_open_position = 0.1
        self.gripper_closed_position = 0.0
        self.gripper_is_open = False

    def _publish_gripper_command(self, position):
        msg = JointState()
        msg.name = ["gripper"]
        msg.position = [position]
        self.controller_node.publisher_gripper_command.publish(msg)

    def _is_command_ready(self, command_name):
        now = time.time()
        last_command_time = self.last_command_time.get(command_name, 0.0)
        if now - last_command_time < self.debounce_time:
            return False

        self.last_command_time[command_name] = now
        return True

    def _button_pressed(self, msg, button_index, command_name):
        return (
            button_index < len(msg.buttons)
            and self.old_buttons[button_index] == 0
            and msg.buttons[button_index] == 1
            and self._is_command_ready(command_name)
        )

    def _axis_pressed(self, msg, axis_index, axis_value, command_name):
        return (
            axis_index < len(msg.axes)
            and not np.isclose(self.old_axes[axis_index], axis_value)
            and np.isclose(msg.axes[axis_index], axis_value)
            and self._is_command_ready(command_name)
        )

    def get_joy_callback(self, msg):
        """
        Callback function to handle joystick input. Joystick used is a
        8Bitdo Ultimate 2C Wireless Controller.
        """
        node = self.controller_node

        if len(self.old_buttons) != len(msg.buttons):
            self.old_buttons = np.zeros(len(msg.buttons))
        if len(self.old_axes) != len(msg.axes):
            self.old_axes = np.zeros(len(msg.axes))

        # Axes 0, 1, 3, 4 are read below without per-access bounds checks;
        # a short/malformed Joy message (e.g. during joystick reconnect) would crash the node.
        if len(msg.axes) < 5:
            return

        if node.console.isArmJoystickActivated:
            now = time.time()
            if node.last_arm_joy_time is None:
                dt = 0.0
            else:
                dt = np.clip(now - node.last_arm_joy_time, 0.0, 0.05)
            node.last_arm_joy_time = now

            filter_joystick = 0.7
            raw_arm_axes = np.array([msg.axes[4], msg.axes[0], msg.axes[1]])  # Forward/Left/Up
            raw_arm_axes[np.abs(raw_arm_axes) < node.arm_joy_deadband] = 0.0

            target_ee_lin_vel = raw_arm_axes * node.arm_joy_max_lin_speed
            if np.allclose(target_ee_lin_vel, 0.0):
                node.ref_ee_lin_vel[:] = 0.0
            else:
                node.ref_ee_lin_vel = (
                    node.ref_ee_lin_vel * filter_joystick
                    + target_ee_lin_vel * (1 - filter_joystick)
                )

            if node.ref_ee_lin_pos is not None:
                node.ref_ee_lin_pos = node.ref_ee_lin_pos + node.ref_ee_lin_vel * dt
        else:
            node.last_arm_joy_time = None
            node.ref_ee_lin_vel[:] = 0.0
            filter_joystick = 0.7
            node.ref_base_lin_vel_H[0] = (
                node.ref_base_lin_vel_H[0] * filter_joystick
                + (msg.axes[1] / 3.5) * (1 - filter_joystick)
            )
            node.ref_base_lin_vel_H[1] = (
                node.ref_base_lin_vel_H[1] * filter_joystick
                + (msg.axes[0] / 3.5) * (1 - filter_joystick)
            )
            node.ref_base_ang_yaw_dot = (
                node.ref_base_ang_yaw_dot * filter_joystick
                + (msg.axes[3] / 2.0) * (1 - filter_joystick)
            )

        node.last_joy_time = time.time()

        # Kill the node if the button is pressed.
        if self._button_pressed(msg, 8, "shutdown"):
            node.get_logger().info("Joystick button pressed, shutting down the node.")
            os.system("kill -9 $(ps -u | grep -m 1 hal | grep -o \"^[^ ]* *[0-9]*\" | grep -o \"[0-9]*\")")
            os.system("pkill -f run_controller_ros2.py")
            exit(0)

        elif self._button_pressed(msg, 7, "activate_rl"):
            # + button
            print("Locomotion activation")
            node.console.isRLActivated = not node.console.isRLActivated

        elif self._axis_pressed(msg, 7, 1.0, "go_up"):
            # up button
            threading.Thread(target=node.console.goUp, daemon=True).start()

        elif self._axis_pressed(msg, 7, -1.0, "go_down"):
            # down button
            threading.Thread(target=node.console.goDown, daemon=True).start()

        elif self._button_pressed(msg, 6, "activate_arm"):
            # - button
            print("Arm activation")
            node.console.isArmActivated = not node.console.isArmActivated

        elif self._button_pressed(msg, 4, "toggle_gripper"):
            # LB button toggles gripper open/closed
            self.gripper_is_open = not self.gripper_is_open
            if self.gripper_is_open:
                self._publish_gripper_command(self.gripper_open_position)
                print("Opening gripper")
            else:
                self._publish_gripper_command(self.gripper_closed_position)
                print("Closing gripper")

        elif self._button_pressed(msg, 0, "activate_arm_joystick"):
            # A button
            print("Arm only Joystick")
            node.console.isArmJoystickActivated = not node.console.isArmJoystickActivated
            if node.ref_ee_lin_pos is None:
                node.ref_ee_lin_pos = np.array([0.2, 0.0, 0.3])

        self.old_buttons = np.array(msg.buttons)
        self.old_axes = np.array(msg.axes)
