#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty

msg = """
Control Your Robomaster EP (TurtleBot3 Style)!
---------------------------
Moving around:
   w    x
   a    d
   q    e

w/x : increase/decrease linear velocity (forward/backward)
a/d : increase/decrease angular velocity (turn left/right)
q/e : increase/decrease angular velocity (strafe left/right, if your robot supports it)

s : force stop (reset speed to 0)

CTRL-C to quit
"""

# TurtleBot3 的精髓：按键不再是赋值，而是“步进增减”
moveBindings = {
    'w': (0.1, 0.0, 0.0),   # 按 w：线速度 +0.1
    'x': (-0.1, 0.0, 0.0),  # 按 x：线速度 -0.1
    'a': (0.0, 0.0, 0.1),   # 按 a：角速度 +0.1 (左转)
    'd': (0.0, 0.0, -0.1),  # 按 d：角速度 -0.1 (右转)
    'q': (0.0, 0.1, 0.0),   # 按 q：横向速度 +0.1 (左平移，麦克纳姆轮适用)
    'e': (0.0, -0.1, 0.0),  # 按 e：横向速度 -0.1 (右平移)
}

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class TeleopKeyboardNode(Node):
    def __init__(self):
        super().__init__('teleop_keyboard_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.settings = termios.tcgetattr(sys.stdin)
        
        # 初始状态：速度全部为 0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_th = 0.0
        
        # 速度限制（防止手抖按太多把电机烧了）
        self.max_speed = 1.0
        self.max_turn = 2.0

        # 0.1秒的心跳，保证持续发送指令，不让底盘超时停机
        self.timer = self.create_timer(0.1, self.run_loop)
        self.get_logger().info(msg)

    def run_loop(self):
        key = getKey(self.settings)
        
        if key in moveBindings.keys():
            # 核心：按键是增减，而不是覆盖
            self.target_x += moveBindings[key][0]
            self.target_y += moveBindings[key][1]
            self.target_th += moveBindings[key][2]
            
            # 限幅处理 (clamp)
            self.target_x = max(-self.max_speed, min(self.max_speed, self.target_x))
            self.target_y = max(-self.max_speed, min(self.max_speed, self.target_y))
            self.target_th = max(-self.max_turn, min(self.max_turn, self.target_th))
            
            # 实时打印当前目标速度
            self.get_logger().info(f'Current Speed -> X:{self.target_x:.1f}, Y:{self.target_y:.1f}, Th:{self.target_th:.1f}')
            
        elif key == 's':
            # 急停：一键归零
            self.target_x = 0.0
            self.target_y = 0.0
            self.target_th = 0.0
            self.get_logger().info('Robot forced stop.')
            
        elif key == '\x03':  # CTRL-C
            self.destroy_node()
            rclpy.shutdown()
            sys.exit()
        
        # 注意：这里没有 else 分支了！如果没按键，什么都不做，保持当前速度继续跑！

        twist = Twist()
        twist.linear.x = self.target_x
        twist.linear.y = self.target_y
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = self.target_th
        
        # 【最关键】：发布必须在这里，和上面的逻辑平齐！
        # 无论有没有按键，每 0.1 秒都必须把当前速度发出去
        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopKeyboardNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except Exception as e:
        print(e)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()