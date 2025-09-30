# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class T1Cfg(LeggedRobotCfg):
    """
    T1人形机器人配置类 - 带滑板功能
    T1是23自由度人形机器人，可以在滑板上进行各种运动
    """
    
    class env(LeggedRobotCfg.env):
        # =========================== 环境基础设置 ===========================
        num_envs = 1                    # 并行仿真环境数量
        
        # =========================== 观测维度设置 ===========================  
        n_scan = 17 * 11                   # 激光雷达扫描点数量 (17x11网格=187维)
        n_priv = 3 + 3 + 3                 # 私有观测维度 (9维):
                                           # - 3维: 质量参数 (base_mass, com_x, com_y, com_z)  
                                           # - 3维: 摩擦系数 (static, dynamic, rolling)
                                           # - 3维: 重力方向 (gravity_x, gravity_y, gravity_z)
        
        n_priv_latent = 4 + 1 + 12 + 12    # 私有潜在观测维度 (29维):
                                           # - 4维: 环境参数 (地形高度等)
                                           # - 1维: 相位信息
                                           # - 12维: 历史动作缓存
                                           # - 12维: 历史状态缓存
        
        n_proprio = 2 + 3 + 3 + 3 + 69 + 1 # 本体感受观测维度 (58维):
                                           # - 2维: IMU姿态 (roll, pitch) 
                                           # - 3维: 角速度 (wx, wy, wz)
                                           # - 3维: 重力投影 (projected_gravity)
                                           # - 3维: 运动命令 (vx_cmd, vy_cmd, vyaw_cmd)
                                           # - 69维: 关节状态 (23*3个关节位置 )
                                           # - 1维: 步态相位 (phase)
        
        n_recon_num = 2 + 3 + 3 + 23 + 23 + 1  # 重构观测维度 (55维)
                                            # 包含: IMU(2) + 角速度(3) + 重力(3) + 关节位置(23) + 关节速度(23) + 相位(1)
        history_len = 20                   # 历史观测长度 (20个时间步)
        
        num_observations = history_len * n_proprio  # 总观测维度 (实际测量值)
                                               # 包含历史观测: 20 × 81 = 1620维
                                               # 这是策略网络的输入维度
        
        num_privileged_obs = 1701          # 特权观测维度 (仅训练时可用) - 已移除contact_buf(200维) + heights(187维):
                                           # 优化后的特权观测构成:
                                           # - 基础观测 (obs_buf): 1620维 (历史观测1539 + 当前观测81)
                                           # - 基础线性速度: 3维  
                                           # - 质量参数: 4维
                                           # - 摩擦系数: 1维
                                           # - 电机强度: 46维 (23*2)
                                           # - 各种距离测量: 24维 (6*4)
                                           # - 滑板姿态: 3维
                                           # 注: 已移除contact_buf(200维) + heights(187维)以节省计算资源和适配真实机器人
        
        num_actions = 23                   # 动作维度 (T1的23个可控关节):
                                           # 头部(2) + 左臂(4) + 右臂(4) + 腰部(1) + 左腿(6) + 右腿(6)
                                           # 不包括滑板关节(滑板通过物理约束控制)
        
        # =========================== 环境物理设置 ===========================
        env_spacing = 3.                   # 环境间距离 (米)
        send_timeouts = True               # 是否发送超时信息给算法
        episode_length_s = 20              # 单个episode长度 (秒)
        obs_type = "og"                    # 观测类型
        
        # =========================== 初始状态随机化 ===========================
        randomize_start_pos = False        # 是否随机化初始位置
        randomize_start_vel = False        # 是否随机化初始速度
        randomize_start_yaw = False        # 是否随机化初始偏航角
        rand_yaw_range = 1.2               # 偏航角随机范围 (弧度)
        randomize_start_y = False          # 是否随机化Y轴初始位置
        rand_y_range = 0.5                 # Y轴随机范围 (米)
        randomize_start_pitch = False      # 是否随机化初始俯仰角
        rand_pitch_range = 1.6             # 俯仰角随机范围 (弧度)

        # =========================== 接触和相位设置 ===========================
        contact_buf_len = 100              # 接触缓冲区长度
        next_goal_threshold = 0.2          # 下一个目标的阈值
        reach_goal_delay = 0.1             # 到达目标的延迟
        num_future_goal_obs = 2            # 未来目标观测数量
        num_contact = 2                    # 接触相位数量 (T1机器人有2只脚)

    class normalization(LeggedRobotCfg.normalization):
        """
        观测归一化配置
        目的: 将不同量纲和数值范围的观测值归一化到相似范围，提高训练稳定性
        """
        class obs_scales:
            # =========================== 观测缩放因子 ===========================
            lin_vel = 2.0                  # 线速度缩放因子
                                           # 将m/s单位的线速度除以0.5 (1/2.0)
                                           # 例: 1m/s的速度 → 1*2.0 = 2.0 (无量纲)
            
            ang_vel = 0.25                 # 角速度缩放因子  
                                           # 将rad/s单位的角速度乘以0.25
                                           # 例: 4rad/s的角速度 → 4*0.25 = 1.0 (无量纲)
            
            dof_pos = 1.0                  # 关节位置缩放因子
                                           # 关节角度(弧度)保持原值，因为通常在[-π, π]范围内
                                           
            dof_vel = 0.05                 # 关节速度缩放因子
                                           # 将rad/s单位的关节速度乘以0.05
                                           # 例: 20rad/s的关节速度 → 20*0.05 = 1.0 (无量纲)
                                           
            height_measurements = 5.0      # 地形高度缩放因子
                                           # 将米单位的高度差乘以5.0
                                           # 例: 0.2m的高度差 → 0.2*5.0 = 1.0 (无量纲)
        
        # =========================== 数值裁剪范围 ===========================        
        clip_observations = 100.           # 观测值裁剪范围 [-100, 100]
                                           # 防止观测值过大导致训练不稳定
        clip_actions = 3                   # 动作值裁剪范围 [-3, 3]  
                                           # 限制策略输出的动作幅度

    class noise(LeggedRobotCfg.noise):
        """传感器噪声配置 - 模拟真实传感器的噪声"""
        add_noise = True                   # 是否添加噪声
        noise_level = 1.0                  # 噪声强度乘数
        class noise_scales:
            # 各传感器噪声标准差
            imu = 0.08                     # IMU姿态角噪声 (弧度)
            base_ang_vel = 0.4             # 基座角速度噪声 (rad/s)
            gravity = 0.05                 # 重力向量噪声
            dof_pos = 0.05                 # 关节位置噪声 (弧度)
            dof_vel = 0.1                  # 关节速度噪声 (rad/s)

    class terrain(LeggedRobotCfg.terrain):
        """地形配置 - 定义训练环境的地形特征"""
        # =========================== 地形基础设置 ===========================
        mesh_type = 'plane'                # 地形类型: 'plane'(平面), 'heightfield'(高度场), 'trimesh'(三角网格)
        hf2mesh_method = "grid"            # 高度场到网格转换方法
        max_error = 0.1                    # 最大误差
        max_error_camera = 2               # 相机最大误差

        # =========================== 地形尺寸设置 ===========================
        y_range = [-0.4, 0.4]              # Y轴范围 (米)
        edge_width_thresh = 0.05           # 边缘宽度阈值
        
        horizontal_scale = 0.05            # 水平缩放因子 (米/像素)
        horizontal_scale_camera = 0.1      # 相机水平缩放因子
        vertical_scale = 0.005             # 垂直缩放因子 (米)
        
        border_size = 5                    # 边界大小 (米)
        terrain_length = 8.                # 地形长度 (米)
        terrain_width = 8                  # 地形宽度 (米)
        num_rows = 6                       # 地形行数
        num_cols = 6                       # 地形列数

        # =========================== 地形特征设置 ===========================
        height = [0.02, 0.06]              # 高度范围 (米)
        simplify_grid = False              # 是否简化网格
        gap_size = [0.02, 0.1]             # 间隙大小范围 (米)
        stepping_stone_distance = [0.02, 0.08]  # 踏脚石距离范围 (米)
        downsampled_scale = 0.075          # 下采样缩放
        max_stair_height = 0.15            # 最大楼梯高度 (米)
        slope_treshold = 1.5               # 坡度阈值
        
        # =========================== 地形类型和比例 ===========================
        terrain_dict = {
            "smooth slope": 0.,            # 平滑坡面
            "rough slope up": 1.5,         # 粗糙上坡  
            "rough slope down": 1.5,       # 粗糙下坡
            "stairs up": 3.,               # 上楼梯
            "stairs down": 3.,             # 下楼梯
            "discrete": 1.5,               # 离散地形
            "stepping stones": 0.,         # 踏脚石
            "gaps": 0.,                    # 间隙
            "smooth flat": 0.,             # 平滑平面
            "pit": 0.0,                    # 坑洞
            "wall": 0.0,                   # 墙壁
            "platform": 0,                 # 平台
            "large stairs up": 0.,         # 大楼梯上
            "large stairs down": 0.,       # 大楼梯下
            "parkour": 0.,                 # 跑酷地形
            "parkour_hurdle": 0.,          # 跑酷障碍
            "parkour_flat": 0.,            # 跑酷平面
            "parkour_step": 0.,            # 跑酷台阶
            "parkour_gap": 0,              # 跑酷间隙
            "plane": 0,                    # 平面
            "demo": 0.0,                   # 演示地形
        }
        terrain_proportions = list(terrain_dict.values())  # 地形比例列表

        # =========================== 物理属性 ===========================
        static_friction = 1.0              # 静摩擦系数
        dynamic_friction = 1.0             # 动摩擦系数  
        restitution = 0.                   # 恢复系数 (弹性)
        
        # =========================== 高度测量设置 ===========================
        measure_heights = False            # 禁用高度测量 - 适配无激光雷达的真实机器人
        # 高度测量点 - 机器人周围的采样点（已禁用）
        # measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
        #                     0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # X轴测量点 (米)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 
        #                     0., 0.1, 0.2, 0.3, 0.4, 0.5]                  # Y轴测量点 (米)
        measure_horizontal_noise = 0.0     # 水平测量噪声（已禁用）
        
        # =========================== 课程学习设置 ===========================
        curriculum = True                  # 是否启用课程学习
        max_init_terrain_level = 5         # 最大初始地形难度级别
        all_vertical = False               # 是否全为垂直地形
        no_flat = True                     # 是否无平地
        selected = False                   # 是否选择特定地形
        terrain_kwargs = None              # 地形关键字参数
        origin_zero_z = False              # 是否将原点Z坐标设为0

        num_goals = 8                      # 目标数量

    class commands(LeggedRobotCfg.commands):
        """运动命令配置 - 定义机器人的目标运动"""
        # =========================== 课程学习设置 ===========================
        curriculum = False                 # 是否启用命令课程学习
        max_curriculum = 1.                # 最大课程学习乘数
        max_reverse_curriculum = 1.        # 最大倒退课程学习乘数
        max_forward_curriculum = 1.        # 最大前进课程学习乘数
        forward_curriculum_threshold = 0.8 # 前进课程学习阈值
        yaw_command_curriculum = False     # 是否启用偏航命令课程学习
        max_yaw_curriculum = 1.            # 最大偏航课程学习乘数
        yaw_curriculum_threshold = 0.5     # 偏航课程学习阈值
        curriculum_seed = 100              # 课程学习随机种子
        
        # =========================== 命令基础设置 ===========================
        num_commands = 4                   # 命令维度 (lin_vel_x, lin_vel_y, ang_vel_yaw, body_height)
        resampling_time = 10.              # 命令重新采样时间 (秒)
        heading_command = False            # 是否使用朝向命令 (vs 角速度命令)
        global_reference = False           # 是否使用全局参考坐标系
        
        # =========================== 命令离散化设置 ===========================
        num_lin_vel_bins = 20              # 线速度离散化区间数
        lin_vel_step = 0.3                 # 线速度步长
        num_ang_vel_bins = 20              # 角速度离散化区间数  
        ang_vel_step = 0.3                 # 角速度步长
        distribution_update_extension_distance = 1  # 分布更新扩展距离
        
        # =========================== 命令裁剪设置 ===========================
        lin_vel_clip = 0.2                 # 线速度裁剪阈值 (小于此值设为0)
        ang_vel_clip = 0.2                 # 角速度裁剪阈值 (小于此值设为0)
        
        # =========================== 命令范围设置 ===========================
        # 训练时的命令采样范围
        lin_vel_x = [-0.1, 0.5]            # X轴线速度范围 (m/s)
        lin_vel_y = [-0.1, 0.1]            # Y轴线速度范围 (m/s) 
        ang_vel_yaw = [0,0]              # 偏航角速度范围 (rad/s)
        body_height_cmd = [-0.05, 0.05]    # 身体高度命令范围 (m)
        impulse_height_commands = False    # 是否使用冲击高度命令
        
        # 物理限制范围
        limit_vel_x = [-10.0, 10.0]        # X轴速度物理限制 (m/s)
        limit_vel_y = [-0.6, 0.6]          # Y轴速度物理限制 (m/s)
        limit_vel_yaw = [-10.0, 10.0]      # 偏航速度物理限制 (rad/s)
        heading = [-3.14, 3.14]            # 朝向角度范围 (rad)

        # =========================== 课程学习命令范围 ===========================
        class curriculum_ranges:
            """课程学习初始命令范围 (逐渐扩展到max_ranges)"""
            lin_vel_x = [0, 0.1]           # 初始X轴线速度范围
            lin_vel_y = [-0, 0]            # 初始Y轴线速度范围 (侧向运动较难)
            ang_vel_yaw = [-0.1, 0.1]      # 初始偏航角速度范围
            heading = [-0.1, 0.1]          # 初始朝向范围

        class max_ranges:
            """课程学习最终命令范围"""
            lin_vel_x = [-0.1, 0.5]        # 最终X轴线速度范围
            lin_vel_y = [0, 0]             # 最终Y轴线速度范围 (保持为0)
            ang_vel_yaw = [-0.8, 0.8]      # 最终偏航角速度范围  
            heading = [-3.14, 3.14]        # 最终朝向范围

        waypoint_delta = 0.7               # 路径点间距 (米)
        body_height_cmd = [-0.05, 0.05]
        impulse_height_commands = False

        limit_vel_x = [-10.0, 10.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-10.0, 10.0]

        heading = [-3.14, 3.14]
        class curriculum_ranges:
            lin_vel_x = [0.5, 1]
            lin_vel_y = [-0, 0]
            ang_vel_yaw = [-0.1,0.1]
            heading = [-0.1, 0.1]

        class max_ranges:
            lin_vel_x = [-1.6, 1.6]
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-3.14, 3.14]

        waypoint_delta = 0.7

    class asset(LeggedRobotCfg.asset):
        """机器人资产配置 - T1人形机器人 + 滑板系统"""
        # =========================== 机器人模型文件 ===========================
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/T1_skate/urdf/T1_skate.urdf'
        name = "T1"
        
        # =========================== 接触和碰撞设置 ===========================
        foot_name = "Ankle_Cross"          # 脚部link名称 (用于识别脚部)
        # 接触惩罚 - 这些部位接触地面会被惩罚
        penalize_contacts_on = ["H1", "H2", "AL1", "AL2", "AL3", "AR1", "AR2", "AR3"]
        # 终止条件 - 这些部位接触地面会终止episode
        terminate_after_contacts_on = ["H1", "H2"]
        
        # =========================== T1关节名称定义 ===========================
        # 髋关节名称
        hip_names = ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", 
                     "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw"]
        # 大腿关节名称 (膝关节被归类为大腿部分)
        thigh_names = ["Left_Knee_Pitch", "Right_Knee_Pitch"]
        # 小腿关节名称 (踝关节)
        calf_names = ["Left_Ankle_Pitch", "Left_Ankle_Roll", 
                      "Right_Ankle_Pitch", "Right_Ankle_Roll"]

        # T1主动控制关节 (23个自由度)
        actuated_dof_names = [
            "AAHead_yaw", "Head_pitch",                                    # 头部 (2 DOF)
            "Left_Shoulder_Pitch", "Left_Shoulder_Roll",                   # 左臂 (4 DOF)
            "Left_Elbow_Pitch", "Left_Elbow_Yaw",
            "Right_Shoulder_Pitch", "Right_Shoulder_Roll",                 # 右臂 (4 DOF)
            "Right_Elbow_Pitch", "Right_Elbow_Yaw",
            "Waist",                                                       # 腰部 (1 DOF)
            "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",            # 左腿 (6 DOF)
            "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
            "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",         # 右腿 (6 DOF)
            "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"
        ]

        # =========================== 滑板系统关节 ===========================
        # 滑板被动关节 (有阻尼控制)
        underact_dof_names = ['front_truck_roll_joint', 'rear_truck_roll_joint']
        
        # 滑板无驱动关节 (完全自由)
        undriven_dof_names = ['skateboard_joint_x', 'skateboard_joint_y', 'skateboard_joint_z',
                              'front_left_wheel_joint', 'front_right_wheel_joint', 
                              'rear_left_wheel_joint', 'rear_right_wheel_joint']
        
        # 滑板连接关节 (3个平移DOF)
        skateboard_dof_names = ['skateboard_joint_x', 'skateboard_joint_y', 'skateboard_joint_z']
        
        # 滑板轮子关节
        wheel_dof_names = ['front_left_wheel_joint', 'front_right_wheel_joint',
                           'rear_left_wheel_joint', 'rear_right_wheel_joint']

        # =========================== 刚体链接名称 ===========================
        # 滑板甲板链接
        skateboard_link_name = ['skateboard_deck']
        
        # 轮子链接
        wheel_link_names = ['front_left_wheel', 'front_right_wheel', 
                           'rear_left_wheel', 'rear_right_wheel']
        
        # 脚部标记点 (用于接触检测) - T1滑板上的标记点
        # T1是两足机器人，使用前两个标记点对应左右脚
        marker_link_names = ["LF_f_marker", "RF_f_marker"]

        # =========================== 物理参数 ===========================
        wheel_radius = 0.030               # 滑板轮子半径 (米)
        density = 0.001                    # 密度 (kg/m³)
        angular_damping = 0.               # 角阻尼
        linear_damping = 0.                # 线性阻尼
        max_angular_velocity = 1000.       # 最大角速度 (rad/s)
        max_linear_velocity = 1000.        # 最大线速度 (m/s)
        armature = 0.                      # 电枢阻抗
        thickness = 0.01                   # 厚度 (米)
        
        # =========================== 仿真设置 ===========================
        disable_gravity = False            # 是否禁用重力
        collapse_fixed_joints = False      # 是否合并固定关节
        fix_base_link = False              # 是否固定基座
        default_dof_drive_mode = 1         # 默认DOF驱动模式 (1=力控制)
        self_collisions = 0                # 自碰撞检测级别
        replace_cylinder_with_capsule = True  # 是否用胶囊体替换圆柱体
        flip_visual_attachments = True     # 是否翻转视觉附件

    class init_state( LeggedRobotCfg.init_state ):
        """初始状态配置 - 自然站立姿态"""
        # =========================== 基座初始状态 ===========================
        pos = [0.0, 0.0, 0.72]               # 初始位置 - 调整高度适配自然站立姿态
        rot = [0.0, 0.0, 0.0, 1.0]         # 初始旋转四元数
        lin_vel = [0.0, 0.0, 0.0]          # 初始线速度
        ang_vel = [0.0, 0.0, 0.0]          # 初始角速度
        
        # =========================== 自然站立关节角度 ===========================
        # 参考标准人形机器人站立姿态，手臂稍微张开，腿部微弯保持平衡
        default_joint_angles = {
            # 头部 - 直视前方
            'AAHead_yaw': 0.0,
            'Head_pitch': 0.0,
            
            # 手臂 - 自然张开姿态，提高平衡性
            'Left_Shoulder_Pitch': 0.25,      # 肩部稍微前倾
            'Left_Shoulder_Roll': -1.4,       # 左臂向外张开
            'Left_Elbow_Pitch': 0.0,          # 肘关节保持伸直
            'Left_Elbow_Yaw': -0.5,           # 前臂稍微内旋
            'Right_Shoulder_Pitch': 0.25,     # 肩部稍微前倾
            'Right_Shoulder_Roll': 1.4,       # 右臂向外张开
            'Right_Elbow_Pitch': 0.0,         # 肘关节保持伸直
            'Right_Elbow_Yaw': 0.5,           # 前臂稍微外旋
            
            # 腰部 - 直立
            'Waist': 0.0,
            
            # 腿部 - 微弯姿态，更稳定的站立
            'Left_Hip_Pitch': -1.21,           # 髋部稍微后倾
            'Left_Hip_Roll': 0.0,             # 髋部侧倾保持中性
            'Left_Hip_Yaw': 0.0,              # 髋部旋转保持中性
            'Left_Knee_Pitch': 1.21,           # 膝盖微弯，增加稳定性
            'Left_Ankle_Pitch': 0.0,         # 踝关节稍微背屈，平衡膝盖弯曲
            'Left_Ankle_Roll': 0.0,           # 踝关节侧倾保持中性
            
            # 滑板系统 - 保持在合适位置
            'skateboard_joint_x': 0,
            'skateboard_joint_y': 0.0,        # 滑板高度调整
            'skateboard_joint_z': 0,
            
            # 滑板配件 - 保持中性
            'front_truck_roll_joint': 0,
            'rear_truck_roll_joint': 0,
            'front_left_wheel_joint': 0,
            'front_right_wheel_joint': 0,
            'rear_left_wheel_joint': 0,
            'rear_right_wheel_joint': 0,


            'Right_Hip_Pitch': -0.1,          # 髋部稍微后倾
            'Right_Hip_Roll': 0.0,            # 髋部侧倾保持中性
            'Right_Hip_Yaw': 0.0,             # 髋部旋转保持中性
            'Right_Knee_Pitch': 0.0,          # 膝盖微弯，增加稳定性
            'Right_Ankle_Pitch': 0.0,        # 踝关节稍微背屈，平衡膝盖弯曲
            'Right_Ankle_Roll': 0.0          # 踝关节侧倾保持中性
        }
        
        # =========================== 滑行默认姿态 ==========================

    class control(LeggedRobotCfg.control):
        """控制系统配置 - T1机器人的关节控制参数"""
        # =========================== 控制类型 ===========================
        control_type = 'P'                 # 控制类型: 'P'(位置), 'V'(速度), 'T'(力矩)
        
        # =========================== PD控制增益 ===========================
        # 不同关节组的刚度系数 (P增益) - 降低增益提高稳定性
        stiffness = {
            'Head': 20.0,                  # 头部关节 (较弱，避免过激运动)
            'Shoulder': 20.0,              # 肩部关节  
            'Elbow': 20.0,                 # 肘部关节
            'Waist': 200.0,                # 腰部关节 
            'Hip': 200.0,                  # 髋部关节 
            'Knee': 300.0,                 # 膝部关节
            'Ankle': 50.0,                # 踝部关节
            'skateboard': 0,               # 滑板关节 (被动)
            'truck': 0,                   # 滑板转向架 (轻微阻尼)
            'wheel': 0                     # 轮子 (自由旋转)
        }
        
        # 阻尼系数 (D增益) - 降低阻尼
        damping = {
            'Head': 0.5,                   # 头部关节阻尼
            'Shoulder': 0.5,               # 肩部关节阻尼
            'Elbow': 0.5,                  # 肘部关节阻尼
            'Waist': 0.5,                  # 腰部关节阻尼
            'Hip': 5.0,                    # 髋部关节阻尼
            'Knee': 8.0,                   # 膝部关节阻尼
            'Ankle': 3.0,                  # 踝部关节阻尼
            'skateboard': 0,               # 滑板关节阻尼
            'truck': 0.5,                  # 滑板转向架阻尼
            'wheel': 0                     # 轮子阻尼
        }
        
        action_scale = 0.25                # 动作缩放因子 (保持与Go1相同)
        decimation = 4                     # 动作重复次数 (每个动作执行4个仿真步)

    class domain_rand(LeggedRobotCfg.domain_rand):
        """领域随机化配置 - 提高策略的泛化能力"""
        # =========================== 摩擦力随机化 ===========================
        randomize_friction = True          # 是否随机化摩擦系数
        friction_range = [0.6, 2.]         # 摩擦系数随机范围
        
        # =========================== 质量随机化 ===========================
        randomize_base_mass = True         # 是否随机化基座质量
        added_mass_range = [0., 3.]        # 额外质量范围 (kg)
        
        # =========================== 质心随机化 ===========================  
        randomize_base_com = True          # 是否随机化基座质心
        added_com_range = [-0.2, 0.2]      # 质心偏移范围 (米)
        
        # =========================== 外力干扰 ===========================
        push_robots = True                 # 是否施加随机推力
        push_interval_s = 8                # 推力间隔 (秒)
        max_push_vel_xy = 0.5              # 最大推力速度 (m/s)
        
        # =========================== 电机随机化 ===========================
        randomize_motor = True             # 是否随机化电机强度
        motor_strength_range = [0.8, 1.2]  # 电机强度范围 (倍数)
        
        # =========================== 延迟和缓冲 ===========================
        action_buf_len = 8                 # 动作缓冲区长度
        randomize_delay = True             # 是否随机化执行延迟

    class rewards(LeggedRobotCfg.rewards):
        """奖励函数配置 - T1滑板运动的奖励设计"""
        
        class scales:
            """各奖励项的权重系数 - T1滑板机器人专用 (仅Push模式)"""
            
            # =============== GROUP1: PUSH模式奖励 (T1主要模式) ===============
            push_tracking_lin_vel = 1.6    # 蹬地时线速度跟踪奖励
            push_tracking_ang_vel = 0.8    # 蹬地时角速度跟踪奖励
            push_joint_pos = 1.2           # 蹬地时关节姿态奖励
            push_hip_pos = 0.6             # 蹬地时髋部姿态奖励
            push_orientation = -2          # 蹬地时姿态稳定性惩罚

            # =============== GROUP2: 滑板控制奖励 ===============
            skateboard_pos = 0.5           # 滑板姿态控制
            reg_wheel_contact_number = 0.8 # 轮子接触地面奖励
            wheel_speed = 0.3              # 轮速奖励

            # =============== GROUP3: 正则化惩罚 ===============
            reg_dof_acc = -2.5e-7          # 关节加速度惩罚 (平滑运动)
            reg_collision = -1.            # 碰撞惩罚
            reg_action_rate = -0.22        # 动作变化率惩罚 (减少抖动)
            reg_delta_torques = -1.0e-7    # 力矩变化惩罚
            reg_torques = -0.00001         # 力矩大小惩罚 (能耗)
            reg_lin_vel_z = -0.1           # Z轴线速度惩罚 (避免跳跃)
            reg_ang_vel_xy = -0.01         # XY轴角速度惩罚 (姿态稳定)
            reg_orientation = -25          # 姿态角度惩罚 (避免倾倒)

        # =========================== 奖励系统参数 ===========================
        cycle_time = 4                     # 运动周期时间 (秒) - 滑行/蹬地切换周期
        only_positive_rewards = True       # 是否只使用正奖励 (负奖励会被裁剪为0)
        
        # =========================== 跟踪精度参数 ===========================
        tracking_sigma = 0.5               # 跟踪奖励的标准差 (速度跟踪宽容度)
        tracking_sigma_yaw = 0.2           # 偏航跟踪奖励标准差
        
        # =========================== 软约束参数 ===========================
        soft_dof_vel_limit = 1             # 关节速度软约束系数
        soft_torque_limit = 0.9            # 力矩软约束系数
        soft_dof_pos_limit = 0.9           # 关节位置软约束系数
        max_contact_force = 70.            # 最大接触力 (N)
        base_height_target = 1.0           # 目标身体高度 (米，T1比Go1高)

    class viewer(LeggedRobotCfg.viewer):
        """可视化观察器配置"""
        ref_env = 0                        # 参考环境ID (用于调试观察)
        pos = [10, 0, 6]                   # 相机初始位置 [x, y, z] (米)
        lookat = [11., 5, 3.]              # 相机朝向点 [x, y, z] (米)

    class sim(LeggedRobotCfg.sim):
        """物理仿真参数配置"""
        dt = 0.005                         # 仿真时间步长 (秒)
        substeps = 1                       # 子步数
        gravity = [0., 0., -9.81]          # 重力向量 [x, y, z] (m/s²)
        up_axis = 1                        # 上轴方向 (1=Z轴向上)

        class physx:
            """PhysX物理引擎参数 - 调整为更宽松的设置提高稳定性"""
            num_threads = 10               # 线程数
            solver_type = 1                # 求解器类型
            num_position_iterations = 8    # 位置迭代次数 (增加)
            num_velocity_iterations = 1    # 速度迭代次数 (增加)
            contact_offset = 0.02          # 接触偏移 (增加)
            rest_offset = 0.0              # 静止偏移 (米)
            bounce_threshold_velocity = 0.2 # 弹跳阈值速度 (降低)
            max_depenetration_velocity = 10.0 # 最大反渗透速度 (增加)
            max_gpu_contact_pairs = 2**23  # GPU最大接触对数
            default_buffer_size_multiplier = 5 # 缓冲区大小乘数
            contact_collection = 2         # 接触收集模式

    class contact_phase():
        """接触相位配置 - 滑行/蹬地相位切换"""
        num_contact_phase = 3              # 接触相位数量

    class normalization:
        """观测归一化配置 (重复定义，用于确保覆盖)"""
        class obs_scales:
            lin_vel = 2.0                  # 线速度缩放
            ang_vel = 0.25                 # 角速度缩放
            dof_pos = 1.0                  # 关节位置缩放
            dof_vel = 0.05                 # 关节速度缩放
            height_measurements = 5.0      # 高度测量缩放
        clip_observations = 100.           # 观测值裁剪范围
        clip_actions = 100.                # 动作值裁剪范围


class T1CfgPPO(LeggedRobotCfgPPO):
    """
    T1机器人PPO算法配置类
    定义强化学习算法的超参数和训练设置
    """
    seed = 42                              # 随机种子 (-1为随机)
    runner_class_name = 'OnPolicyRunner'   # 运行器类名
    
    class policy(LeggedRobotCfgPPO.policy):
        """策略网络配置"""
        continue_from_last_std = True      # 是否从上次标准差继续
        scan_encoder_dims = [128, 64, 32]  # 扫描编码器维度
        actor_hidden_dims = [512, 256, 128] # Actor网络隐藏层维度
        critic_hidden_dims = [512, 256, 128] # Critic网络隐藏层维度
        priv_encoder_dims = [256, 128]     # 私有信息编码器维度
        dha_hidden_dims = [256, 64, 32]    # DHA隐藏层维度
        num_modes = 3                      # 模式数量
        tsdyn_hidden_dims = [256, 128, 64] # 时序动力学隐藏层维度
        tsdyn_latent_dims = 20             # 时序动力学潜在维度
        rnn_hidden_size = 512              # RNN隐藏层大小
        rnn_num_layers = 1                 # RNN层数
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        """PPO算法参数"""
        value_loss_coef = 1.0              # 价值损失系数
        use_clipped_value_loss = True      # 使用裁剪价值损失
        clip_param = 0.2                   # PPO裁剪参数
        entropy_coef = 0.01                # 熵系数 (探索性)
        num_learning_epochs = 5            # 学习轮数
        num_mini_batches = 4               # 小批次数量
        learning_rate = 2.e-4              # 学习率
        schedule = 'adaptive'              # 学习率调度策略
        gamma = 0.99                       # 折扣因子
        lam = 0.9                          # GAE lambda参数
        desired_kl = 0.01                  # 期望KL散度
        max_grad_norm = 1.                 # 最大梯度范数
        glide_advantage_w = 0.35           # 滑行优势权重
        push_advantage_w = 0.4             # 推进优势权重
        sim2real_advantage_w = 0.25        # 仿真到现实优势权重
    
    class depth_encoder(LeggedRobotCfgPPO.depth_encoder):
        """深度编码器配置"""
        if_depth = False                   # 是否使用深度信息
        depth_shape = T1Cfg.depth.resized if hasattr(T1Cfg, 'depth') else [64, 64]  # 深度图形状
        buffer_len = T1Cfg.depth.buffer_len if hasattr(T1Cfg, 'depth') else 100     # 缓冲区长度
        hidden_dims = 512                  # 隐藏层维度
        learning_rate = 1.e-3              # 学习率
        num_steps_per_env = 24             # 每环境步数
    
    class estimator(LeggedRobotCfgPPO.estimator):
        """状态估计器配置"""
        train_with_estimated_states = True # 使用估计状态训练
        learning_rate = 1.e-4              # 学习率
        hidden_dims = [128, 64]            # 隐藏层维度
        priv_states_dim = T1Cfg.env.n_priv # 私有状态维度
        num_prop = T1Cfg.env.n_proprio     # 本体感知维度
        num_scan = T1Cfg.env.n_scan        # 扫描维度
        
    class runner(LeggedRobotCfgPPO.runner):
        """训练运行器参数"""  
        policy_class_name = 'ActorCriticMLP' # 策略类名
        algorithm_class_name = 'PPO_HDS'   # 算法类名
        num_steps_per_env = 24             # 每环境步数
        max_iterations = 100000            # 最大训练迭代次数
        save_interval = 300                # 模型保存间隔
        experiment_name = 't1'             # 实验名称
        run_name = ''                      # 运行名称
        resume = False                     # 是否恢复训练
        load_run = -1                      # 加载运行ID
        checkpoint = -1                    # 检查点
        resume_path = None                 # 恢复路径
