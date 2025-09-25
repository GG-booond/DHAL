# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation
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
import os
from omegaconf import OmegaConf

class T1Cfg(LeggedRobotCfg):
    def __init__(self):
        super().__init__()
        # Load configuration from YAML using Hydra/OmegaConf
        config_path = os.path.join(os.path.dirname(__file__), 't1_config.yaml')
        cfg = OmegaConf.load(config_path)
        
        # Environment
        self.env.num_envs = cfg.env.num_envs
        self.env.n_scan = cfg.env.n_scan
        self.env.n_priv = cfg.env.n_priv
        self.env.n_priv_latent = cfg.env.n_priv_latent
        self.env.n_proprio = cfg.env.n_proprio
        self.env.n_recon_num = cfg.env.n_recon_num
        self.env.history_len = cfg.env.history_len
        self.env.num_observations = cfg.env.num_observations
        self.env.num_privileged_obs = cfg.env.num_privileged_obs
        self.env.num_actions = cfg.env.num_actions
        self.env.env_spacing = cfg.env.env_spacing
        self.env.send_timeouts = cfg.env.send_timeouts
        self.env.episode_length_s = cfg.env.episode_length_s
        self.env.obs_type = cfg.env.obs_type
        self.env.randomize_start_pos = cfg.env.randomize_start_pos
        self.env.randomize_start_vel = cfg.env.randomize_start_vel
        self.env.randomize_start_yaw = cfg.env.randomize_start_yaw
        self.env.rand_yaw_range = cfg.env.rand_yaw_range
        self.env.randomize_start_y = cfg.env.randomize_start_y
        self.env.rand_y_range = cfg.env.rand_y_range
        self.env.randomize_start_pitch = cfg.env.randomize_start_pitch
        self.env.rand_pitch_range = cfg.env.rand_pitch_range
        self.env.contact_buf_len = cfg.env.contact_buf_len
        self.env.next_goal_threshold = cfg.env.next_goal_threshold
        self.env.reach_goal_delay = cfg.env.reach_goal_delay
        self.env.num_future_goal_obs = cfg.env.num_future_goal_obs
        self.env.num_contact = cfg.env.num_contact

        # Normalization
        self.normalization.obs_scales.lin_vel = cfg.normalization.obs_scales.lin_vel
        self.normalization.obs_scales.ang_vel = cfg.normalization.obs_scales.ang_vel
        self.normalization.obs_scales.dof_pos = cfg.normalization.obs_scales.dof_pos
        self.normalization.obs_scales.dof_vel = cfg.normalization.obs_scales.dof_vel
        self.normalization.obs_scales.height_measurements = cfg.normalization.obs_scales.height_measurements
        self.normalization.clip_observations = cfg.normalization.clip_observations
        self.normalization.clip_actions = cfg.normalization.clip_actions

        # Noise
        self.noise.add_noise = cfg.noise.add_noise
        self.noise.noise_level = cfg.noise.noise_level
        self.noise.noise_scales.imu = cfg.noise.noise_scales.imu
        self.noise.noise_scales.base_ang_vel = cfg.noise.noise_scales.base_ang_vel
        self.noise.noise_scales.gravity = cfg.noise.noise_scales.gravity
        self.noise.noise_scales.dof_pos = cfg.noise.noise_scales.dof_pos
        self.noise.noise_scales.dof_vel = cfg.noise.noise_scales.dof_vel

        # Terrain (simplified)
        self.terrain.mesh_type = cfg.terrain.mesh_type
        self.terrain.curriculum = cfg.terrain.curriculum
        self.terrain.num_goals = cfg.terrain.num_goals

        # Commands
        self.commands.curriculum = cfg.commands.curriculum
        self.commands.resampling_time = cfg.commands.resampling_time
        self.commands.lin_vel_clip = cfg.commands.lin_vel_clip
        self.commands.ang_vel_clip = cfg.commands.ang_vel_clip
        self.commands.lin_vel_x = cfg.commands.lin_vel_x
        self.commands.lin_vel_y = cfg.commands.lin_vel_y
        self.commands.ang_vel_yaw = cfg.commands.ang_vel_yaw
        self.commands.body_height_cmd = cfg.commands.body_height_cmd

        # Init state
        self.init_state.pos = cfg.init_state.pos
        self.init_state.rot = cfg.init_state.rot
        self.init_state.lin_vel = cfg.init_state.lin_vel
        self.init_state.ang_vel = cfg.init_state.ang_vel
        self.init_state.default_joint_angles = cfg.init_state.default_joint_angles

        # Control
        self.control.control_type = cfg.control.control_type
        self.control.stiffness = cfg.control.stiffness
        self.control.damping = cfg.control.damping
        self.control.action_scale = cfg.control.action_scale
        self.control.decimation = cfg.control.decimation

        # Asset
        self.asset.file = cfg.asset.file
        self.asset.foot_name = cfg.asset.foot_name
        self.asset.penalize_contacts_on = cfg.asset.penalize_contacts_on
        self.asset.actuated_dof_names = cfg.asset.actuated_dof_names
        self.asset.terminate_after_contacts_on = cfg.asset.terminate_after_contacts_on

        # Rewards
        self.rewards.scales = cfg.rewards.scales
        self.rewards.cycle_time = cfg.rewards.cycle_time
        self.rewards.only_positive_rewards = cfg.rewards.only_positive_rewards
        self.rewards.base_height_target = cfg.rewards.base_height_target

        # Sim
        self.sim.dt = cfg.sim.dt
        self.sim.substeps = cfg.sim.substeps
        self.sim.gravity = cfg.sim.gravity
        self.sim.up_axis = cfg.sim.up_axis

class T1CfgPPO(LeggedRobotCfgPPO):
    def __init__(self):
        super().__init__()
        # Load PPO configuration from YAML
        config_path = os.path.join(os.path.dirname(__file__), 't1_config.yaml')
        cfg = OmegaConf.load(config_path)
        
        self.seed = cfg.ppo.seed
        self.runner.policy_class_name = cfg.ppo.runner.policy_class_name
        self.runner.algorithm_class_name = cfg.ppo.runner.algorithm_class_name
        self.runner.num_steps_per_env = cfg.ppo.runner.num_steps_per_env
        self.runner.max_iterations = cfg.ppo.runner.max_iterations
        self.runner.save_interval = cfg.ppo.runner.save_interval
        self.runner.experiment_name = cfg.ppo.runner.experiment_name
        self.runner.run_name = cfg.ppo.runner.run_name
        self.runner.resume = cfg.ppo.runner.resume
        self.runner.load_run = cfg.ppo.runner.load_run
        self.runner.checkpoint = cfg.ppo.runner.checkpoint
        
        self.algorithm.value_loss_coef = cfg.ppo.algorithm.value_loss_coef
        self.algorithm.use_clipped_value_loss = cfg.ppo.algorithm.use_clipped_value_loss
        self.algorithm.clip_param = cfg.ppo.algorithm.clip_param
        self.algorithm.entropy_coef = cfg.ppo.algorithm.entropy_coef
        self.algorithm.num_learning_epochs = cfg.ppo.algorithm.num_learning_epochs
        self.algorithm.num_mini_batches = cfg.ppo.algorithm.num_mini_batches
        self.algorithm.learning_rate = cfg.ppo.algorithm.learning_rate
        self.algorithm.schedule = cfg.ppo.algorithm.schedule
        self.algorithm.gamma = cfg.ppo.algorithm.gamma
        self.algorithm.lam = cfg.ppo.algorithm.lam
        self.algorithm.desired_kl = cfg.ppo.algorithm.desired_kl
        self.algorithm.max_grad_norm = cfg.ppo.algorithm.max_grad_norm
        self.algorithm.glide_advantage_w = cfg.ppo.algorithm.glide_advantage_w
        self.algorithm.push_advantage_w = cfg.ppo.algorithm.push_advantage_w
        self.algorithm.sim2real_advantage_w = cfg.ppo.algorithm.sim2real_advantage_w
        
        self.policy.continue_from_last_std = cfg.ppo.policy.continue_from_last_std
        self.policy.actor_hidden_dims = cfg.ppo.policy.actor_hidden_dims
        self.policy.critic_hidden_dims = cfg.ppo.policy.critic_hidden_dims
        self.policy.dha_hidden_dims = cfg.ppo.policy.dha_hidden_dims
        self.policy.num_modes = cfg.ppo.policy.num_modes
        self.policy.tsdyn_hidden_dims = cfg.ppo.policy.tsdyn_hidden_dims
        self.policy.tsdyn_latent_dims = cfg.ppo.policy.tsdyn_latent_dims
        self.policy.rnn_hidden_size = cfg.ppo.policy.rnn_hidden_size
        self.policy.rnn_num_layers = cfg.ppo.policy.rnn_num_layers
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 3

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            imu = 0.08
            base_ang_vel = 0.4
            gravity = 0.05
            dof_pos = 0.05
            dof_vel = 0.1

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        hf2mesh_method = "grid"
        max_error = 0.1
        max_error_camera = 2

        y_range = [-0.4, 0.4]
        
        edge_width_thresh = 0.05 

        horizontal_scale = 0.05
        horizontal_scale_camera = 0.1

        vertical_scale = 0.005 
        border_size = 5 
        height = [0.02, 0.06]
        simplify_grid = False
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        max_stair_height = 0.15
        curriculum = True

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = True
        measure_horizontal_noise = 0.0

        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 8.
        terrain_width = 8
        num_rows= 6
        num_cols = 6
        
        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 1.5,
                        "rough slope down":1.5,
                        "stairs up": 3., 
                        "stairs down": 3., 
                        "discrete": 1.5, 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "smooth flat": 0.,
                        "pit": 0.0,
                        "wall": 0.0,
                        "platform": 0,
                        "large stairs up": 0.,
                        "large stairs down": 0.,
                        "parkour": 0.,
                        "parkour_hurdle": 0.,
                        "parkour_flat": 0.,
                        "parkour_step": 0.,
                        "parkour_gap": 0,
                        "plane": 0,
                        "demo": 0.0,}
        terrain_proportions = list(terrain_dict.values())
        
        slope_treshold = 1.5
        origin_zero_z = False

        num_goals = 8

    class terrain_parkour(terrain):
        mesh_type = "trimesh"
        num_rows = 8
        num_cols = 6
        num_goals = 8
        selected = "BarrierTrack"
        max_init_terrain_level = 2
        border_size = 5
        slope_treshold = 20.

        curriculum = True
        horizontal_scale = 0.025
        pad_unavailable_info = True

        BarrierTrack_kwargs = dict(
            options= [
                # "jump",
                "crawl",
                "tilt",
                "leap",
            ],
            track_width= 1.6,
            track_block_length= 1.6,
            wall_thickness= (0.04, 0.2),
            wall_height= -0.05,
            jump= dict(
                height= (0.1, 0.4),
                depth= (0.1, 0.8),
                fake_offset= 0.0,
                jump_down_prob= 0.,
            ),
            crawl= dict(
                height= (0.22, 0.5),
                depth= (0.1, 0.6),
                wall_height= 0.6,
                no_perlin_at_obstacle= False,
            ),
            tilt= dict(
                width= (0.27, 0.38),
                depth= (0.4, 1.),
                opening_angle= 0.0,
                wall_height= 0.5,
            ),
            leap= dict(
                length= (0.8, 1.2),
                depth= (-0.05, -0.4),
                height= 0.5,
            ),
            add_perlin_noise= True,
            border_perlin_noise= True,
            border_height= 0.,
            virtual_terrain= False,
            draw_virtual_terrain= True,
            engaging_next_threshold= 1.2,
            engaging_finish_threshold= 0.,
            curriculum_perlin= False,
            no_perlin_threshold= 0.1,
        )

        TerrainPerlin_kwargs = dict(
            zScale= 0.025,
            frequency= 10,
        )
    class contact_phase():
        num_contact_phase = 2  # Bipedal


    class commands(LeggedRobotCfg.commands): 
        curriculum = False
        max_curriculum = 1.
        max_reverse_curriculum = 1.
        max_forward_curriculum = 1.
        forward_curriculum_threshold = 0.8
        yaw_command_curriculum = False
        max_yaw_curriculum = 1.
        yaw_curriculum_threshold = 0.5
        num_commands = 4
        resampling_time = 10.
        heading_command = False
        global_reference = False

        num_lin_vel_bins = 20
        lin_vel_step = 0.3
        num_ang_vel_bins = 20
        ang_vel_step = 0.3
        distribution_update_extension_distance = 1
        curriculum_seed = 100
        
        lin_vel_clip = 0.2
        ang_vel_clip = 0.2

        lin_vel_x = [-1.0, 1.0]
        lin_vel_y = [-1.0, 1.0]
        ang_vel_yaw = [-1, 1]
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

        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.68]  # Adjusted for T1 height
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'AAHead_yaw': 0.0,
            'Head_pitch': 0.0,
            'Left_Shoulder_Pitch': 0.2,
            'Left_Shoulder_Roll': -1.35,
            'Left_Elbow_Pitch': 0.0,
            'Left_Elbow_Yaw': -0.5,
            'Right_Shoulder_Pitch': 0.2,
            'Right_Shoulder_Roll': 1.35,
            'Right_Elbow_Pitch': 0.0,
            'Right_Elbow_Yaw': 0.5,
            'Waist': 0.0,
            'Left_Hip_Pitch': -0.2,
            'Left_Hip_Roll': 0.0,
            'Left_Hip_Yaw': 0.0,
            'Left_Knee_Pitch': 0.4,
            'Left_Ankle_Pitch': -0.25,
            'Left_Ankle_Roll': 0.0,
            'Right_Hip_Pitch': -0.2,
            'Right_Hip_Roll': 0.0,
            'Right_Hip_Yaw': 0.0,
            'Right_Knee_Pitch': 0.4,
            'Right_Ankle_Pitch': -0.25,
            'Right_Ankle_Roll': 0.0
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P' # P: position, V: velocity, T: torques
        stiffness = {'joint': 40., 'skateboard':0, 'truck':100, 'wheel': 0}  # Adjusted for T1
        damping = {'joint': 1, 'skateboard':0, 'truck':10, 'wheel': 0 }
        action_scale = 0.25
        decimation = 4


    class asset(LeggedRobotCfg.asset):
        
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/t1/urdf/t1.urdf'  # Assuming URDF is created
        foot_name = "foot_link"
        penalize_contacts_on = ["Trunk", "Shank", "Ankle"]
        hip_names = ["Hip_Pitch_Left", "Hip_Pitch_Right"]
        thigh_names = ["Shank_Left", "Shank_Right"]
        calf_names = ["Ankle_Cross_Left", "Ankle_Cross_Right"]

        actuated_dof_names = ['AAHead_yaw', 'Head_pitch',
                              'Left_Shoulder_Pitch', 'Left_Shoulder_Roll', 'Left_Elbow_Pitch', 'Left_Elbow_Yaw', 
                              'Right_Shoulder_Pitch', 'Right_Shoulder_Roll', 'Right_Elbow_Pitch', 'Right_Elbow_Yaw', 
                              'Waist',
                              'Left_Hip_Pitch', 'Left_Hip_Roll', 'Left_Hip_Yaw', 'Left_Knee_Pitch', 'Left_Ankle_Pitch', 'Left_Ankle_Roll',
                              'Right_Hip_Pitch', 'Right_Hip_Roll', 'Right_Hip_Yaw', 'Right_Knee_Pitch', 'Right_Ankle_Pitch', 'Right_Ankle_Roll']

        terminate_after_contacts_on = ["Trunk", "Shank", "Ankle"]

        disable_gravity = False
        collapse_fixed_joints = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 2.]
        randomize_base_mass = True
        added_mass_range = [0., 3.]
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        action_buf_len = 8
        randomize_delay = True
        
    class rewards(LeggedRobotCfg.rewards):
        class scales:
            # Adapted for bipedal robot
            tracking_lin_vel = 1.6
            tracking_ang_vel = 0.8
            hip_pos = 0.6
            orientation = -2
            dof_acc = -2.5e-7
            collision = -1.
            action_rate = -0.22
            delta_torques = -1.0e-7
            torques = -0.00001
            lin_vel_z = -0.1
            ang_vel_xy = -0.01
            reg_orientation = -25

        cycle_time = 4
        only_positive_rewards = True
        tracking_sigma = 0.5
        tracking_sigma_yaw = 0.2
        soft_dof_vel_limit = 1
        soft_torque_limit = 0.9
        max_contact_force = 70.
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5  # Adjusted for T1

    class viewer(LeggedRobotCfg.viewer):
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim(LeggedRobotCfg.sim):
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2

class T1CfgPPO( LeggedRobotCfgPPO ):
    seed = -1
    runner_class_name = 'OnPolicyRunner'
 
    class policy( LeggedRobotCfgPPO.policy ):
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [256, 128]
        dha_hidden_dims = [256, 64, 32]
        num_modes = 3
        tsdyn_hidden_dims = [256, 128, 64]
        tsdyn_latent_dims = 20
        rnn_hidden_size = 512
        rnn_num_layers = 1
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 2.e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.9
        desired_kl = 0.01
        max_grad_norm = 1.
        glide_advantage_w = 0.35
        push_advantage_w = 0.4
        sim2real_advantage_w = 0.25
    
    class depth_encoder( LeggedRobotCfgPPO.depth_encoder ):
        if_depth = False    
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * 24

    class estimator( LeggedRobotCfgPPO.estimator ):
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = T1Cfg.env.n_priv
        num_prop = T1Cfg.env.n_proprio
        num_scan = T1Cfg.env.n_scan

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticMLP'
        algorithm_class_name = 'PPO_HDS'
        num_steps_per_env = 24
        max_iterations = 100000

        save_interval = 300
        experiment_name = 't1'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None