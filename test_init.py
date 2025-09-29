#!/usr/bin/env python3
"""
测试T1机器人是否能成功初始化第一帧
"""

import numpy as np
from legged_gym.envs import T1Cfg
from legged_gym import LEGGED_GYM_ROOT_DIR
import isaacgym
from isaacgym import gymapi, gymtorch
import torch  # 必须在isaacgym之后导入
import os

def test_t1_initialization():
    print("=" * 50)
    print("测试T1机器人初始化...")
    print("=" * 50)
    
    try:
        # 创建gym实例
        gym = gymapi.acquire_gym()
        
        # 仿真参数
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.005
        sim_params.substeps = 1
        sim_params.gravity = gymapi.Vec3(0., 0., -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        
        # PhysX参数 - 更宽松的设置
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 2  # 降低精度
        sim_params.physx.num_velocity_iterations = 1  # 增加速度迭代
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        
        # 创建仿真
        device_id = 0
        graphics_device_id = -1  # headless
        sim = gym.create_sim(device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        
        if sim is None:
            print("❌ 创建仿真失败")
            return False
            
        print("✅ 仿真创建成功")
        
        # 创建地面
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        gym.add_ground(sim, plane_params)
        print("✅ 地面创建成功")
        
        # 加载T1资产
        cfg = T1Cfg()
        asset_path = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        
        print(f"加载资产: {asset_path}")
        
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = cfg.asset.fix_base_link
        asset_options.density = cfg.asset.density
        asset_options.angular_damping = cfg.asset.angular_damping
        asset_options.linear_damping = cfg.asset.linear_damping
        asset_options.max_angular_velocity = cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = cfg.asset.max_linear_velocity
        asset_options.armature = cfg.asset.armature
        asset_options.thickness = cfg.asset.thickness
        asset_options.disable_gravity = cfg.asset.disable_gravity
        
        robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
        if robot_asset is None:
            print("❌ T1资产加载失败")
            return False
            
        print("✅ T1资产加载成功")
        
        # 获取资产信息
        num_dof = gym.get_asset_dof_count(robot_asset)
        num_bodies = gym.get_asset_rigid_body_count(robot_asset)
        dof_names = gym.get_asset_dof_names(robot_asset)
        body_names = gym.get_asset_rigid_body_names(robot_asset)
        
        print(f"DOF数量: {num_dof}")
        print(f"刚体数量: {num_bodies}")
        
        # 创建环境
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        env = gym.create_env(sim, env_lower, env_upper, 1)
        
        if env is None:
            print("❌ 环境创建失败")
            return False
        print("✅ 环境创建成功")
        
        # 设置初始姿态
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(cfg.init_state.pos[0], cfg.init_state.pos[1], cfg.init_state.pos[2])
        start_pose.r = gymapi.Quat(cfg.init_state.rot[0], cfg.init_state.rot[1], cfg.init_state.rot[2], cfg.init_state.rot[3])
        
        print(f"初始位置: {cfg.init_state.pos}")
        print(f"初始旋转: {cfg.init_state.rot}")
        
        # 创建角色
        actor = gym.create_actor(env, robot_asset, start_pose, "t1", 0, 0, 0)
        if actor is None:
            print("❌ 角色创建失败")
            return False
        print("✅ 角色创建成功")
        
        # 设置DOF属性
        dof_props = gym.get_asset_dof_properties(robot_asset)
        gym.set_actor_dof_properties(env, actor, dof_props)
        
        # 设置初始DOF状态
        dof_states = np.zeros(num_dof, dtype=gymapi.DofState.dtype)
        for i, name in enumerate(dof_names):
            if name in cfg.init_state.default_joint_angles:
                dof_states[i]['pos'] = cfg.init_state.default_joint_angles[name]
                print(f"设置关节 {name}: {cfg.init_state.default_joint_angles[name]}")
            else:
                dof_states[i]['pos'] = 0.0
                
        gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
        print("✅ 关节状态设置成功")
        
        # 准备仿真
        gym.prepare_sim(sim)
        print("✅ 仿真准备成功")
        
        # 获取状态张量
        print("获取状态张量...")
        rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        
        # 包装为PyTorch张量
        rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)
        print(f"刚体状态张量形状: {rigid_body_states.shape}")
        
        # 检查是否有NaN
        has_nan = torch.isnan(rigid_body_states).any()
        print(f"包含NaN: {has_nan}")
        
        if has_nan:
            print("❌ 初始状态包含NaN值")
            nan_bodies = torch.isnan(rigid_body_states).any(dim=-1).any(dim=0)
            nan_indices = torch.where(nan_bodies)[0]
            print(f"NaN刚体索引: {nan_indices.tolist()}")
            
            # 显示前几个刚体的状态
            for i in range(min(5, rigid_body_states.shape[1])):
                state = rigid_body_states[0, i, :3]  # 只显示位置
                print(f"刚体 {i} ({body_names[i] if i < len(body_names) else 'unknown'}): {state}")
            return False
        else:
            print("✅ 初始状态正常，无NaN值")
            
            # 显示一些关键刚体的状态
            print("\n关键刚体状态:")
            for i in range(min(3, rigid_body_states.shape[1])):
                pos = rigid_body_states[0, i, :3]
                vel = rigid_body_states[0, i, 7:10]
                print(f"刚体 {i} ({body_names[i] if i < len(body_names) else 'unknown'}):")
                print(f"  位置: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                print(f"  速度: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
            
            return True
            
    except Exception as e:
        print(f"❌ 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 清理资源
        try:
            if 'sim' in locals():
                gym.destroy_sim(sim)
            gym.release()
        except:
            pass

if __name__ == "__main__":
    success = test_t1_initialization()
    if success:
        print("\n🎉 T1初始化测试成功！")
        exit(0)
    else:
        print("\n💥 T1初始化测试失败！")
        exit(1)
