#!/usr/bin/env python3

import os
import sys
import numpy as np
sys.path.append('/home/user/ccz/DHAL')

# 必须先导入isaacgym，再导入torch
from isaacgym import gymapi, gymutil
import torch

def test_initialization_with_params(base_height, skateboard_height, joint_angles=None):
    """测试特定参数组合的初始化"""
    try:
        from legged_gym.envs.t1.t1_config import T1Cfg
        from legged_gym import LEGGED_GYM_ROOT_DIR
        
        # 动态修改配置
        cfg = T1Cfg()
        
        # 修改基座高度
        cfg.init_state.pos = [0.0, 0.0, base_height]
        
        # 修改滑板高度
        cfg.init_state.default_joint_angles['skateboard_joint_y'] = skateboard_height
        
        # 如果提供了自定义关节角度，应用它们
        if joint_angles:
            for joint, angle in joint_angles.items():
                cfg.init_state.default_joint_angles[joint] = angle
        
        # 设置简单环境
        cfg.env.num_envs = 1
        cfg.terrain.mesh_type = 'plane'
        
        # 宽松的物理参数
        cfg.sim.dt = 0.01  # 更大的时间步长
        cfg.sim.physx.num_position_iterations = 2  # 降低精度要求
        cfg.sim.physx.num_velocity_iterations = 1
        
        print(f"\n=== 测试配置 ===")
        print(f"基座高度: {base_height}m")
        print(f"滑板高度: {skateboard_height}m")
        print(f"脚部预期高度: {base_height - 0.4:.2f}m (假设腿长0.4m)")
        print(f"脚-滑板距离: {abs((base_height - 0.4) - skateboard_height):.2f}m")
        
        # 尝试创建环境
        from legged_gym.utils.helpers import get_args
        from legged_gym import LEGGED_GYM_ROOT_DIR
        from legged_gym.envs.base.legged_robot import LeggedRobot
        from isaacgym import gymutil, gymapi
        
        # 创建仿真参数
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = cfg.sim.dt
        sim_params.substeps = cfg.sim.substeps
        
        # PhysX参数
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = cfg.sim.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = cfg.sim.physx.num_velocity_iterations
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 0.2
        sim_params.physx.max_depenetration_velocity = 1.0
        sim_params.physx.default_buffer_size_multiplier = 5
        
        sim_params.use_gpu_pipeline = True
        
        print("创建环境...")
        env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda:0', headless=True)
        
        print("检查初始状态...")
        # 手动触发一次状态更新
        env.gym.refresh_rigid_body_state_tensor(env.sim)
        
        rigid_body_states = env.rigid_body_states
        print(f"刚体状态张量形状: {rigid_body_states.shape}")
        
        has_nan = torch.isnan(rigid_body_states).any()
        print(f"包含NaN: {has_nan}")
        
        if not has_nan:
            print("✅ 初始化成功！")
            
            # 打印一些关键刚体的位置
            print("\n关键刚体位置:")
            if hasattr(env, 'feet_indices') and len(env.feet_indices) > 0:
                feet_pos = rigid_body_states[0, env.feet_indices, :3]
                print(f"脚部位置: {feet_pos}")
                
            if hasattr(env, 'skateboard_deck_indices') and len(env.skateboard_deck_indices) > 0:
                skateboard_pos = rigid_body_states[0, env.skateboard_deck_indices, :3]
                print(f"滑板位置: {skateboard_pos}")
            
            # 计算实际的脚-滑板距离
            if hasattr(env, 'feet_indices') and hasattr(env, 'marker_link_indices'):
                if len(env.feet_indices) > 0 and len(env.marker_link_indices) > 0:
                    feet_pos = rigid_body_states[0, env.feet_indices, :3]
                    marker_pos = rigid_body_states[0, env.marker_link_indices, :3]
                    distances = torch.norm(marker_pos - feet_pos, dim=-1)
                    print(f"脚-标记点距离: {distances}")
            
            return True, cfg
        else:
            print("❌ 初始状态包含NaN值")
            
            # 找出哪些刚体包含NaN
            nan_mask = torch.isnan(rigid_body_states).any(dim=-1).any(dim=0)
            nan_indices = torch.where(nan_mask)[0].cpu().numpy()
            print(f"NaN刚体索引: {nan_indices.tolist()}")
            
            return False, None
            
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def explore_configurations():
    """系统地探索不同的配置组合"""
    print("=== 系统探索T1初始化配置 ===\n")
    
    # 测试用例：基座高度和滑板高度的组合
    test_cases = [
        # (基座高度, 滑板高度, 描述)
        (0.38, 0.0, "Go1原始高度 + 滑板在地面"),
        (0.6, 0.2, "中等高度 + 滑板稍高"),
        (0.8, 0.4, "较高基座 + 滑板较高"),
        (1.0, 0.6, "高基座 + 滑板高"),
        (0.5, 0.1, "低一点的配置"),
        (0.4, 0.0, "接近Go1的配置"),
        (0.7, 0.3, "另一个中等配置"),
    ]
    
    successful_configs = []
    
    for i, (base_h, skate_h, desc) in enumerate(test_cases):
        print(f"\n--- 测试 {i+1}/{len(test_cases)}: {desc} ---")
        success, cfg = test_initialization_with_params(base_h, skate_h)
        
        if success:
            successful_configs.append((base_h, skate_h, desc, cfg))
            print(f"🎉 找到可行配置: 基座{base_h}m, 滑板{skate_h}m")
        
        print("-" * 50)
    
    if successful_configs:
        print(f"\n✅ 找到 {len(successful_configs)} 个可行配置:")
        for base_h, skate_h, desc, cfg in successful_configs:
            print(f"  - {desc}: 基座{base_h}m, 滑板{skate_h}m")
        
        # 返回第一个成功的配置
        return successful_configs[0]
    else:
        print("\n❌ 未找到任何可行配置")
        return None

def test_joint_angles():
    """测试不同的关节角度配置"""
    print("\n=== 测试关节角度配置 ===")
    
    # 使用一个基本可行的高度配置（如果之前找到的话）
    base_configs = [
        (0.38, 0.0),  # Go1风格
        (0.5, 0.1),   # 保守配置
    ]
    
    # 不同的关节角度组合
    joint_configs = [
        {
            "name": "零角度配置",
            "angles": {
                'Left_Hip_Pitch': 0.0,
                'Right_Hip_Pitch': 0.0,
                'Left_Knee_Pitch': 0.0,
                'Right_Knee_Pitch': 0.0,
                'Left_Ankle_Pitch': 0.0,
                'Right_Ankle_Pitch': 0.0,
            }
        },
        {
            "name": "轻微弯曲配置",
            "angles": {
                'Left_Hip_Pitch': 0.1,
                'Right_Hip_Pitch': 0.1,
                'Left_Knee_Pitch': 0.2,
                'Right_Knee_Pitch': 0.2,
                'Left_Ankle_Pitch': -0.1,
                'Right_Ankle_Pitch': -0.1,
            }
        },
        {
            "name": "Go1风格腿部配置",
            "angles": {
                'Left_Hip_Pitch': 0.1,
                'Right_Hip_Pitch': -0.1,
                'Left_Knee_Pitch': 1.2,
                'Right_Knee_Pitch': 1.2,
                'Left_Ankle_Pitch': -1.2,
                'Right_Ankle_Pitch': -1.2,
            }
        }
    ]
    
    for base_h, skate_h in base_configs:
        print(f"\n基座配置: {base_h}m, 滑板: {skate_h}m")
        
        for joint_config in joint_configs:
            print(f"  测试: {joint_config['name']}")
            success, cfg = test_initialization_with_params(
                base_h, skate_h, joint_config['angles']
            )
            if success:
                print(f"    ✅ 成功!")
                return base_h, skate_h, joint_config['angles']
            else:
                print(f"    ❌ 失败")
    
    return None

if __name__ == "__main__":
    print("T1机器人初始化配置探索器")
    print("=" * 50)
    
    # 首先探索基本的高度配置
    result = explore_configurations()
    
    if result:
        print(f"\n使用最佳配置进行详细测试...")
        base_h, skate_h, desc, cfg = result
        
        # 尝试测试关节角度
        joint_result = test_joint_angles()
        if joint_result:
            base_h, skate_h, joint_angles = joint_result
            print(f"\n🎉 最终推荐配置:")
            print(f"  基座高度: {base_h}m")
            print(f"  滑板高度: {skate_h}m") 
            print(f"  关节角度: {joint_angles}")
    else:
        print("\n需要进一步调试URDF或物理参数...")
