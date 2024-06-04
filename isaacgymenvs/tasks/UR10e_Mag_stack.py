# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import numpy as np
import os
import torch
import csv

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym import gymutil
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask

def quat_to_rot_mat(q):
    """ Convert a quaternion into a rotation matrix.
    Args:
        q (torch.Tensor): a tensor of quaternions, shape [*, 4], where * can be any shape
    Returns:
        torch.Tensor: a tensor of 3x3 rotation matrices, shape [*, 3, 3]
    """
    # Ensure the quaternion is normalized
    q = q / q.norm(p=2, dim=-1, keepdim=True)

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    # Compute rotation matrix elements
    r00 = 2 * (ww + xx) - 1
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = 2 * (ww + yy) -1
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 2 * (ww + zz) -1

    # Stack into a single tensor
    rot_mat = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    return rot_mat

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

class Magnet:

    u0 = 4 * np.pi * 1e-7
    def __init__(self,ma_norm,mc_norm,num_env,device):
        self.ma_norm = ma_norm
        self.mc_norm = mc_norm
        self.num_env = num_env
        self.device = device
    def Z(self):
        I = torch.eye(3).repeat(self.num_env, 1, 1).to(self.device)

        p_hat_outer = torch.bmm(self.p_hat, self.p_hat_T)
        return I - 5 * p_hat_outer

    def D(self):
        I = torch.eye(3).repeat(self.num_env, 1, 1).to(self.device)
        p_hat_outer = torch.bmm(self.p_hat, self.p_hat_T)

        return 3 * p_hat_outer - I
    
    def calculate_mc_hat(self):

        temp = torch.bmm(self.D(), self.ma_hat)
        temp_norm = torch.norm(temp,dim=1,p=2, keepdim=True)

        target_mc_hat = temp / temp_norm
        return target_mc_hat
    
    def magnet_update(self, p, ma,mc):

        
        self.ma_hat = ma / torch.norm(ma, p=2, dim=1, keepdim=True)
        self.ma_hat_T = torch.transpose(ma,1,2)
        self.mc_hat = mc / torch.norm(mc, p=2, dim=1, keepdim=True)
        self.mc_hat_T = torch.transpose(mc,1,2)

        self.p = p
        self.p_norm =torch.norm(self.p,dim=1,p=2,keepdim=True)
        self.p_hat = self.p / self.p_norm
        # 行向量 
        self.p_hat_T = torch.transpose(self.p_hat,1,2)
        # 得到mc的目标朝向
        self.target_mc_hat = self.calculate_mc_hat()
    
    
    def get_magnetic_force(self):
        I = torch.eye(3).repeat(self.num_env, 1, 1).to(self.device)

        """consider ma_hat and mc_hat"""
        magnetic_force = 3 * self.u0 * self.ma_norm * self.mc_norm/(4 * np.pi * (self.p_norm**4)) * torch.bmm((torch.bmm(self.ma_hat,self.mc_hat_T)+torch.bmm(self.mc_hat, self.ma_hat_T) + (torch.bmm(torch.bmm(self.mc_hat_T, self.Z()), self.ma_hat))*I), self.p_hat)
        # magnetic_force = (3 * u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_delta_norm ** 4)) *
        #      np.dot((np.outer(self.ma_hat, self.mc_hat) + np.outer(self.mc_hat, self.ma_hat) +
        #              ((self.mc_hat.T.dot(self.Z())).dot(self.ma_hat)) * I), self.p_delta_hat))

        # print("magnetic_force:", magnetic_force)
        """just consider ma_hat"""
        # D_ma_norm = torch.norm(torch.bmm(self.D(),self.ma_hat),dim=1,p=2,keepdim=True)
        # magnetic_force = (3 * self.u0 * self.ma_norm * self.mc_norm)/(4 * np.pi * (self.p_norm ** 4) * D_ma_norm)*torch.bmm((torch.bmm(self.ma_hat,self.ma_hat_T)-(1+4*(torch.bmm(self.ma_hat_T,self.p_hat)**2))*I),self.p_hat)
        # magnetic_force_norm = np.linalg.norm(magnetic_force)
        # magnetic_force_hat = magnetic_force / magnetic_force_norm

        # #print("magnetic_force_hat:", magnetic_force_hat)
        # if magnetic_force_norm > 0.5:
        #     #magnetic_force = 0.9 * self.ma_hat
        #     magnetic_force = 0.5* magnetic_force_hat
        #print((torch.bmm(torch.bmm(self.mc_hat_T, self.Z()), self.ma_hat)))
        return magnetic_force
    
    def get_magnetic_torque(self):
        magnetic_torque = self.u0 * self.ma_norm * self.mc_norm / (4 * np.pi * (self.p_norm ** 3)) * torch.bmm(torch.cross(self.mc_hat, self.D()), self.ma_hat)

        # temp = torch.bmm(torch.cross(self.mc_hat, self.D()), self.ma_hat).squeeze(-1).cpu().numpy()
        # with open('temp_data.txt', 'a') as file:
        #     np.savetxt(file, temp)
        
        return magnetic_torque
    
    def get_buoyancy_force(self):

        buoyancy_force = torch.tensor(self.num_env*[0, 0, 0.036764]).view(self.num_env,3,1).to(self.device)
        return buoyancy_force
    
    def total_force(self):


        total_force = self.get_magnetic_force() + self.get_buoyancy_force()
        return total_force
    
def generate_random_coordinates(num_env):
    x = torch.rand(num_env) * 0.5  # X coordinate in the range 0 to 0.5
    y = torch.rand(num_env) * 1 - 0.5  # Y coordinate in the range -0.5 to 0.5
    z = torch.rand(num_env) * 0.3 + 1.1  # Z coordinate in the range 1.1 to 1.3
    return torch.stack([x, y, z], dim=1)

def generate_random_coordinates_around(num_env, current_pos, device, distance=0.02):
    """
    在当前点位置周围生成随机目标点,距离为distance。

    参数:
    - current_pos: 当前点的位置，形状为(num_env, 3)的Tensor。
    - distance: 目标点距离当前点的距离,默认值为0.02。

    返回:
    - target_pos: 生成的目标点，形状为(num_env, 3)的Tensor。
    """
    # 随机生成方向向量
    direction = torch.randn(num_env, 3).to(device)

    # 将方向向量标准化
    direction = direction / torch.norm(direction, dim=-1, keepdim=True)

    # 生成随机点在当前位置周围的目标点
    target_pos = current_pos + direction * distance

    return target_pos

def generate_random_point(num_env):
    x = torch.rand(num_env) -0.5
    y = torch.rand(num_env) -0.5
    z = torch.rand(num_env) -0.5
    point = torch.stack([x, y, z], dim=1)
    point_norm = torch.norm(point,dim=1,p=2, keepdim=True)
    point_hat = point/point_norm

    return point_hat

def generate_trajectory(num_points,num_env):
    trajectory = []
    radius = 0.2
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = torch.tensor(num_env*[0.3 + radius * np.cos(angle)])
        y = torch.tensor(num_env*[radius * np.sin(angle)])
        z = torch.tensor(num_env*[1.12])
        trajectory.append(torch.stack([x, y, z], dim=1))  # 目标位置（x, y, z）
    return torch.stack(trajectory, dim=0).permute(1, 0, 2)

class UR10eMagStack(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # 设置最大单次时长
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        
        # 设置 action scale和噪声等
        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.UR10e_position_noise = self.cfg["env"]["UR10ePositionNoise"]
        self.UR10e_rotation_noise = self.cfg["env"]["UR10eRotationNoise"]
        self.UR10e_dof_noise = self.cfg["env"]["UR10eDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # TODO: 更改奖励设置
        # Create dicts to pass to reward function
        # 设置奖励scale
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "contact_scale": self.cfg["env"]["contactRewardScale"],
            "ori_scale": self.cfg["env"]["oriRewardScale"],
            "energy_scale": self.cfg["env"]["energyRewardScale"],
        }

        # 设置控制模式，末端执行器控制还是关节角控制
        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor", "joint_pos","osc_pos"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor, joint_pose, osc_pos}"

        # dimensions
        # obs include: cubeA_pose (7) + cubeB_pos (3) + eef_pose (7) + q_gripper (2)
        # TODO:
        # 设置观测的维度，本项目为obj_pos(3) + obj_vel(3) + DOF_pos(6) + DOF_vel(6)

        # 原来是42,28
        self.cfg["env"]["numObservations"] = 50
        # actions include: delta EEF if OSC (6) or joint torques (7) + bool gripper (1)
        #self.cfg["env"]["numActions"] = 6 if self.control_type == "osc" else 5
        self.cfg["env"]["numActions"] = 6 if self.control_type == "osc_pos" else 6
        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        # 关节角数
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed


        # self._init_cubeA_state = None           # Initial state of cubeA for the current env
        # self._cubeA_state = None                # Current state of cubeA for the current env
        # self._cubeA_id = None                   # Actor ID corresponding to cubeA for a given env


        self._init_capsule_state = None           # Initial state of cubeA for the current env
        self._capsule_state = None                # Current state of cubeA for the current env
        self._capsule_id = None                   # Actor ID corresponding to cubeA for a given env

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_state = None
        self.pre_actions = None
        # self._eef_state = None  # end effector state (at grasping point)
        # self._eef_lf_state = None  # end effector state (at left fingertip)
        # self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        # self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        # self._effort_control = None         # Torque actions
        # self._UR10e_effort_limits = None        # Actuator effort limits for UR10e
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # UR10e defaults
        self.UR10e_default_dof_pos = to_torch(
            [-0.294, -1.650, 2.141, -2.062, -1.572, 1.277], device=self.device
        )
        """set target position"""
        self.point_num = torch.zeros(self.num_envs).to(self.device)
        self.trajectory = generate_trajectory(200,self.num_envs).to(self.device)
        self.target_pos_fre = 15

        self.target_pos = torch.tensor(self.num_envs*[0.2,0.0,1.3]).to(self.device).view(self.num_envs,3)
        """set target velocity"""
        self.target_vel = torch.tensor(self.num_envs*[0.0,0.0,0.0]).to(self.device).view(self.num_envs,3)
        """set target capsule point"""
        self.target_point = torch.tensor(self.num_envs*[0.0,0.0,-1.0]).to(self.device).view(self.num_envs,3)
        # Set control limits
        # TODO:根据position控制修改
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc_pos" else self._UR10e_effort_limits[:6].unsqueeze(0)

        self.action_fre = 4
        
        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        self.mag = Magnet(82.5, 0.1664, self.num_envs, self.device)
        # Refresh tensors
        self._refresh()


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        print("create sim successfully")

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)


        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        UR10e_asset_file = "urdf/UR10e_description/robot/ur10e.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            UR10e_asset_file = self.cfg["env"]["asset"].get("assetFileNameUR10e", UR10e_asset_file)

        # load UR10e asset
        
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        #asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS if \
        self.control_type == "osc_pos" else gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        UR10e_asset = self.gym.load_asset(self.sim, asset_root, UR10e_asset_file, asset_options)
        self.UR10e_link_dict = self.gym.get_asset_rigid_body_dict(UR10e_asset)
        UR10e_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device) if \
        self.control_type == "osc_pos" else to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        UR10e_dof_damping = to_torch([40, 40, 40, 40, 40, 40], dtype=torch.float, device=self.device) if \
        self.control_type == "osc_pos" else to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)
        UR10e_dof_velocity = to_torch([0.8, 0.8, 0.8, 0.8, 0.8, 0.8], dtype=torch.float, device=self.device)
        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_opts.disable_gravity = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # load capsule
        capsule_asset_file = "urdf/UR10e_description/mc_capsule/urdf/mc_capsule.urdf"
        self.capsule_r, self.capsule_h = 0.013/2, 0.03
        capsule_opts = gymapi.AssetOptions()
        capsule_opts.linear_damping = 15
        capsule_opts.angular_damping = 5
        capsule_opts.max_linear_velocity = 1
        capsule_opts.max_angular_velocity = 10

        capsule_opts.disable_gravity = False
        capsule_opts.flip_visual_attachments = True
        # capsule_opts.override_com = True
        # capsule_opts.override_inertia = True
        capsule_asset = self.gym.load_asset(self.sim, asset_root, capsule_asset_file, capsule_opts)
        capsule_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # Create cubeA asset
        # cubeA_opts = gymapi.AssetOptions()
        # cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        # cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # 得到连杆数量，算上基座8
        self.num_UR10e_bodies = self.gym.get_asset_rigid_body_count(UR10e_asset)
        # 活动关节数，6
        self.num_UR10e_dofs = self.gym.get_asset_dof_count(UR10e_asset)

        print("num UR10e bodies: ", self.num_UR10e_bodies)
        print("num UR10e dofs: ", self.num_UR10e_dofs)

        # set UR10e dof properties
        # 获取机械臂属性
        UR10e_dof_props = self.gym.get_asset_dof_properties(UR10e_asset)
        
        self.UR10e_dof_lower_limits = []
        self.UR10e_dof_upper_limits = []
        self._UR10e_effort_limits = []
        # self.UR10e_velocity_limits = []
        for i in range(self.num_UR10e_dofs):

            UR10e_dof_props['velocity'][i] = UR10e_dof_velocity[i]
            if self.control_type == "osc_pose":
                UR10e_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS 
            else :
                UR10e_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT

            if self.physics_engine == gymapi.SIM_PHYSX:
                UR10e_dof_props['stiffness'][i] = UR10e_dof_stiffness[i]
                UR10e_dof_props['damping'][i] = UR10e_dof_damping[i]
            else:
                UR10e_dof_props['stiffness'][i] = 7000.0
                UR10e_dof_props['damping'][i] = 50.0

            self.UR10e_dof_lower_limits.append(UR10e_dof_props['lower'][i])
            self.UR10e_dof_upper_limits.append(UR10e_dof_props['upper'][i])
            self._UR10e_effort_limits.append(UR10e_dof_props['effort'][i])

        self.UR10e_dof_lower_limits = to_torch(self.UR10e_dof_lower_limits, device=self.device)
        self.UR10e_dof_upper_limits = to_torch(self.UR10e_dof_upper_limits, device=self.device)
        self._UR10e_effort_limits = to_torch(self._UR10e_effort_limits, device=self.device)
        self.UR10e_dof_speed_scales = torch.ones_like(self.UR10e_dof_lower_limits)
        # self.UR10e_dof_speed_scales[[7, 8]] = 0.1
        # UR10e_dof_props['effort'][7] = 200
        # UR10e_dof_props['effort'][8] = 200

        # Define start pose for UR10e
        UR10e_start_pose = gymapi.Transform()
        UR10e_start_pose.p = gymapi.Vec3(-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        UR10e_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        capsule_start_pose = gymapi.Transform()
        capsule_start_pose.p = gymapi.Vec3(-1.0, 0.0, 1.0 + table_thickness / 2 + self.capsule_r + 0.01)
        capsule_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # TODO: 需要考虑ma，mc
        # compute aggregate size
        num_UR10e_bodies = self.gym.get_asset_rigid_body_count(UR10e_asset)
        num_UR10e_shapes = self.gym.get_asset_rigid_shape_count(UR10e_asset)
        max_agg_bodies = num_UR10e_bodies + 3     # 1 for table, table stand, capsule
        max_agg_shapes = num_UR10e_shapes + 3     # 1 for table, table stand, capsule

        self.UR10es_handles = []
        self.envs_handles = []
        self.capsules_idxs = []
        self.ma_idxs = []
        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: UR10e should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create UR10e
            # Potentially randomize start pose
            if self.UR10e_position_noise > 0:
                rand_xy = self.UR10e_position_noise * (-1. + np.random.rand(2) * 2.0)
                UR10e_start_pose.p = gymapi.Vec3(-0.5 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.UR10e_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.UR10e_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                UR10e_start_pose.r = gymapi.Quat(*new_quat)
            # 0-7
            self.UR10e_actor = self.gym.create_actor(env_ptr, UR10e_asset, UR10e_start_pose, "ur10e", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, self.UR10e_actor, UR10e_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            # 8
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            # 9
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)
            # TODO：将aggregate设置为1，即将机械臂和ma聚合在一起，需要在此之前将ma创建
            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)


            # Create cubes
            # 10
            self.capsule_actor = self.gym.create_actor(env_ptr, capsule_asset, capsule_start_pose, "capsule", i, 2, 0)
            self.gym.set_rigid_body_color(env_ptr, self.capsule_actor, 0, gymapi.MESH_VISUAL, capsule_color)
            capsule_idx = self.gym.get_actor_rigid_body_index(env_ptr, self.capsule_actor, 0, gymapi.DOMAIN_SIM)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            
            ma_handle = self.gym.find_actor_rigid_body_handle(env_ptr, self.UR10e_actor, "ma")
            _ma_idx = self.gym.find_actor_rigid_body_index(env_ptr, self.UR10e_actor, "ma", gymapi.DOMAIN_SIM)
            # Store the created env pointers

            # 得到 
            self.envs_handles.append(env_ptr)
            # [num env],[0,0,0,0....]
            self.UR10es_handles.append(self.UR10e_actor)
            self.ma_idxs.append(_ma_idx)
            self.capsules_idxs.append(capsule_idx) 
        
        self._init_capsule_state = torch.zeros(self.num_envs, 13, device=self.device)
    
        # Setup data
        self.init_data()
        #print("env create successfully")

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs_handles[0]
        UR10e_handle = 0

        # TODO：获取句柄，将ma，和mc获取出来
        self.handles = {
            # UR10e

             "ma_handle": self.gym.find_actor_rigid_body_handle(env_ptr, UR10e_handle, "ma"),

            "capsule_body_handle": self.gym.find_actor_rigid_body_handle(env_ptr, self.capsule_actor, "capsule"),
        }

        # Get total DOFs
        # 获得总的关节数量
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        
        # Setup tensor buffers
        # (num_actors, 13) actor：机械臂，桌子，mc
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        # print("_actor_root_state_tensor",_actor_root_state_tensor)
        # (num_dofs, 2) dof：机械臂的关节角
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        # (num_rigid_bodies, 13) rigid_bodies：和root类似
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # (num envs, num_actors, 13)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        # (num envs, num_dofs, 2)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        # (num envs, num_rigid_bodies, 13)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        # (num envs, num_dofs, 1)表示位置
        self._q = self._dof_state[..., 0]
        # (num envs, num_dofs, 1)表示速度
        self._qd = self._dof_state[..., 1]
        # (num envs, ma_index, 13)
        # 碰撞矩阵[num_env,body_num,3]
        self._contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self._contact_state = gymtorch.wrap_tensor(self._contact_tensor).view(self.num_envs,-1,3)
        """ma的状态"""
        self._eef_state = self._rigid_body_state[:, self.handles["ma_handle"], :]
        UR10e_lastlink_index = self.UR10e_link_dict["ma"]
        "jacobian矩阵"
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur10e")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self._j_eef = jacobian[:, UR10e_lastlink_index-1, :, :6]
        # (num envs, num_actors, 13)
        """获取胶囊的state,形状为[num_env,capsule_actor,13]"""
        self._capsule_state = self._root_state[:, self.capsule_actor, :]
        mc_basic_vector = torch.tensor(self.num_envs*[1.0,0.0,0]).to(self.device).view(self.num_envs,3)
        self._mc_hat = quat_rotate(normalize(self._capsule_state[:, 3:7]),mc_basic_vector).view(self.num_envs,3,1)
        """获取桌面碰撞state"""
        self._table_contact = self._contact_state[:,8,:]
        self._ma_contact = self._contact_state[:,7,:]
        """action"""
        self.pre_actions = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        # Initialize states
        """mass"""
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "ur10e")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :6, :6]
        self.states.update({
            "capsule_heights": torch.ones_like(self._capsule_state[:, 0]) * self.capsule_r ,
            # "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })

        # Initialize actions
        # (num_envs, num dofs) (num_envs, 6)
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)


        #self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control

        self._arm_control = self._pos_control[:, :6]

        # 创建了一个 (num envs,4)，每个环境创造4个索引,四个东西
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)


    def _update_states(self):
        # for force and torque
        mc_basic_vector = torch.tensor(self.num_envs*[0.0,1.0,0.0]).to(self.device).view(self.num_envs,3)
        self.mc_hat = quat_rotate(normalize(self._capsule_state[:, 3:7]),mc_basic_vector).view(self.num_envs,3,1)
        ma_basic_vector = torch.tensor(self.num_envs*[0.0,1.0,0.0]).to(self.device).view(self.num_envs,3)
        self.ma_hat = quat_rotate(normalize(self._eef_state[:, 3:7]),ma_basic_vector).view(self.num_envs,3,1)
        
        # for reward
        self._mc_hat = quat_rotate(normalize(self._capsule_state[:, 3:7]),mc_basic_vector).view(self.num_envs,3)
        #self.target_pos[:,0] =  self.target_pos[:,0] + 0.00025

        """生成轨迹点"""
        # temp = (self.progress_buf % self.target_pos_fre == 0).int().squeeze(-1).clone()
        # self.point_num = self.point_num + temp
        # self.point_num = self.point_num.long()
        # indices = self.point_num.unsqueeze(1).unsqueeze(2).expand(self.num_envs, 1, 3)
        # indices = torch.clamp(indices, 0, 199)
        # self.target_pos = torch.gather(self.trajectory, 1, indices).squeeze(1).float()
        
        #print(self.target_pos)
        #print(self.point_num) 
        # print(self.target_pos)
        self.states.update({
            # 50 + 6
            # UR10e 12 + 13  25
            "q_pos": self._q[:, :],
            "q_vel": self._qd[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],

            # 10 + 3 + 3 + 3 + 3 + 3   25
            "capsule_pos" : self._capsule_state[:, :3],
            "capsule_quat" : self._capsule_state[:, 3:7],
            "capsule_linear_vel"  : self._capsule_state[:, 7:10],
            "capsule_angle_vel"  : self._capsule_state[:, 10:13],
            "capsule_pos_relative": self._capsule_state[:, :3] - self._eef_state[:, :3],
            "mc_hat": self._mc_hat,
            # target
            "target_point": self.target_point,
            "target_pos": self.target_pos,
            "target_vel": self.target_vel,
            # contact
            "table_contact": self._table_contact,
            "ma_contact": self._ma_contact,
            # action 6
            "previous_actions": self.pre_actions
        })
        # print(self._table_contact.shape)
        self.mag.magnet_update(self.states["capsule_pos_relative"].view(self.num_envs,3,1),self.ma_hat,self.mc_hat)
        

        

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

        mag_force = self.mag.total_force().squeeze(-1)
        self.forces = torch.zeros((self.num_envs,11, 3), device=self.device, dtype=torch.float)
        self.forces[:,10,:] = mag_force
        mag_torque = self.mag.get_magnetic_torque().squeeze(-1)
        self.torques = torch.zeros((self.num_envs,11, 3), device=self.device, dtype=torch.float)
        self.torques[:,10,:] = mag_torque


    # TODO：待修改
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_UR10e_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, self.max_episode_length
        )

    # TODO: 修改成ma和mc，加上末端执行器（如果是osc），若是dof_position则修改成ma，mc，joint_pos + osc_pos
    def compute_observations(self):
        self._refresh()
        # obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        # obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        # self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)
        obs = [ "q_pos", "q_vel", "eef_pos", "eef_quat",  "eef_vel", "capsule_pos", "capsule_quat", 
               "capsule_linear_vel", "capsule_pos_relative","mc_hat","target_pos","target_vel","target_point"]
        # 28
        # obs = [ "q_pos", "q_vel", "eef_pos", "eef_quat",  "eef_vel", "target_pos"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)


        """写入文件"""
        # # 创建 capsule_pos 数据的拷贝
        # capsule_pos_data = self.states["capsule_pos"][23].clone().detach().cpu().numpy()  # 转换为 NumPy 数组以便于写入 CSV
        # if capsule_pos_data.ndim == 1:
        #     capsule_pos_data = capsule_pos_data.reshape(-1, 1)
        # capsule_pos_list = capsule_pos_data.tolist()
        # # 将 capsule_pos 数据写入 CSV 文件
        # with open('capsule_pos.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     #writer.writerow(["capsule_pos"])  # 写入表头
        #     writer.writerows(capsule_pos_list)  # 写入数据

        #         # 创建 capsule_pos 数据的拷贝
        # target_pos_data = self.states["target_pos"][23].clone().detach().cpu().numpy()  # 转换为 NumPy 数组以便于写入 CSV
        # if target_pos_data.ndim == 1:
        #     target_pos_data = target_pos_data.reshape(-1, 1)
        # target_pos_list = target_pos_data.tolist()
        # # 将 capsule_pos 数据写入 CSV 文件
        # with open('target_pos.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     #writer.writerow(["target_pos"])  # 写入表头
        #     writer.writerows(target_pos_list)  # 写入数据


        # 暂时不知道有什么用
        maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf

    def reset_idx(self, env_ids):
        
        """生成随机三维坐标"""
        self.target_pos[env_ids] = generate_random_coordinates(self.num_envs).to(self.device)[env_ids]
        #print("target_pos",self.target_pos[0])
        # self.target_point = generate_random_point(self.num_envs).to(self.device)
        # [0 -- 1023]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # 生成轨迹时用的
        self.point_num[env_ids] = 0
        self._reset_init_cube_state(env_ids=env_ids, check_valid=False)
        #self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=True)
        # self._i = True

        # Write these new init states to the sim states
        self._capsule_state[env_ids] = self._init_capsule_state[env_ids]
        #self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 6), device=self.device)
        # 确保偏移量不会超出上下限 TODO：是否加入噪声要重新探讨
        # [num env, 6]
        pos = tensor_clamp(
            self.UR10e_default_dof_pos.unsqueeze(0) +
            self.UR10e_dof_noise * 2.0 * (reset_noise - 0.5),
            self.UR10e_dof_lower_limits.unsqueeze(0), self.UR10e_dof_upper_limits)

        self._q[env_ids, :] = pos
        
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        #self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        # len:1024,[0,5,10 .....]
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update cube states
        
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, 3].flatten()

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_init_cube_state(self, env_ids, check_valid=True):
        """
        Simple method to sample @cube's position based on self.startPositionNoise and self.startRotationNoise, and
        automaticlly reset the pose internally. Populates the appropriate self._init_cubeX_state

        If @check_valid is True, then this will also make sure that the sampled position is not in contact with the
        other cube.

        Args:
            cube(str): Which cube to sample location for. Either 'A' or 'B'
            env_ids (tensor or None): Specific environments to reset cube for
            check_valid (bool): Whether to make sure sampled position is collision-free with the other cube.
        """
        
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values 位置，方向，线速度，角速度
        num_resets = len(env_ids)
        # [num env, 13]
        sampled_obj_state = torch.zeros(num_resets, 13, device=self.device)
        
        # # Get correct references depending on which one was selected
        # if cube.lower() == 'a':
        # [num env, 13]
        this_obj_state_all = self._init_capsule_state
        # [num env]
        obj_heights = self.states["capsule_heights"]
        
        # elif cube.lower() == 'b':
        #     this_cube_state_all = self._init_cubeB_state
        #     other_cube_state = self._init_cubeA_state[env_ids, :]
        #     cube_heights = self.states["cubeA_size"]
        # else:
        #     raise ValueError(f"Invalid cube specified, options are 'A' and 'B'; got: {cube}")

        # Minimum cube distance for guarenteed collision-free sampling is the sum of each cube's effective radius
        # min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # We scale the min dist by 2 so that the cubes aren't too close together
        # min_dists = min_dists * 2.0

        # Sampling is "centered" around middle of table
        # [2]
        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)
        
        # Set z value, which is fixed height
        sampled_obj_state[:, 2] = self._table_surface_pos[2] + obj_heights.squeeze(-1)[env_ids] / 2

        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_obj_state[:, 6] = 1.0

        # If we're verifying valid sampling, we need to check and re-sample if any are not collision-free
        # We use a simple heuristic of checking based on cubes' radius to determine if a collision would occur
        # if check_valid:
        #     success = False
        #     # Indexes corresponding to envs we're still actively sampling for
        #     active_idx = torch.arange(num_resets, device=self.device)
        #     num_active_idx = len(active_idx)
        #     for i in range(100):
        #         # Sample x y values
        #         sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
        #                                              2.0 * self.start_position_noise * (
        #                                                      torch.rand_like(sampled_obj_state[active_idx, :2]) - 0.5)
        #         # Check if sampled values are valid
        #         cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
        #         active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
        #         num_active_idx = len(active_idx)
        #         # If active idx is empty, then all sampling is valid :D
        #         if num_active_idx == 0:
        #             success = True
        #             break
        #     # Make sure we succeeded at sampling
        #     assert success, "Sampling cube locations was unsuccessful! ):"
        # else:
            # We just directly sample
        sampled_obj_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # Sample rotation value
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_obj_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_obj_state[:, 3:7])

        # Lastly, set these sampled values as the new init state
        this_obj_state_all[env_ids, :] = sampled_obj_state
        # [num env, 13]
        

    # TODO:不需要该函数
    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :6], self._qd[:, :6]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.UR10e_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 6:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(6, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._UR10e_effort_limits[:6].unsqueeze(0), self._UR10e_effort_limits[:6].unsqueeze(0))

        return u

    def _control_ik(self, dpose):
    # solve damped least squares

        
        damping = 0.05
        j_eef_T = torch.transpose(self._j_eef, 1, 2)

        lmbda = torch.eye(6, device= self.device) * (damping ** 2)

        dpose = dpose.unsqueeze(-1)
        u = (j_eef_T @ torch.inverse(self._j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)


        return u

    def pre_physics_step(self, actions):
        

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.torques), gymapi.ENV_SPACE)
        self.actions = actions.clone().to(self.device)
        osc_dp = self.actions[:, :]
        
        mask = (self.progress_buf % self.action_fre == 0).float().unsqueeze(-1)
        
        # Control arm (scale value first)
        osc_dp = osc_dp * self.cmd_limit / self.action_scale
        
        if self.control_type == "osc_pos":
            u_arm = self._control_ik(dpose=osc_dp)*mask
            self._arm_control[:, :] = u_arm + self.states["q_pos"]
            self._arm_control = tensor_clamp(self._arm_control.squeeze(-1),
            self.UR10e_dof_lower_limits.unsqueeze(0), self.UR10e_dof_upper_limits.unsqueeze(0))
        # print("arm_control shape",self._arm_control.shape)
        # Deploy actions
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._arm_control))
        else:
            u_arm = self._compute_osc_torques(dpose=osc_dp)*mask
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(u_arm))

        
        



    def post_physics_step(self):
        self.progress_buf += 1
        # print(self.target_pos)
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        #self.pre_actions = self.actions
        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            capsule_pos = self.states["capsule_pos"]
            capsule_rot = self.states["capsule_quat"]
            # cubeB_pos = self.states["cubeB_pos"]
            # cubeB_rot = self.states["cubeB_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, capsule_pos), (eef_rot, capsule_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


#@torch.jit.script
def compute_UR10e_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    """distance reward"""
    diff = states["target_pos"] - states["capsule_pos"]

    # 计算距离
    d = torch.norm(diff, dim=-1)

    # 计算每个轴上的差异
    diff_x = torch.abs(diff[:, 0])
    diff_y = torch.abs(diff[:, 1])
    diff_z = torch.abs(diff[:, 2])

    diff_xy = torch.sqrt(torch.pow(diff_x,2) + torch.pow(diff_x,2))
    # 距离奖励
    distance_z_reward = 1 - torch.tanh(10.0 * (diff_z) / 3.0)
    distance_y_reward = 1 - torch.tanh(10.0 * (diff_y) / 0.5)
    distance_x_reward = 1 - torch.tanh(10.0 * (diff_x) / 0.5)
    distance_xy_reward = 1 - torch.tanh(10.0 * (diff_xy) / 0.5)
    distance_reward = -torch.tanh(10.0 * (d) / 3.0)
    #print(distance_xy_reward)
    # 轴对齐惩罚（示例：惩罚x轴上的大误差）
    #axis_penalty = torch.tanh(torch.abs(diff_x)) + torch.tanh(torch.abs(diff_y))
   
    # 远近奖励（近距离高奖励，远距离低奖励）
    near_reward = torch.exp(-d / 0.5)  # 距离越近，奖励越高

    # 总奖励函数
    d_reward =  distance_reward 


    
    """contact reward"""
    contact1 = torch.norm(states["table_contact"],dim=-1)
    contact1_reward = (contact1 > 0).float()*(1.0)
 
    
    contact2 = torch.norm(states["ma_contact"],dim=-1)
    contact2_reward = (contact2 > 0).float()*(1.0)

    """vel reward"""
    v = torch.norm(states["capsule_linear_vel"], dim=-1)
    v_delta = torch.norm(states["capsule_linear_vel"] - states["target_vel"], dim=-1)
    v_reward = -torch.tanh(10.0 * (v_delta) / 3.0)
    v_penalty = -torch.ones_like(v_reward)
    """ori reward"""
    point_delta = torch.norm(states["mc_hat"] - states["target_point"], dim=-1)
    p_reward = -torch.tanh(10.0 * (point_delta) / 3)

    """energy consumption reward"""
    a = torch.norm(actions, dim=-1)
    energy_reward = -torch.tanh(6.0 * (a) / 3)

    # """smoothness reward"""
    # action_diff = torch.norm(actions - states["previous_actions"], dim=-1)
    # smoothness_reward = 1 - torch.tanh(10.0 * action_diff / 3)


    """process reward"""
    reach_reward = reward_settings["r_dist_scale"]*d_reward  + reward_settings["contact_scale"]*(contact1_reward+contact2_reward) + \
        reward_settings["ori_scale"]*p_reward + reward_settings["energy_scale"]*energy_reward

    reach = (d < 0.5)
    """keep reward"""
    
    # print("target_pos:",states["target_pos"][23])
    # print("capsule_pos:",states["capsule_pos"][23])
    # print("d",d[23])
    # print("v",v[23])
    # print("v_reward",v_reward[23])
    keep_reward = v_penalty
    
    keep_reward = torch.where((contact1 == 0.0) & reach,torch.zeros_like(keep_reward) ,keep_reward)
    
    # keep_reward = torch.zeros_like(v_reward)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (contact2 > 0), torch.ones_like(reset_buf), reset_buf)
    rewards = reach_reward + keep_reward

    return rewards, reset_buf
