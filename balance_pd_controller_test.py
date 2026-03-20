import argparse
import math
import torch
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Cascade PD Balance Controller Test.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab imports (AppLauncher 이후) ─────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from dataclasses import field
from lib.env.GOAT_base_env_cfg import GOAT_Cfg

# ─────────────────────────────────────────────────────────────────────────────
# Scene 설정  (fix_root_link=False, 중력 ON)
# ─────────────────────────────────────────────────────────────────────────────
@configclass
class BalanceSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = GOAT_Cfg.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=GOAT_Cfg.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                fix_root_link=False,        # 부유 로봇 (밸런스 필요)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.53),
            joint_pos={
                "hip_L_Joint": 0.0,
                "hip_R_Joint": 0.0,
                "thigh_L_Joint":  0.7382743,
                "thigh_R_Joint": -0.7382743,
                "knee_L_Joint":  1.46260337,
                "knee_R_Joint": -1.46260337,
                "wheel_L_Joint": 0.0,
                "wheel_R_Joint": 0.0,
                },
            ),
        )
    # 내부 PD 비활성화 → 외부 토크 제어
    robot.actuators["hip"].stiffness    = 0.0
    robot.actuators["hip"].damping      = 0.0
    robot.actuators["thigh"].stiffness  = 0.0
    robot.actuators["thigh"].damping    = 0.0
    robot.actuators["knee"].stiffness   = 0.0
    robot.actuators["knee"].damping     = 0.0
    robot.actuators["wheel"].stiffness  = 0.0
    robot.actuators["wheel"].damping    = 0.0


@configclass
class BalanceControllerConfig:
    """캐스케이드 PD 밸런스 컨트롤러 설정."""
    # 로봇 지오메트리   
    wheel_radius: float      = 72.75e-3
    com_height: float        = 0.25

    # Inner loop (자세 PD): theta_cmd -> wheel_tau
    kp_att: float            = 30.0
    kd_att: float            = 4.0
    tau_limit: float         = 4.5
    pitch_trim: float        = math.radians(0.0)

    # Outer loop (위치 PD): phi_comp -> theta_cmd
    kp_pos: float            = 0.1
    kd_pos: float            = 0.07125
    theta_cmd_limit: float   = math.radians(5.0)

    # 다리 관절 위치 고정 PD
    leg_kp: float            = 30.0
    leg_kd: float            = 2.0
    leg_tau_limit: float     = 4.5

    # 관절 인덱스
    leg_indices: list[int]   = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    wheel_indices: list[int] = field(default_factory=lambda: [6, 7])


class BalancePDController:

    def __init__(self, cfg: BalanceControllerConfig, num_envs: int, device: str):
        self.cfg           = cfg
        self.num_envs      = num_envs
        self.device        = device
        self.num_joints    = len(cfg.leg_indices) + len(cfg.wheel_indices)
        self.leg_indices   = torch.tensor(cfg.leg_indices, device=device, dtype=torch.long)
        self.wheel_indices = torch.tensor(cfg.wheel_indices, device=device, dtype=torch.long)
        self.initial_phi    = None

        def vec(x, y, z): return torch.tensor([x, y, z], device=device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)

        self.m_base = 3.075;   self.c_base = vec(-9.521e-3, 0.016e-3, -72.346e-3)
            
        self.m_hip = 0.316
        self.p_hip_L = vec(38.386e-3, 94.233e-3, -152.597e-3);  self.c_hip_L = vec(-48.092e-3, -38.296e-3, -0.162e-3)
        self.p_hip_R = vec(38.386e-3, -94.233e-3, -152.597e-3); self.c_hip_R = vec(-48.092e-3, 38.296e-3, -0.162e-3)
        
        self.m_thigh = 0.473
        self.p_thigh_L = vec(-54e-3, -17e-3, 0.0);  self.c_thigh_L = vec(-2.292e-3, 44.134e-3, -32.074e-3)
        self.p_thigh_R = vec(-54e-3, 17e-3, 0.0);   self.c_thigh_R = vec(-2.292e-3, -44.134e-3, -32.074e-3)
        
        self.m_calf = 0.321
        self.p_calf_L = vec(0.0, 18e-3, -205e-3);   self.c_calf_L = vec(1.922e-3, 0.564e-3, -169.664e-3)
        self.p_calf_R = vec(0.0, -18e-3, -205e-3);  self.c_calf_R = vec(1.922e-3, -0.564e-3, -169.664e-3)
        
        self.p_wheel_L = vec(0.0, 16.555e-3, -200e-3)
        self.p_wheel_R = vec(0.0, -16.555e-3, -200e-3)
            
        self.M_total = self.m_base + 2*(self.m_hip + self.m_thigh + self.m_calf)
    

    def _COM_angle_cal_FK(self, root_pos, root_quat, root_lin_vel, root_ang_vel, q, dq):
        """오직 IMU와 엔코더 정보만으로 실시간 CoM, 속도, 각도를 도출하는 정방향 기구학 엔진"""
        N = self.num_envs
        
        # 1. 쿼터니언 -> 회전 행렬 (Base)
        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        R_base = torch.zeros((N, 3, 3), device=self.device)
        R_base[:, 0, 0] = 1 - 2*(y**2 + z**2); R_base[:, 0, 1] = 2*(x*y - z*w);   R_base[:, 0, 2] = 2*(x*z + y*w)
        R_base[:, 1, 0] = 2*(x*y + z*w);       R_base[:, 1, 1] = 1 - 2*(x**2 + z**2); R_base[:, 1, 2] = 2*(y*z - x*w)
        R_base[:, 2, 0] = 2*(x*z - y*w);       R_base[:, 2, 1] = 2*(y*z + x*w);       R_base[:, 2, 2] = 1 - 2*(x**2 + y**2)

        P_base = root_pos
        V_base = root_lin_vel
        W_base = root_ang_vel

        # 벡터 및 회전 연산 도구
        def cross(w, r): return torch.cross(w, r, dim=1)
        def transf(R, v): return torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)
        def rot_x(ang, sign):
            R = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
            c = torch.cos(ang * sign); s = torch.sin(ang * sign)
            R[:, 1, 1] = c; R[:, 1, 2] = -s; R[:, 2, 1] = s; R[:, 2, 2] = c
            return R
        def rot_y(ang, sign):
            R = torch.eye(3, device=self.device).unsqueeze(0).repeat(N, 1, 1)
            c = torch.cos(ang * sign); s = torch.sin(ang * sign)
            R[:, 0, 0] = c; R[:, 0, 2] = s; R[:, 2, 0] = -s; R[:, 2, 2] = c
            return R

        # Base CoM
        P_com_base = P_base + transf(R_base, self.c_base)
        V_com_base = V_base + cross(W_base, P_com_base - P_base)

        # Hip L (axis: -X)
        P_hip_L = P_base + transf(R_base, self.p_hip_L)
        R_hip_L = torch.bmm(R_base, rot_x(q[:, 0], -1.0))
        W_hip_L = W_base + transf(R_base, torch.tensor([-1.,0.,0.], device=self.device).view(1,3) * dq[:, 0].unsqueeze(1))
        V_hip_L = V_base + cross(W_base, P_hip_L - P_base)
        P_com_hip_L = P_hip_L + transf(R_hip_L, self.c_hip_L)
        V_com_hip_L = V_hip_L + cross(W_hip_L, P_com_hip_L - P_hip_L)

        # Hip R (axis: -X)
        P_hip_R = P_base + transf(R_base, self.p_hip_R)
        R_hip_R = torch.bmm(R_base, rot_x(q[:, 1], -1.0))
        W_hip_R = W_base + transf(R_base, torch.tensor([-1.,0.,0.], device=self.device).view(1,3) * dq[:, 1].unsqueeze(1))
        V_hip_R = V_base + cross(W_base, P_hip_R - P_base)
        P_com_hip_R = P_hip_R + transf(R_hip_R, self.c_hip_R)
        V_com_hip_R = V_hip_R + cross(W_hip_R, P_com_hip_R - P_hip_R)

        # Thigh L (axis: +Y)
        P_thigh_L = P_hip_L + transf(R_hip_L, self.p_thigh_L)
        R_thigh_L = torch.bmm(R_hip_L, rot_y(q[:, 2], 1.0))
        W_thigh_L = W_hip_L + transf(R_hip_L, torch.tensor([0.,1.,0.], device=self.device).view(1,3) * dq[:, 2].unsqueeze(1))
        V_thigh_L = V_hip_L + cross(W_hip_L, P_thigh_L - P_hip_L)
        P_com_thigh_L = P_thigh_L + transf(R_thigh_L, self.c_thigh_L)
        V_com_thigh_L = V_thigh_L + cross(W_thigh_L, P_com_thigh_L - P_thigh_L)

        # Thigh R (axis: -Y)
        P_thigh_R = P_hip_R + transf(R_hip_R, self.p_thigh_R)
        R_thigh_R = torch.bmm(R_hip_R, rot_y(q[:, 3], -1.0))
        W_thigh_R = W_hip_R + transf(R_hip_R, torch.tensor([0.,-1.,0.], device=self.device).view(1,3) * dq[:, 3].unsqueeze(1))
        V_thigh_R = V_hip_R + cross(W_hip_R, P_thigh_R - P_hip_R)
        P_com_thigh_R = P_thigh_R + transf(R_thigh_R, self.c_thigh_R)
        V_com_thigh_R = V_thigh_R + cross(W_thigh_R, P_com_thigh_R - P_thigh_R)

        # Calf L (axis: -Y)
        P_calf_L = P_thigh_L + transf(R_thigh_L, self.p_calf_L)
        R_calf_L = torch.bmm(R_thigh_L, rot_y(q[:, 4], -1.0))
        W_calf_L = W_thigh_L + transf(R_thigh_L, torch.tensor([0.,-1.,0.], device=self.device).view(1,3) * dq[:, 4].unsqueeze(1))
        V_calf_L = V_thigh_L + cross(W_thigh_L, P_calf_L - P_thigh_L)
        P_com_calf_L = P_calf_L + transf(R_calf_L, self.c_calf_L)
        V_com_calf_L = V_calf_L + cross(W_calf_L, P_com_calf_L - P_calf_L)

        # Calf R (axis: +Y)
        P_calf_R = P_thigh_R + transf(R_thigh_R, self.p_calf_R)
        R_calf_R = torch.bmm(R_thigh_R, rot_y(q[:, 5], 1.0))
        W_calf_R = W_thigh_R + transf(R_thigh_R, torch.tensor([0.,1.,0.], device=self.device).view(1,3) * dq[:, 5].unsqueeze(1))
        V_calf_R = V_thigh_R + cross(W_thigh_R, P_calf_R - P_thigh_R)
        P_com_calf_R = P_calf_R + transf(R_calf_R, self.c_calf_R)
        V_com_calf_R = V_calf_R + cross(W_calf_R, P_com_calf_R - P_calf_R)

        # Wheel Centers
        P_wheel_L = P_calf_L + transf(R_calf_L, self.p_wheel_L)
        V_wheel_L = V_calf_L + cross(W_calf_L, P_wheel_L - P_calf_L)
        
        P_wheel_R = P_calf_R + transf(R_calf_R, self.p_wheel_R)
        V_wheel_R = V_calf_R + cross(W_calf_R, P_wheel_R - P_calf_R)

        # 전체 질량 중심점 (바퀴 제외)
        P_com_total = (self.m_base*P_com_base + self.m_hip*(P_com_hip_L + P_com_hip_R) + 
                       self.m_thigh*(P_com_thigh_L + P_com_thigh_R) + self.m_calf*(P_com_calf_L + P_com_calf_R)) / self.M_total
        V_com_total = (self.m_base*V_com_base + self.m_hip*(V_com_hip_L + V_com_hip_R) + 
                       self.m_thigh*(V_com_thigh_L + V_com_thigh_R) + self.m_calf*(V_com_calf_L + V_com_calf_R)) / self.M_total

        # 바퀴 평균 위치
        P_wheel_avg = (P_wheel_L + P_wheel_R) / 2.0
        V_wheel_avg = (V_wheel_L + V_wheel_R) / 2.0

        # 상대 위치/속도를 이용해 역진자 파라미터 계산
        P_rel = P_com_total - P_wheel_avg
        V_rel = V_com_total - V_wheel_avg

        theta = torch.atan2(P_rel[:, 0], P_rel[:, 2])
        theta_dot = (V_rel[:, 0]*P_rel[:, 2] - V_rel[:, 2]*P_rel[:, 0]) / (P_rel[:, 0]**2 + P_rel[:, 2]**2 + 1e-6)
        L = torch.sqrt(P_rel[:, 0]**2 + P_rel[:, 2]**2)

        return theta, theta_dot, L


    def _position_control(self, phi: torch.Tensor, phi_dot: torch.Tensor, 
                                theta: torch.Tensor, theta_dot: torch.Tensor, 
                                target_phi: torch.Tensor,
                                L: torch.Tensor) -> torch.Tensor:
        ratio = L / self.cfg.wheel_radius
        phi_comp     = phi + ratio*torch.sin(theta)
        phi_comp_dot = phi_dot + ratio*torch.cos(theta)*theta_dot
        phi_err      = target_phi - phi_comp
        theta_cmd    = self.cfg.kp_pos * phi_err + self.cfg.kd_pos * (0.0 - phi_comp_dot)
        return torch.clamp(theta_cmd, -self.cfg.theta_cmd_limit, self.cfg.theta_cmd_limit)

    def _attitude_control(self, theta: torch.Tensor, theta_dot: torch.Tensor, theta_cmd: torch.Tensor) -> torch.Tensor:
        theta_err = (theta - self.cfg.pitch_trim) - theta_cmd
        wheel_tau = self.cfg.kp_att * theta_err + self.cfg.kd_att * theta_dot
        return torch.clamp(wheel_tau, -self.cfg.tau_limit, self.cfg.tau_limit)

    def _leg_position_hold(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        pos = joint_pos[:, self.leg_indices]
        vel = joint_vel[:, self.leg_indices]
        tgt = target_pos[:, self.leg_indices]
        tau = self.cfg.leg_kp * (tgt - pos) + self.cfg.leg_kd * (0.0 - vel)
        return torch.clamp(tau, -self.cfg.leg_tau_limit, self.cfg.leg_tau_limit)

    def compute_torque(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        body_pos: torch.Tensor,
        body_lin_vel: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        target_leg_pos: torch.Tensor,
        target_phi: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        # 1. 상태 변수 추출 (MATLAB 변수명과 대응)
        raw_phi          = (joint_pos[:, self.wheel_indices[0]] - joint_pos[:, self.wheel_indices[1]]) / 2.0
        if self.initial_phi is None:
            self.initial_phi = raw_phi.clone()
        phi              = raw_phi - self.initial_phi
        phi_dot          = (joint_vel[:, self.wheel_indices[0]] - joint_vel[:, self.wheel_indices[1]]) / 2.0
        theta, theta_dot, L = self._COM_angle_cal_FK(
            root_pos, root_quat, root_lin_vel, root_ang_vel, 
            joint_pos[:, self.leg_indices], joint_vel[:, self.leg_indices]
        )
        r                = (body_pos[:, 7, 0] + body_pos[:, 8, 0]) / 2.0
        r_dot            = (body_lin_vel[:, 7, 0] + body_lin_vel[:, 8, 0]) / 2.0

        # 2. 캐스케이드 PD 제어
        theta_cmd = self._position_control(phi, phi_dot, theta, theta_dot, target_phi, L)  # Outer loop
        wheel_tau = self._attitude_control(theta, theta_dot, theta_cmd)                 # Inner loop

        # 3. 다리 관절 위치 고정
        leg_tau = self._leg_position_hold(joint_pos, joint_vel, target_leg_pos)

        # 4. 전체 토크 벡터 조립
        torque = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        torque[:, self.leg_indices]      = leg_tau     # 다리 토크 할당
        torque[:, self.wheel_indices[0]] = wheel_tau   # 왼쪽 바퀴
        torque[:, self.wheel_indices[1]] = -wheel_tau  # 오른쪽 바퀴



        # 5. 로깅을 위한 중간값 반환
        debug_states = {
            "r": r,
            "r_dot": r_dot,
            "phi": phi,
            "phi_dot": phi_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "wheel_tau": wheel_tau,
            "leg_tau": leg_tau,
        }
        return torque, debug_states


# ─────────────────────────────────────────────────────────────────────────────
# 메인 시뮬레이션 루프
# ─────────────────────────────────────────────────────────────────────────────
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot   = scene["robot"]
    sim_dt  = sim.get_physics_dt()
    device  = scene.device
    n_envs  = scene.num_envs

    sim_len = 10.0   # 총 시뮬 시간 [s]

    # 컨트롤러 설정 및 초기화
    controller_cfg = BalanceControllerConfig()
    controller     = BalancePDController(controller_cfg, n_envs, device)

    # 초기 관절 상태 저장 (다리 고정 목표)
    robot.update(sim_dt)
    default_joint_pos  = robot.data.default_joint_pos.clone().to(device)   # [n_envs, 8]
    default_joint_vel  = robot.data.default_joint_vel.clone().to(device)   # [n_envs, 8]
    default_root_state = robot.data.default_root_state.clone().to(device)
    link_mass          = robot.data.default_mass.clone().to(device)
    n_joint            = robot.num_joints

    zero_torque = torch.zeros(n_envs, n_joint, device=device)

    # 리셋
    robot.write_root_state_to_sim(default_root_state)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_effort_target(zero_torque)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt)

    # 로깅
    log_t         = []
    log_r         = []
    log_phi       = []   # 바퀴 회전각 (평균) [rad]
    log_theta     = []   # 본체 피치각 [rad]
    log_r_dot     = []
    log_phi_dot   = []
    log_theta_dot = []
    log_tau       = []   # 바퀴 토크 [Nm]
    log_leg_tau   = []   # 다리 관절 토크 [Nm], shape: [T, 6]

    t = 0.0
    target_phi = torch.zeros(n_envs, device=device) # 목표 바퀴 회전각

    print("[INFO] Balance control started.")

    while t <= sim_len:
        # ── 제어 ───────────────────────────────────────────────────────────
        # 1. 컨트롤러에 현재 로봇 상태를 전달하여 토크 계산
        torque, debug_states = controller.compute_torque(
            joint_pos=robot.data.joint_pos,
            joint_vel=robot.data.joint_vel,
            root_pos=robot.data.root_pos_w,         
            root_quat=robot.data.root_quat_w,
            body_pos=robot.data.body_pos_w,
            body_lin_vel=robot.data.body_lin_vel_w,       
            root_lin_vel=robot.data.root_lin_vel_w, 
            root_ang_vel=robot.data.root_ang_vel_w,
            target_leg_pos=default_joint_pos,
            target_phi=target_phi,
        )

        # 2. 계산된 토크를 시뮬레이션에 적용
        robot.set_joint_effort_target(torque)
        robot.write_data_to_sim()

        # 3. 시뮬레이션 스텝 진행
        sim.step()
        robot.update(sim_dt)
        scene.update(sim_dt)

        # ── 로그 ───────────────────────────────────────────────────────────
        log_t.append(t)
        log_r.append(debug_states["r"][0].item())
        log_phi.append(debug_states["phi"][0].item())
        log_theta.append(debug_states["theta"][0].item())
        log_r_dot.append(debug_states["r_dot"][0].item())
        log_phi_dot.append(debug_states["phi_dot"][0].item())
        log_theta_dot.append(debug_states["theta_dot"][0].item())
        log_tau.append(debug_states["wheel_tau"][0].item())
        log_leg_tau.append(debug_states["leg_tau"][0].cpu().tolist())

        t += sim_dt

    print("[INFO] Simulation complete. Plotting...")

    # ── 플롯 (MATLAB Fig2와 동일 레이아웃) ──────────────────────────────────
    import numpy as np
    t_arr      = np.array(log_t)
    leg_tau_arr = np.array(log_leg_tau)   # [T, 6]: hip_L/R, thigh_L/R, knee_L/R
    cfg = controller_cfg

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Cascade PD Balance Controller – Simulation", fontsize=13)

    axes[0, 0].plot(t_arr, np.array(log_r), label='r')
    axes[0, 0].set_title('Position  r [m]')
    axes[0, 0].set_xlabel('t [s]'); axes[0, 0].set_ylabel('r [m]')
    axes[0, 0].grid(True); axes[0, 0].legend()

    axes[0, 1].plot(t_arr, np.degrees(np.array(log_theta)))
    axes[0, 1].axhline(math.degrees(cfg.pitch_trim), color='r', ls='--', label='theta_cmd')
    axes[0, 1].set_title('Body Pitch  θ [deg]')
    axes[0, 1].set_xlabel('t [s]'); axes[0, 1].set_ylabel('θ [deg]')
    axes[0, 1].grid(True); axes[0, 1].legend()

    axes[0, 2].plot(t_arr, np.degrees(np.array(log_phi)))
    axes[0, 2].axhline(math.degrees(float(target_phi[0])), color='r', ls='--', label='phi_des')
    axes[0, 2].set_title('Wheel Angle  φ [deg]')
    axes[0, 2].set_xlabel('t [s]'); axes[0, 2].set_ylabel('φ [deg]')
    axes[0, 2].grid(True); axes[0, 2].legend()

    axes[1, 0].plot(t_arr, np.array(log_r_dot))
    axes[1, 0].set_title('Velocity  ṙ [m/s]')
    axes[1, 0].set_xlabel('t [s]'); axes[1, 0].set_ylabel('ṙ [m/s]')
    axes[1, 0].grid(True)

    axes[1, 1].plot(t_arr, np.degrees(np.array(log_theta_dot)))
    axes[1, 1].set_title('Pitch Rate  θ̇ [deg/s]')
    axes[1, 1].set_xlabel('t [s]'); axes[1, 1].set_ylabel('θ̇ [deg/s]')
    axes[1, 1].grid(True)

    axes[1, 2].plot(t_arr, log_tau)
    axes[1, 2].axhline( cfg.tau_limit, color='r', ls='--', label='τ_max')
    axes[1, 2].axhline(-cfg.tau_limit, color='r', ls='--', label='τ_min')
    axes[1, 2].set_title('Wheel Torque  τ [Nm]')
    axes[1, 2].set_xlabel('t [s]'); axes[1, 2].set_ylabel('τ [Nm]')
    axes[1, 2].grid(True); axes[1, 2].legend()

    axes[2, 0].plot(t_arr, leg_tau_arr[:, 0], label='hip_L')
    axes[2, 0].plot(t_arr, leg_tau_arr[:, 1], label='hip_R')
    axes[2, 0].axhline( cfg.leg_tau_limit, color='r', ls='--')
    axes[2, 0].axhline(-cfg.leg_tau_limit, color='r', ls='--')
    axes[2, 0].set_title('Hip Torque [Nm]')
    axes[2, 0].set_xlabel('t [s]'); axes[2, 0].set_ylabel('τ [Nm]')
    axes[2, 0].grid(True); axes[2, 0].legend()

    axes[2, 1].plot(t_arr, leg_tau_arr[:, 2], label='thigh_L')
    axes[2, 1].plot(t_arr, leg_tau_arr[:, 3], label='thigh_R')
    axes[2, 1].axhline( cfg.leg_tau_limit, color='r', ls='--')
    axes[2, 1].axhline(-cfg.leg_tau_limit, color='r', ls='--')
    axes[2, 1].set_title('Thigh Torque [Nm]')
    axes[2, 1].set_xlabel('t [s]'); axes[2, 1].set_ylabel('τ [Nm]')
    axes[2, 1].grid(True); axes[2, 1].legend()

    axes[2, 2].plot(t_arr, leg_tau_arr[:, 4], label='knee_L')
    axes[2, 2].plot(t_arr, leg_tau_arr[:, 5], label='knee_R')
    axes[2, 2].axhline( cfg.leg_tau_limit, color='r', ls='--')
    axes[2, 2].axhline(-cfg.leg_tau_limit, color='r', ls='--')
    axes[2, 2].set_title('Knee Torque [Nm]')
    axes[2, 2].set_xlabel('t [s]'); axes[2, 2].set_ylabel('τ [Nm]')
    axes[2, 2].grid(True); axes[2, 2].legend()

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.002, device=args_cli.device)
    sim     = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 1.5, 1.5], [0.0, 0.0, 0.3])

    scene_cfg = BalanceSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene     = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()


    
