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
        self.cfg        = cfg
        self.num_envs   = num_envs
        self.device     = device
        self.num_joints = len(cfg.leg_indices) + len(cfg.wheel_indices)

        # 인덱스를 텐서로 변환하여 GPU에서 효율적인 슬라이싱 가능하도록 함
        self.leg_indices   = torch.tensor(cfg.leg_indices, device=device, dtype=torch.long)
        self.wheel_indices = torch.tensor(cfg.wheel_indices, device=device, dtype=torch.long)

    def _COM_angle_cal(self, body_pos: torch.Tensor, body_lin_vel: torch.Tensor, link_mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        link_mass_expanded   = link_mass[:, 0:7].unsqueeze(-1)
        com                  = torch.sum(body_pos[:, 0:7, :]*link_mass_expanded, dim=1) / torch.sum(link_mass_expanded, dim=1)
        com_lin_vel          = torch.sum(body_lin_vel[:, 0:7, :]*link_mass_expanded, dim=1) / torch.sum(link_mass_expanded, dim=1)
        wheel_center         = torch.sum(body_pos[:, 7:9, :], dim=1)/2
        wheel_center_lin_vel = torch.sum(body_lin_vel[:, 7:9, :], dim=1)/2
        com_rel              = com - wheel_center
        com_rel_lin_vel      = com_lin_vel - wheel_center_lin_vel
        theta                = torch.atan2(com_rel[:, 0], com_rel[:, 2])
        theta_dot            = (com_rel_lin_vel[:, 0]*com_rel[:, 2] - com_rel_lin_vel[:, 2]*com_rel[:, 0])/((com_rel[:, 0])**2 + (com_rel[:, 2])**2+ 1e-6)
        L                    = torch.sqrt(com_rel[:, 0]**2 + com_rel[:, 2]**2)
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
        body_pos: torch.Tensor,
        body_lin_vel: torch.Tensor,
        link_mass: torch.Tensor,
        target_leg_pos: torch.Tensor,
        target_phi: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:

        # 1. 상태 변수 추출 (MATLAB 변수명과 대응)
        phi              = (joint_pos[:, self.wheel_indices[0]] - joint_pos[:, self.wheel_indices[1]]) / 2.0
        phi_dot          = (joint_vel[:, self.wheel_indices[0]] - joint_vel[:, self.wheel_indices[1]]) / 2.0
        theta, theta_dot, L = self._COM_angle_cal(body_pos, body_lin_vel, link_mass)
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
            body_pos=robot.data.body_com_pos_w,
            body_lin_vel=robot.data.body_lin_vel_w,
            link_mass=link_mass,
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


    
