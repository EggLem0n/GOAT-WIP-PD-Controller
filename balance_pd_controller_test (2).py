"""
Wheeled Inverted Pendulum – Cascade PD Balance Controller (Isaac Lab Simulation)

GOAT_PD_Ctrl_WIP.m 포팅

제어 구조 (MATLAB과 동일한 2중 루프 캐스케이드 PD):
    Outer loop (위치 제어): phi_cmd → theta_cmd
        theta_cmd = Kp_phi * (phi_cmd - phi) + Kd_phi * (0 - phi_dot)

    Inner loop (자세 제어): theta_cmd → tau
        tau = Kp_theta * (theta - theta_cmd) + Kd_theta * theta_dot

상태 변수:
    phi       : 바퀴 회전각 평균 [rad]          (wheel_L/R joint pos 평균)
    theta     : 본체 피치각, 수직 기준 0 [rad]   (root quaternion → pitch)
    phi_dot   : 바퀴 각속도 평균 [rad/s]
    theta_dot : 본체 피치 각속도 [rad/s]         (root_ang_vel_w y축)

비바퀴 관절 (hip/thigh/knee): 초기 위치 PD 고정
"""

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
            pos=(0.0, 0.0, 0.63),
            joint_pos={
                "hip_L_Joint": 0.0,
                "hip_R_Joint": 0.0,
                "thigh_L_Joint": 0.25,
                "thigh_R_Joint": -0.25,
                "knee_L_Joint": 0.6,
                "knee_R_Joint": -0.6,
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
    wheel_radius_m: float = 72.75e-3

    # Inner loop (자세 PD): theta_cmd -> wheel_tau
    kp_att: float = 30.0
    kd_att: float = 4.0
    tau_limit_nm: float = 15.0

    # Outer loop (위치 PD): r_cmd -> theta_cmd
    kp_pos: float = 0.03
    kd_pos: float = 0.05
    theta_cmd_limit_rad: float = math.radians(15.0)
    pitch_trim_rad: float = math.radians(3.70)

    # 다리 관절 위치 고정 PD
    leg_kp: float = 30.0
    leg_kd: float = 2.0
    leg_tau_limit_nm: float = 10.0

    # 목표 위치 (바퀴 기준)
    target_pos_m: float = 0.0

    # 관절 인덱스
    leg_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    wheel_indices: list[int] = field(default_factory=lambda: [6, 7])


class BalancePDController:
    """
    캐스케이드 PD 밸런스 컨트롤러 로직을 클래스로 캡슐화.
    모든 연산은 torch를 사용하여 GPU에서 효율적으로 수행됩니다.
    """

    def __init__(self, cfg: BalanceControllerConfig, num_envs: int, device: str):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.num_joints = len(cfg.leg_indices) + len(cfg.wheel_indices)

        # 인덱스를 텐서로 변환하여 GPU에서 효율적인 슬라이싱 가능하도록 함
        self.leg_indices = torch.tensor(cfg.leg_indices, device=device, dtype=torch.long)
        self.wheel_indices = torch.tensor(cfg.wheel_indices, device=device, dtype=torch.long)

    def _get_pitch_from_quat(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """루트 쿼터니언(w,x,y,z) → 본체 피치각 [rad]."""
        qw, qx, qy, qz = root_quat_w.split(1, dim=-1)
        sin_p = 2.0 * (qw * qy - qz * qx).squeeze(-1)
        sin_p = torch.clamp(sin_p, -1.0, 1.0)
        return torch.asin(sin_p)

    def _position_control(self, r_m: torch.Tensor, r_dot_m_per_s: torch.Tensor) -> torch.Tensor:
        """Outer loop PD: 바퀴 위치 오차 → theta_cmd."""
        r_err = self.cfg.target_pos_m - r_m
        r_dot_err = 0.0 - r_dot_m_per_s
        theta_cmd = -(self.cfg.kp_pos * r_err + self.cfg.kd_pos * r_dot_err)
        return torch.clamp(theta_cmd, -self.cfg.theta_cmd_limit_rad, self.cfg.theta_cmd_limit_rad)

    def _attitude_control(self, theta: torch.Tensor, theta_dot: torch.Tensor, theta_cmd: torch.Tensor) -> torch.Tensor:
        """Inner loop PD: 자세 오차 → 바퀴 토크."""
        theta_err = (theta - self.cfg.pitch_trim_rad) - theta_cmd
        tau = -(self.cfg.kp_att * theta_err + self.cfg.kd_att * theta_dot)
        return torch.clamp(tau, -self.cfg.tau_limit_nm, self.cfg.tau_limit_nm)

    def _leg_position_hold(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """비바퀴 관절 위치 고정 PD."""
        pos = joint_pos[:, self.leg_indices]
        vel = joint_vel[:, self.leg_indices]
        tgt = target_pos[:, self.leg_indices]
        tau = self.cfg.leg_kp * (tgt - pos) + self.cfg.leg_kd * (0.0 - vel)
        return torch.clamp(tau, -self.cfg.leg_tau_limit_nm, self.cfg.leg_tau_limit_nm)

    def compute_torque(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_quat_w: torch.Tensor,
        root_ang_vel_w: torch.Tensor,
        target_leg_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        로봇 상태를 입력받아 모든 관절에 대한 최종 토크 명령을 계산합니다.

        Returns:
            - torque (torch.Tensor): [num_envs, 8] 크기의 최종 토크 벡터.
            - debug_states (dict): 로깅 및 분석을 위한 중간 상태 변수 딕셔너리.
        """
        # 1. 상태 변수 추출 (MATLAB 변수명과 대응)
        # phi, phi_dot: 좌우 바퀴 평균
        phi = (-joint_pos[:, self.wheel_indices[0]] + joint_pos[:, self.wheel_indices[1]]) / 2.0
        phi_dot = (-joint_vel[:, self.wheel_indices[0]] + joint_vel[:, self.wheel_indices[1]]) / 2.0
        r_m = phi * self.cfg.wheel_radius_m
        r_dot_m_per_s = phi_dot * self.cfg.wheel_radius_m

        # theta, theta_dot: 본체 피치
        theta = self._get_pitch_from_quat(root_quat_w)
        theta_dot = root_ang_vel_w[:, 1]  # y축이 피치 회전축

        # 2. 캐스케이드 PD 제어
        theta_cmd = self._position_control(r_m, r_dot_m_per_s)       # Outer loop
        wheel_tau = self._attitude_control(theta, theta_dot, theta_cmd)  # Inner loop

        # 3. 다리 관절 위치 고정
        leg_tau = self._leg_position_hold(joint_pos, joint_vel, target_leg_pos)

        # 4. 전체 토크 벡터 조립 (효율적인 텐서 연산 사용)
        torque = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        # 바퀴 토크 할당 (양쪽 바퀴에 동일한 토크 적용)
        # 요청사항 반영: 왼쪽 바퀴의 제어 입력을 뒤집어 무한 회전 문제 해결
        torque[:, self.wheel_indices[0]] = -wheel_tau  # 왼쪽 바퀴
        torque[:, self.wheel_indices[1]] = wheel_tau  # 오른쪽 바퀴
        # 다리 토크 할당
        torque[:, self.leg_indices] = leg_tau

        # 5. 로깅을 위한 중간값 반환
        debug_states = {
            "phi": phi,
            "phi_dot": phi_dot,
            "theta": theta,
            "theta_dot": theta_dot,
            "wheel_tau": wheel_tau,
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
    controller = BalancePDController(controller_cfg, n_envs, device)

    # 초기 관절 상태 저장 (다리 고정 목표)
    robot.update(sim_dt)
    default_joint_pos = robot.data.default_joint_pos.clone()   # [n_envs, 8]
    default_joint_vel = robot.data.default_joint_vel.clone()   # [n_envs, 8]
    n_joint = robot.num_joints

    zero_torque = torch.zeros(n_envs, n_joint, device=device)

    # 리셋
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_effort_target(zero_torque)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt)

    # 로깅
    log_t         = []
    log_phi       = []   # 바퀴 회전각 (평균) [rad]
    log_theta     = []   # 본체 피치각 [rad]
    log_phi_dot   = []
    log_theta_dot = []
    log_tau       = []   # 바퀴 토크 [Nm]

    t = 0.0

    print("[INFO] Balance control started.")

    while t <= sim_len:
        # ── 제어 ───────────────────────────────────────────────────────────
        # 1. 컨트롤러에 현재 로봇 상태를 전달하여 토크 계산
        torque, debug_states = controller.compute_torque(
            joint_pos=robot.data.joint_pos,
            joint_vel=robot.data.joint_vel,
            root_quat_w=robot.data.root_quat_w,
            root_ang_vel_w=robot.data.root_ang_vel_w,
            target_leg_pos=default_joint_pos,
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
        log_phi.append(debug_states["phi"][0].item())
        log_theta.append(debug_states["theta"][0].item())
        log_phi_dot.append(debug_states["phi_dot"][0].item())
        log_theta_dot.append(debug_states["theta_dot"][0].item())
        log_tau.append(debug_states["wheel_tau"][0].item())

        t += sim_dt

    print("[INFO] Simulation complete. Plotting...")

    # ── 플롯 (MATLAB Fig2와 동일 레이아웃) ──────────────────────────────────
    import numpy as np
    t_arr = np.array(log_t)
    R2D = 180.0 / math.pi
    cfg = controller_cfg # 플로팅에 설정값 사용

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Cascade PD Balance Controller – Simulation", fontsize=13)

    axes[0, 0].plot(t_arr, np.array(log_phi) * cfg.wheel_radius_m)
    axes[0, 0].axhline(cfg.target_pos_m, color='r', ls='--', label='r_cmd')
    axes[0, 0].set_title('Position  r [m]')
    axes[0, 0].set_xlabel('t [s]'); axes[0, 0].set_ylabel('r [m]')
    axes[0, 0].grid(True); axes[0, 0].legend()

    axes[0, 1].plot(t_arr, np.array(log_theta) * R2D)
    axes[0, 1].axhline(0, color='r', ls='--', label='0 deg')
    axes[0, 1].set_title('Body Pitch  θ [deg]')
    axes[0, 1].set_xlabel('t [s]'); axes[0, 1].set_ylabel('θ [deg]')
    axes[0, 1].grid(True); axes[0, 1].legend()

    axes[0, 2].plot(t_arr, np.array(log_phi) * R2D)
    axes[0, 2].set_title('Wheel Angle  φ [deg]')
    axes[0, 2].set_xlabel('t [s]'); axes[0, 2].set_ylabel('φ [deg]')
    axes[0, 2].grid(True)

    axes[1, 0].plot(t_arr, np.array(log_phi_dot) * cfg.wheel_radius_m)
    axes[1, 0].set_title('Velocity  ṙ [m/s]')
    axes[1, 0].set_xlabel('t [s]'); axes[1, 0].set_ylabel('ṙ [m/s]')
    axes[1, 0].grid(True)

    axes[1, 1].plot(t_arr, np.array(log_theta_dot) * R2D)
    axes[1, 1].set_title('Pitch Rate  θ̇ [deg/s]')
    axes[1, 1].set_xlabel('t [s]'); axes[1, 1].set_ylabel('θ̇ [deg/s]')
    axes[1, 1].grid(True)

    axes[1, 2].plot(t_arr, log_tau)
    axes[1, 2].axhline( cfg.tau_limit_nm, color='r', ls='--', label='τ_max')
    axes[1, 2].axhline(-cfg.tau_limit_nm, color='r', ls='--', label='τ_min')
    axes[1, 2].set_title('Wheel Torque  τ [Nm]')
    axes[1, 2].set_xlabel('t [s]'); axes[1, 2].set_ylabel('τ [Nm]')
    axes[1, 2].grid(True); axes[1, 2].legend()

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
