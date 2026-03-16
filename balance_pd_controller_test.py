import argparse
import math
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Cascade PD Balance Controller Test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video", action="store_true", help="비디오 녹화 활성화 (--video 입력 시 작동)") # [추가됨]
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Isaac Lab imports ─────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg # [추가됨] 카메라 센서
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from dataclasses import field
from lib.env.GOAT_base_env_cfg import GOAT_Cfg

# ─────────────────────────────────────────────────────────────────────────────
# Scene 설정
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
                fix_root_link=False,
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
    # 내부 PD 비활성화
    robot.actuators["hip"].stiffness    = 0.0
    robot.actuators["hip"].damping      = 0.0
    robot.actuators["thigh"].stiffness  = 0.0
    robot.actuators["thigh"].damping    = 0.0
    robot.actuators["knee"].stiffness   = 0.0
    robot.actuators["knee"].damping     = 0.0
    robot.actuators["wheel"].stiffness  = 0.0
    robot.actuators["wheel"].damping    = 0.0

    # 녹화용 카메라 센서 (방향은 main()에서 뷰포트 카메라와 동기화)
    record_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, f_stop=0.0, clipping_range=(0.1, 1.0e5)
        ),
    )


@configclass
class BalanceControllerConfig:
    wheel_radius: float = 72.75e-3
    kp_att: float    = 30.0
    kd_att: float    = 4.0
    tau_limit: float = 4.5
    kp_pos: float          = 1.0
    kd_pos: float          = 0.4
    theta_cmd_limit: float = math.radians(5.0)
    pitch_trim: float      = math.radians(0.0) 
    leg_kp: float        = 30.0
    leg_kd: float        = 2.0
    leg_tau_limit: float = 10.0
    leg_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    wheel_indices: list[int] = field(default_factory=lambda: [6, 7])


class BalancePDController:
    def __init__(self, cfg: BalanceControllerConfig, num_envs: int, device: str):
        self.cfg        = cfg
        self.num_envs   = num_envs
        self.device     = device
        self.num_joints = len(cfg.leg_indices) + len(cfg.wheel_indices)
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
        return theta, theta_dot

    def _position_control(self, r: torch.Tensor, r_dot: torch.Tensor, target_r: torch.Tensor) -> torch.Tensor:
        r_err = target_r - r
        theta_cmd = self.cfg.kp_pos * r_err + self.cfg.kd_pos * (0- r_dot)
        return torch.clamp(theta_cmd, -self.cfg.theta_cmd_limit, self.cfg.theta_cmd_limit)

    def _attitude_control(self, theta: torch.Tensor, theta_dot: torch.Tensor, theta_cmd: torch.Tensor) -> torch.Tensor:
        theta_err = (theta - self.cfg.pitch_trim) - theta_cmd
        tau = self.cfg.kp_att * theta_err + self.cfg.kd_att * theta_dot
        return torch.clamp(tau, -self.cfg.tau_limit, self.cfg.tau_limit)

    def _leg_position_hold(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        pos = joint_pos[:, self.leg_indices]
        vel = joint_vel[:, self.leg_indices]
        tgt = target_pos[:, self.leg_indices]
        tau = self.cfg.leg_kp * (tgt - pos) + self.cfg.leg_kd * (0.0 - vel)
        return torch.clamp(tau, -self.cfg.leg_tau_limit, self.cfg.leg_tau_limit)

    def compute_torque(
        self, joint_pos, joint_vel, body_pos, body_lin_vel, link_mass, target_leg_pos, r, r_dot, target_r, target_phi
    ):
        phi     = (joint_pos[:, self.wheel_indices[0]] - joint_pos[:, self.wheel_indices[1]]) / 2.0
        phi_dot = (joint_vel[:, self.wheel_indices[0]] - joint_vel[:, self.wheel_indices[1]]) / 2.0
        theta, theta_dot = self._COM_angle_cal(body_pos, body_lin_vel, link_mass)

        theta_cmd = self._position_control(r, r_dot, target_r)       
        wheel_tau = self._attitude_control(theta, theta_dot, theta_cmd)  
        leg_tau = self._leg_position_hold(joint_pos, joint_vel, target_leg_pos)

        torque = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        torque[:, self.leg_indices] = leg_tau         
        torque[:, self.wheel_indices[0]] = wheel_tau 
        torque[:, self.wheel_indices[1]] = -wheel_tau  

        phi = (joint_pos[:, self.wheel_indices[0]] - joint_pos[:, self.wheel_indices[1]]) / 2.0
        debug_states = {
            "r": r, "r_dot": r_dot, "phi": phi, "phi_dot": phi_dot,
            "theta": theta, "theta_dot": theta_dot, "wheel_tau": wheel_tau,
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

    sim_len = 10.0   

    # [수정됨] 60 FPS 및 리얼타임 설정
    sim.set_setting("/app/runLoops/main/realTime", True)
    render_interval = 8 # 500Hz 물리 / 62.5Hz 렌더링
    count = 0
    video_frames = [] # [추가됨] 비디오 프레임 버퍼

    controller_cfg = BalanceControllerConfig()
    controller = BalancePDController(controller_cfg, n_envs, device)

    scene.update(sim_dt)
    default_joint_pos = robot.data.default_joint_pos.clone().to(device)
    default_joint_vel = robot.data.default_joint_vel.clone().to(device)
    default_root_state = robot.data.default_root_state.clone().to(device)
    link_mass = robot.data.default_mass.clone().to(device)
    n_joint = robot.num_joints

    zero_torque = torch.zeros(n_envs, n_joint, device=device)

    robot.write_root_state_to_sim(default_root_state)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_effort_target(zero_torque)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt)

    log_t, log_r, log_phi, log_theta, log_r_dot, log_phi_dot, log_theta_dot, log_tau = [], [], [], [], [], [], [], []

    t = 0.0
    target_r   = torch.zeros(n_envs, device=device) 
    target_phi = torch.zeros(n_envs, device=device) 

    print("[INFO] Balance control started.")
    if args_cli.video:
        print("[INFO] Video recording is ENABLED. Frames will be saved at 60 FPS.")

    while t <= sim_len:
        # 1. 제어
        torque, debug_states = controller.compute_torque(
            joint_pos=robot.data.joint_pos,
            joint_vel=robot.data.joint_vel,
            body_pos=robot.data.body_com_pos_w,        
            body_lin_vel=robot.data.body_lin_vel_w, 
            link_mass=link_mass,   
            target_leg_pos=default_joint_pos,
            r=(robot.data.body_com_pos_w[:, 7, 0]+robot.data.body_com_pos_w[:, 8, 0])/2,
            r_dot=(robot.data.body_lin_vel_w[:, 7, 0]+robot.data.body_lin_vel_w[:, 8, 0])/2,
            target_r=target_r,
            target_phi=target_phi,
        )

        robot.set_joint_effort_target(torque)
        robot.write_data_to_sim()

        # 2. 60 FPS 렌더링 및 비디오 녹화 로직
        is_render_step = (count % render_interval == 0)
        sim.step(render=is_render_step)

        # 렌더링 스텝에서만 카메라를 수동 업데이트하여 프레임 캡처
        if args_cli.video and is_render_step:
            scene.sensors["record_camera"].update(sim_dt)
            rgb_data = scene.sensors["record_camera"].data.output["rgb"]

            # 카메라 버퍼가 비어있는 초기 스텝 무시
            if rgb_data is not None and rgb_data.numel() > 0:
                rgb_tensor = rgb_data[0].cpu()

                if rgb_tensor.dim() == 1:
                    rgb_tensor = rgb_tensor.view(480, 640, -1)

                if rgb_tensor.dim() == 3 and rgb_tensor.shape[2] >= 3:
                    frame = rgb_tensor[:, :, :3]
                    # float32 [0,1] 포맷이면 *255 스케일링, uint8이면 그대로 변환
                    if frame.is_floating_point():
                        frame = (frame * 255).clamp(0, 255).to(torch.uint8)
                    else:
                        frame = frame.to(torch.uint8)
                    video_frames.append(frame.clone())

        robot.update(sim_dt)

        # 3. 로그
        log_t.append(t)
        log_r.append(debug_states["r"][0].item())
        log_phi.append(debug_states["phi"][0].item())
        log_theta.append(debug_states["theta"][0].item())
        log_r_dot.append(debug_states["r_dot"][0].item())
        log_phi_dot.append(debug_states["phi_dot"][0].item()) 
        log_theta_dot.append(debug_states["theta_dot"][0].item())
        log_tau.append(debug_states["wheel_tau"][0].item())

        t += sim_dt
        count += 1

    print("[INFO] Simulation complete.")

    # 시뮬레이션 종료 후 버퍼에 쌓인 프레임들을 mp4로 인코딩하여 저장 (OpenCV 사용)
    if args_cli.video and len(video_frames) > 0:
        print("[INFO] Encoding video... Please wait.")
        output_path = "balance_robot_record.mp4"
        h, w = video_frames[0].shape[0], video_frames[0].shape[1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, 60, (w, h))
        for frame in video_frames:
            # RGB → BGR 변환 (OpenCV 포맷)
            bgr = cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        print(f"[INFO] Video saved successfully to {output_path}")

    # ── 플롯 (MATLAB Fig2와 동일 레이아웃) ──────────────────────────────────
    t_arr = np.array(log_t)
    cfg = controller_cfg 

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Cascade PD Balance Controller – Simulation", fontsize=13)

    axes[0, 0].plot(t_arr, np.array(log_r), label='r')
    axes[0, 0].axhline(float(target_r[0]), color='r', ls='--', label='r_cmd')
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

    # sim.set_camera_view()로 설정된 뷰포트 카메라 트랜스폼을 센서 카메라에 그대로 복사
    if args_cli.video:
        from pxr import UsdGeom, Gf, Usd
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        vp_prim  = stage.GetPrimAtPath("/OmniverseKit_Persp")
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/RecordCamera")
        if vp_prim.IsValid() and cam_prim.IsValid():
            world_mat = UsdGeom.Xformable(vp_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            xform = UsdGeom.Xformable(cam_prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(world_mat)
            print("[INFO] Sensor camera synced to viewport camera.")
        else:
            print("[WARN] Could not find viewport or sensor camera prim.")

    print("[INFO] Setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    # Isaac Sim 5.1 내부 버그 회피: omni.syntheticdata 플러그인이 Python finalizer 단계에서
    # 이미 해제된 OmniGraph 노드에 접근하여 access violation 발생.
    # close() 완료 후 os._exit()로 Python finalizer를 건너뛰어 크래시 방지.
    os._exit(0)
