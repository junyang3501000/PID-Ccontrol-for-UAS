import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np



ROOT = Path(__file__).resolve().parent
p = None


@dataclass
class ParamSpec:
    name: str
    key: str
    indices: tuple | None
    low: float
    high: float

# 注释：角度归一化函数：将角度限制在 [-pi, pi] 范围内。
def wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

# 注释：加载目标点文件：从 CSV 文件中读取目标点坐标和航向。
def load_targets(path):
    targets = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) != 4:
                continue
            target = tuple(float(v) for v in row)
            if target[2] >= 0.0:
                targets.append(target)
    return targets or [(0.0, 0.0, 1.0, 0.0)]

# 注释：导入控制器模块：根据控制器名称选择并加载相应的控制器模块。
def import_controller_module(controller_name):
    if controller_name == "plan_a":
        path = ROOT / "controller.py"
        module_name = "controller_plan_a_eval"
    elif controller_name == "cascade":
        path = ROOT / "controller_cascade.py"
        module_name = "controller_cascade_eval"
    else:
        raise ValueError(f"unknown controller: {controller_name}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    spec.loader.exec_module(module)

    # 自动调参时不需要实时绘图和控制器内部日志，避免 GUI/文件 IO 影响评分速度。
    if hasattr(module, "_send_error_plot"):
        module._send_error_plot = lambda errors: None
    if hasattr(module, "_write_flight_log"):
        module._write_flight_log = lambda *args, **kwargs: None
    return module

# 注释：获取基础参数：根据控制器名称和风干扰状态，加载对应的控制器模块并返回其默认参数。
def get_base_params(controller_name, wind_enabled):
    module = import_controller_module(controller_name)
    return deepcopy(module._make_params(wind_enabled))

# 注释：获取参数值：根据参数字典和参数规范，获取指定参数的当前值。
def get_param(params, spec):
    value = params[spec.key]
    if spec.indices is None:
        return float(value)
    return float(value[spec.indices[0]])

# 注释：设置参数值：根据参数字典和参数规范，设置指定参数的当前值。
def set_param(params, spec, value):
    if spec.indices is None:
        params[spec.key] = float(value)
        return
    for idx in spec.indices:
        params[spec.key][idx] = float(value)

# 注释：生成 plan_a 控制器的参数规范：根据基础参数，生成对应的参数规范。
def make_plan_a_specs(base):
    return [
        ParamSpec("kp_far_xy", "kp_far", (0, 1), 0.45, 1.35),
        ParamSpec("kp_near_xy", "kp_near", (0, 1), 0.20, 0.80),
        ParamSpec("ki_xy", "ki", (0, 1), 0.00, 0.35),
        ParamSpec("kv_far_xy", "kv_far", (0, 1), 0.02, 0.35),
        ParamSpec("kv_near_xy", "kv_near", (0, 1), 0.15, 0.85),
        ParamSpec("schedule_xy", "schedule_radius", (0, 1), 0.18, 0.70),
        ParamSpec("v_cap_far_xy", "v_cap_far", (0, 1), 0.45, 1.00),
        ParamSpec("v_cap_near_xy", "v_cap_near", (0, 1), 0.18, 0.55),
        ParamSpec("max_rate_xy", "max_rate_xyz", (0, 1), 1.2, 5.0),
        ParamSpec("kp_near_z", "kp_near", (2,), 0.20, 0.75),
        ParamSpec("ki_z", "ki", (2,), 0.00, 0.35),
        ParamSpec("kv_near_z", "kv_near", (2,), 0.18, 0.90),
        ParamSpec("v_cap_near_z", "v_cap_near", (2,), 0.12, 0.45),
        ParamSpec("kp_near_yaw", "kp_near", (3,), 0.25, 1.20),
        ParamSpec("ki_yaw", "ki", (3,), 0.00, 0.08),
        ParamSpec("kv_near_yaw", "kv_near", (3,), 0.05, 0.35),
        ParamSpec("yaw_rate_cap_near", "yaw_rate_cap_near", None, 0.25, 0.95),
    ]


def make_cascade_specs(base):
    return [
        ParamSpec("kp_pos_far_xy", "kp_pos_far", (0, 1), 0.45, 1.45),
        ParamSpec("kp_pos_near_xy", "kp_pos_near", (0, 1), 0.18, 0.75),
        ParamSpec("ki_pos_xy", "ki_pos", (0, 1), 0.00, 0.12),
        ParamSpec("kp_vel_xy", "kp_vel", (0, 1), 0.05, 0.65),
        ParamSpec("ki_vel_xy", "ki_vel", (0, 1), 0.00, 0.12),
        ParamSpec("vel_ff_xy", "vel_ff", (0, 1), 0.35, 1.15),
        ParamSpec("schedule_xy", "schedule_radius", (0, 1), 0.20, 0.75),
        ParamSpec("v_ref_near_xy", "v_ref_cap_near", (0, 1), 0.16, 0.55),
        ParamSpec("cmd_near_xy", "cmd_cap_near", (0, 1), 0.16, 0.60),
        ParamSpec("max_rate_xy", "max_rate_xyz", (0, 1), 1.2, 5.0),
        ParamSpec("kp_pos_near_z", "kp_pos_near", (2,), 0.18, 0.75),
        ParamSpec("kp_vel_z", "kp_vel", (2,), 0.05, 0.70),
        ParamSpec("vel_ff_z", "vel_ff", (2,), 0.35, 1.15),
        ParamSpec("v_ref_near_z", "v_ref_cap_near", (2,), 0.12, 0.45),
        ParamSpec("cmd_near_z", "cmd_cap_near", (2,), 0.12, 0.45),
        ParamSpec("kp_pos_near_yaw", "kp_pos_near", (3,), 0.20, 1.10),
        ParamSpec("kp_vel_yaw", "kp_vel", (3,), 0.04, 0.35),
        ParamSpec("vel_ff_yaw", "vel_ff", (3,), 0.35, 1.15),
        ParamSpec("yaw_cmd_cap_near", "yaw_cmd_cap_near", None, 0.25, 0.90),
    ]

# 注释：生成控制器参数规范：根据控制器名称和基础参数，生成对应的参数规范。
def make_specs(controller_name, base):
    if controller_name == "plan_a":
        return make_plan_a_specs(base)
    if controller_name == "cascade":
        return make_cascade_specs(base)
    raise ValueError(controller_name)

# 注释：编码默认参数：将基础参数转换为单位向量，便于后续优化算法处理。
def encode_defaults(base, specs):
    values = []
    for spec in specs:
        raw = get_param(base, spec)
        values.append((raw - spec.low) / (spec.high - spec.low))
    return np.clip(np.array(values, dtype=float), 0.0, 1.0)

# 注释：解码参数：将单位向量转换为实际参数值，恢复原始参数范围。
def decode_params(base, specs, unit_vector):
    params = deepcopy(base)
    for spec, unit_value in zip(specs, unit_vector):
        value = spec.low + float(unit_value) * (spec.high - spec.low)
        set_param(params, spec, value)
    return params

# 注释：修补控制器参数：将优化算法找到的候选参数应用到控制器模块中。
def patch_controller_params(module, candidate_params):
    def _candidate_make_params(wind_enabled):
        return deepcopy(candidate_params)

    module._make_params = _candidate_make_params


def resolve_trial_log_path(args):
    if args.trial_log:
        return ROOT / args.trial_log
    wind_tag = "wind" if args.wind else "nowind"
    return ROOT / f"auto_tune_trials_{args.controller}_{wind_tag}.csv"

# 注释：初始化试验日志：如果日志文件不存在，则创建新的日志文件。
def init_trial_log(path, specs):
    if path.exists() and path.stat().st_size > 0:
        return
    fieldnames = [
        "generation",
        "candidate",
        "controller",
        "wind",
        "cost",
        "is_generation_best",
        "is_global_best",
    ] + [spec.name for spec in specs]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# 注释：追加试验日志：将当前试验结果追加到日志文件中。
def append_trial_log(path, args, specs, params, generation, candidate, cost, is_generation_best, is_global_best):
    row = {
        "generation": generation,
        "candidate": candidate,
        "controller": args.controller,
        "wind": int(args.wind),
        "cost": f"{cost:.10f}",
        "is_generation_best": int(is_generation_best),
        "is_global_best": int(is_global_best),
    }
    for spec in specs:
        row[spec.name] = f"{get_param(params, spec):.10f}"
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

# 注释：无头评估器：在模拟环境中评估控制器性能，不进行实时绘图和日志记录。
class HeadlessEvaluator:
    def __init__(self, controller_name, targets, wind_enabled, segment_time, settle_delay, eval_window):
        self.controller_name = controller_name
        self.targets = targets
        self.wind_enabled = wind_enabled
        self.segment_time = segment_time
        self.settle_delay = settle_delay
        self.eval_window = eval_window

        self.timestep = 1.0 / 1000.0
        self.control_dt = 1.0 / 50.0
        self.steps_between_control = int(round(self.control_dt / self.timestep))

        self.mass = 0.088
        self.arm = 0.06
        self.kf = 0.566e-5
        self.km = 0.762e-7
        self.k_trans = np.array([3.365e-2, 3.365e-2, 3.365e-2])
        self.tm = 0.0163

    def compute_dynamics(self, rpm_values, lin_vel_world, quat):
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        omega = rpm_values * (2.0 * np.pi / 60.0) # 注释：将 RPM 转换为弧度/秒。
        omega_squared = omega**2
        motor_forces = omega_squared * self.kf
        thrust = np.array([0.0, 0.0, np.sum(motor_forces)])
        vel_body = np.dot(rotation.T, lin_vel_world)
        drag_body = -self.k_trans * vel_body
        force = drag_body + thrust

        z_torques = omega_squared * self.km
        z_torque = -z_torques[0] - z_torques[1] + z_torques[2] + z_torques[3]
        x_torque = (-motor_forces[0] + motor_forces[1] + motor_forces[2] - motor_forces[3]) * self.arm
        y_torque = (-motor_forces[0] + motor_forces[1] - motor_forces[2] + motor_forces[3]) * self.arm
        return force, np.array([x_torque, y_torque, z_torque])
# 注释：电机模型：根据期望 RPM 和当前 RPM，计算实际 RPM。
    def motor_model(self, desired_rpm, current_rpm):
        rpm_derivative = (desired_rpm - current_rpm) / self.tm
        return current_rpm + rpm_derivative * self.timestep

# 注释：检查动作：检查控制器输出是否为合法的四元数动作，并返回合法动作。
    def check_action(self, action):
        if not isinstance(action, (tuple, list)) or len(action) not in (4, 5):
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(np.clip(action[0], -1.0, 1.0)),
            float(np.clip(action[1], -1.0, 1.0)),
            float(np.clip(action[2], -1.0, 1.0)),
            float(np.clip(action[3], -1.74533, 1.74533)),
        )
# 注释：评估控制器性能：在模拟环境中评估控制器性能，返回总成本。
    def evaluate(self, candidate_params):
        global p
        import pybullet as pybullet_module
        import pybullet_data
        from src.tello_controller import TelloController
        from src.wind import Wind

        p = pybullet_module
        module = import_controller_module(self.controller_name)
        patch_controller_params(module, candidate_params)

        cid = p.connect(p.DIRECT)
        records = []
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0.0, 0.0, -9.81)
            p.loadURDF("plane.urdf")
            drone_id = p.loadURDF(str(ROOT / "resources" / "tello.urdf"), [0, 0, 1], p.getQuaternionFromEuler([0, 0, 0]))

            tello_controller = TelloController(9.81, self.mass, self.arm, 0.35, self.kf, self.km)
            wind_sim = Wind(max_steady_state=0.02, max_gust=0.02, k_gusts=0.1)
            prev_rpm = np.array([0.0, 0.0, 0.0, 0.0])
            desired_vel = np.array([0.0, 0.0, 0.0])
            yaw_rate_setpoint = 0.0
            previous_cmd = None
            segment_control_counts = [0 for _ in self.targets]
            loop_counter = 0

            total_time = self.segment_time * len(self.targets)
            total_steps = int(total_time / self.timestep)

            for step in range(total_steps):
                sim_time = step * self.timestep
                target_idx = min(int(sim_time // self.segment_time), len(self.targets) - 1)
                target = self.targets[target_idx]

                pos, quat = p.getBasePositionAndOrientation(drone_id)
                lin_vel_world, ang_vel_world = p.getBaseVelocity(drone_id)
                roll, pitch, yaw = p.getEulerFromQuaternion(quat)

                yaw_quat = p.getQuaternionFromEuler([0, 0, yaw])
                _, inverted_quat = p.invertTransform([0, 0, 0], quat)
                _, inverted_quat_yaw = p.invertTransform([0, 0, 0], yaw_quat)
                lin_vel = np.array(p.rotateVector(inverted_quat_yaw, lin_vel_world))
                ang_vel = np.array(p.rotateVector(inverted_quat, ang_vel_world))

                if any(not math.isfinite(v) for v in pos) or pos[2] < -0.2 or max(abs(v) for v in pos) > 20.0:
                    return 1e6

                loop_counter += 1
                if loop_counter >= self.steps_between_control:
                    loop_counter = 0
                    local_time = segment_control_counts[target_idx] * self.control_dt
                    segment_control_counts[target_idx] += 1
                    state = np.array([pos[0], pos[1], pos[2], roll, pitch, yaw])
                    action = self.check_action(module.controller(state, target, self.control_dt, self.wind_enabled))
                    desired_vel = np.array(action[:3])
                    yaw_rate_setpoint = action[3]

                    pos_err = math.sqrt(
                        (target[0] - pos[0]) ** 2
                        + (target[1] - pos[1]) ** 2
                        + (target[2] - pos[2]) ** 2
                    )
                    yaw_err = abs(wrap_pi(target[3] - yaw))
                    cmd = np.array([action[0], action[1], action[2], action[3]])
                    if previous_cmd is None:
                        cmd_delta = 0.0
                        switch_count = 0
                    else:
                        cmd_delta = float(np.sum(np.abs(cmd - previous_cmd)))
                        switch_count = int(np.sum(np.sign(cmd[:3]) != np.sign(previous_cmd[:3])))
                    previous_cmd = cmd
                    records.append(
                        {
                            "segment": target_idx,
                            "local_time": local_time,
                            "pos_err": pos_err,
                            "yaw_err": yaw_err,
                            "effort": float(np.sum(cmd * cmd)),
                            "cmd_delta": cmd_delta,
                            "switch_count": switch_count,
                        }
                    )

                rpm = tello_controller.compute_control(
                    desired_vel,
                    lin_vel,
                    quat,
                    ang_vel,
                    yaw_rate_setpoint,
                    self.timestep,
                )
                rpm = self.motor_model(rpm, prev_rpm)
                prev_rpm = rpm

                force, torque = self.compute_dynamics(rpm, lin_vel_world, quat)
                p.applyExternalForce(drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)
                p.applyExternalTorque(drone_id, -1, torque, p.LINK_FRAME)

                if self.wind_enabled:
                    wind = wind_sim.get_wind(self.timestep)
                    p.applyExternalForce(drone_id, -1, wind, pos, p.WORLD_FRAME)

                for joint_index in range(4):
                    rad_s = rpm[joint_index] * (2.0 * np.pi / 60.0)
                    current_angle = p.getJointState(drone_id, joint_index)[0]
                    p.resetJointState(drone_id, joint_index, current_angle + rad_s * self.timestep)

                p.stepSimulation()
        finally:
            p.disconnect(cid)

        return score_records(records, len(self.targets), self.segment_time, self.settle_delay, self.eval_window)


def mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def score_records(records, target_count, segment_time, settle_delay, eval_window):
    if not records:
        return 1e6

    total_cost = 0.0
    valid_segments = 0
    for segment in range(target_count):
        seg_records = [r for r in records if r["segment"] == segment]
        if not seg_records:
            total_cost += 1e5
            continue

        start_t = settle_delay
        end_t = settle_delay + eval_window
        window = [r for r in seg_records if start_t <= r["local_time"] <= end_t]
        if len(window) < 5:
            fallback_start = max(0.0, segment_time - eval_window)
            window = [r for r in seg_records if r["local_time"] >= fallback_start]
        if len(window) < 5:
            total_cost += 1e5
            continue

        pos_mean, pos_std = mean_std([r["pos_err"] for r in window])
        yaw_mean, yaw_std = mean_std([r["yaw_err"] for r in window])
        effort_mean, _ = mean_std([r["effort"] for r in window])
        smooth_mean, _ = mean_std([r["cmd_delta"] for r in window])
        switch_mean, _ = mean_std([r["switch_count"] for r in window])
        overshoot = max(r["pos_err"] for r in window)

        segment_cost = (
            120.0 * pos_mean
            + 80.0 * pos_std
            + 20.0 * yaw_mean
            + 10.0 * yaw_std
            + 8.0 * overshoot
            + 0.40 * effort_mean
            + 0.60 * smooth_mean
            + 0.20 * switch_mean
        )
        total_cost += segment_cost
        valid_segments += 1

    if valid_segments == 0:
        return 1e6
    return total_cost / valid_segments

# 注释：运行 CEM 算法：初始化随机种子，加载目标点，获取基础参数和规范，设置初始均值和标准差，初始化试验日志，创建评估器，运行 CEM 算法。
def run_cem(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    targets = load_targets(ROOT / "targets.csv")
    base = get_base_params(args.controller, args.wind)
    specs = make_specs(args.controller, base)
    mean = encode_defaults(base, specs)
    std = np.full_like(mean, args.initial_std)
    trial_log_path = resolve_trial_log_path(args)
    init_trial_log(trial_log_path, specs)
# 注释：创建评估器：创建无头评估器，用于在模拟环境中评估控制器性能。
    evaluator = HeadlessEvaluator(
        controller_name=args.controller,
        targets=targets,
        wind_enabled=args.wind,
        segment_time=args.segment_time,
        settle_delay=args.settle_delay,
        eval_window=args.eval_window,
    )

    best_cost = float("inf")
    best_vector = mean.copy()
    history = []
# 注释：运行 CEM 算法：初始化随机种子，加载目标点，获取基础参数和规范，设置初始均值和标准差，初始化试验日志，创建评估器，运行 CEM 算法。
    for generation in range(args.generations):
        candidates = []
        if generation == 0:
            candidates.append(mean.copy())
        while len(candidates) < args.population:
            candidates.append(np.clip(np.random.normal(mean, std), 0.0, 1.0))

        scored = []
        for i, vector in enumerate(candidates):
            params = decode_params(base, specs, vector)
            cost = evaluator.evaluate(params)
            is_global_best = cost < best_cost
            if is_global_best:
                best_cost = cost
                best_vector = vector.copy()
            scored.append((cost, vector, params, i + 1, is_global_best))
            print(
                f"generation={generation + 1:02d} "
                f"candidate={i + 1:02d}/{len(candidates):02d} "
                f"cost={cost:.6f}",
                flush=True,
            )

        scored.sort(key=lambda item: item[0])
        elite = scored[: max(2, args.elite)]
        elite_vectors = np.array([item[1] for item in elite])
        mean = np.mean(elite_vectors, axis=0)
        std = np.maximum(np.std(elite_vectors, axis=0), args.min_std)

        generation_best_candidate = scored[0][3]
        for cost, _vector, params, candidate_no, is_global_best in scored:
            append_trial_log(
                trial_log_path,
                args,
                specs,
                params,
                generation + 1,
                candidate_no,
                cost,
                candidate_no == generation_best_candidate,
                is_global_best,
            )

        history.append({"generation": generation + 1, "best_cost": scored[0][0], "global_best_cost": best_cost})
        print(
            f"GENERATION {generation + 1} DONE: "
            f"best={scored[0][0]:.6f}, global_best={best_cost:.6f}",
            flush=True,
        )

        save_result(args, base, specs, best_vector, best_cost, history)

    return best_cost


def save_result(args, base, specs, best_vector, best_cost, history):
    params = decode_params(base, specs, best_vector)
    output = {
        "controller": args.controller,
        "wind": args.wind,
        "best_cost": best_cost,
        "method": "Cross-Entropy Method (CEM)",
        "tuned_parameters": {
            spec.name: get_param(params, spec)
            for spec in specs
        },
        "params": params,
        "history": history,
    }
    output_path = ROOT / args.output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"saved best result to {output_path}", flush=True)

# 注释：解析命令行参数：解析命令行参数，返回参数对象。
def parse_args():
    parser = argparse.ArgumentParser(
        description="Headless auto tuner for Plan A controller.py or Plan B controller_cascade.py"
    )
    parser.add_argument("--controller", choices=("plan_a", "cascade"), default="cascade")
    parser.add_argument("--wind", action="store_true", help="tune with wind disturbance enabled")
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--initial-std", type=float, default=0.18)
    parser.add_argument("--min-std", type=float, default=0.03)
    parser.add_argument("--segment-time", type=float, default=8.0)
    parser.add_argument("--settle-delay", type=float, default=3.0)
    parser.add_argument("--eval-window", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", default="auto_tune_best.json")
    parser.add_argument(
        "--trial-log",
        default=None,
        help="CSV file for every candidate's parameters and cost; default auto_tune_trials_<controller>_<wind>.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(ROOT)
    run_cem(parse_args())
