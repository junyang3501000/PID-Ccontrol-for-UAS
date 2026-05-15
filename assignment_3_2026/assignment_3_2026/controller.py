import math
import multiprocessing
import queue
from collections import deque
from datetime import datetime

# ---------------------------------------------------------------------------
# 实时误差绘图（与 controller_normal.py 相同：子进程 matplotlib，避免阻塞仿真）
# ---------------------------------------------------------------------------
_ERROR_PLOT_MAXLEN = 600
_ERROR_TIME_WINDOW = 22.0
_ERROR_PLOT_QUEUE_MAX = 128

_ctx = multiprocessing.get_context("spawn")
_ERROR_PLOT_QUEUE = None
_ERROR_PLOT_PROC = None


def _build_timestamped_log_name(base_name):
    dot_idx = base_name.rfind(".")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dot_idx == -1:
        return f"{base_name}_{stamp}"
    return f"{base_name[:dot_idx]}_{stamp}{base_name[dot_idx:]}"


_FLIGHT_LOG_FILE = _build_timestamped_log_name("flight_log.csv")


def _get_error_plot_queue():
    global _ERROR_PLOT_QUEUE
    if _ERROR_PLOT_QUEUE is None:
        _ERROR_PLOT_QUEUE = _ctx.Queue(maxsize=_ERROR_PLOT_QUEUE_MAX)
    return _ERROR_PLOT_QUEUE


def _error_plot_worker(rx_queue):
    try:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
    except Exception:
        while True:
            msg = rx_queue.get()
            if msg is None:
                break
        return

    from queue import Empty

    times = deque(maxlen=_ERROR_PLOT_MAXLEN)
    series = tuple(deque(maxlen=_ERROR_PLOT_MAXLEN) for _ in range(4))
    plt.ion()
    fig, axes = plt.subplots(4, 1, num="Position errors (live)", figsize=(7, 6), sharex=True)
    labels = ("e_x body (m)", "e_y body (m)", "e_z (m)", "e_yaw (rad)")
    lines = []

    for ax, lab in zip(axes, labels):
        (line,) = ax.plot([], [], lw=1.2)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
        lines.append(line)

    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.show()

    def apply_sample(msg):
        t, errs = msg
        times.append(float(t))
        for i in range(4):
            series[i].append(float(errs[i]))

    def redraw():
        if len(times) < 2:
            return

        xs = list(times)
        t_hi = xs[-1]
        t_lo = max(xs[0], t_hi - _ERROR_TIME_WINDOW)

        for i in range(4):
            ys = list(series[i])
            lines[i].set_data(xs, ys)
            y_max = max((abs(v) for v in ys), default=0.0)
            y_max = max(y_max, 1e-3)
            pad = 0.08 * y_max + 1e-4
            axes[i].set_ylim(-y_max - pad, y_max + pad)

        axes[-1].set_xlim(t_lo, t_hi)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    while True:
        try:
            msg = rx_queue.get(timeout=0.25)
        except Empty:
            plt.pause(0.02)
            continue

        if msg is None:
            plt.close("all")
            return

        apply_sample(msg)

        while True:
            try:
                msg2 = rx_queue.get_nowait()
            except Empty:
                break

            if msg2 is None:
                plt.close("all")
                return

            apply_sample(msg2)

        redraw()
        plt.pause(0.001)


def _ensure_error_plot_process():
    global _ERROR_PLOT_PROC
    if _ERROR_PLOT_PROC is not None and _ERROR_PLOT_PROC.is_alive():
        return
    if _ERROR_PLOT_PROC is not None:
        return

    q = _get_error_plot_queue()
    p = _ctx.Process(
        target=_error_plot_worker,
        args=(q,),
        daemon=True,
        name="LiveErrorPlot",
    )
    p.start()
    _ERROR_PLOT_PROC = p


def _update_error_plot(errors, t):
    _ensure_error_plot_process()
    if _ERROR_PLOT_PROC is None or not _ERROR_PLOT_PROC.is_alive():
        return

    sample = (
        float(t),
        (
            float(errors[0]),
            float(errors[1]),
            float(errors[2]),
            float(errors[3]),
        ),
    )
    q = _get_error_plot_queue()

    try:
        q.put_nowait(sample)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(sample)
        except queue.Full:
            pass


cascade_state = {
    "pos_integral": [0.0, 0.0, 0.0, 0.0],
    "pos_prev_error": [0.0, 0.0, 0.0, 0.0],
    "vel_integral": [0.0, 0.0, 0.0, 0.0],
    "prev_cmd": [0.0, 0.0, 0.0, 0.0],
    "sim_time": 0.0,
    "prev_target": None,
    "prev_position": None,
    "prev_yaw": None,
    "vel_est_body": [0.0, 0.0, 0.0],
    "yaw_rate_est": 0.0,
}


def _send_error_plot(errors):
    try:
        _update_error_plot(errors, cascade_state["sim_time"])
    except Exception:
        pass


def _clamp(value, lower, upper):
    return max(min(value, upper), lower)


def _wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

# 注释：平滑限幅函数：误差小时用 near_value，误差大时逐渐过渡到 far_value。
def _smooth_limit(value, limit):
    if limit <= 0.0:
        return 0.0
    return limit * math.tanh(value / limit)

# 注释：增益调度函数：误差小时用 near_value，误差大时逐渐过渡到 far_value。
def _scheduled_gain(abs_err, near_value, far_value, schedule_radius):
    if schedule_radius <= 1e-9:
        return far_value
    blend = _clamp(abs_err / schedule_radius, 0.0, 1.0)
    return near_value + (far_value - near_value) * blend


def _parse_state_and_target(state, target_pos):
    current = {
        "x": state[0],
        "y": state[1],
        "z": state[2],
        "roll": state[3],
        "pitch": state[4],
        "yaw": state[5],
    }
    target = {
        "x": target_pos[0],
        "y": target_pos[1],
        "z": target_pos[2],
        "yaw": target_pos[3],
    }
    return current, target


def _compute_errors(current, target):
    err_x_global = target["x"] - current["x"]
    err_y_global = target["y"] - current["y"]
    err_z_global = target["z"] - current["z"]

    cos_yaw = math.cos(current["yaw"])
    sin_yaw = math.sin(current["yaw"])

    err_x_body = err_x_global * cos_yaw + err_y_global * sin_yaw
    err_y_body = -err_x_global * sin_yaw + err_y_global * cos_yaw
    err_yaw = _wrap_pi(target["yaw"] - current["yaw"])
    return [err_x_body, err_y_body, err_z_global, err_yaw]


def _reset_if_target_changed(current, target):
    target_tuple = (target["x"], target["y"], target["z"], target["yaw"])
    target_changed = (
        cascade_state["prev_target"] is None
        or any(abs(target_tuple[i] - cascade_state["prev_target"][i]) > 1e-6 for i in range(4))
    )

    if target_changed:
        cascade_state["pos_integral"] = [0.0, 0.0, 0.0, 0.0]
        cascade_state["pos_prev_error"] = [0.0, 0.0, 0.0, 0.0]
        cascade_state["vel_integral"] = [0.0, 0.0, 0.0, 0.0]
        cascade_state["prev_cmd"] = [0.0, 0.0, 0.0, 0.0]
        cascade_state["prev_position"] = (current["x"], current["y"], current["z"])
        cascade_state["prev_yaw"] = current["yaw"]
        cascade_state["vel_est_body"] = [0.0, 0.0, 0.0]
        cascade_state["yaw_rate_est"] = 0.0

    cascade_state["prev_target"] = target_tuple

# 注释：根据当前和上一周期的位置、姿态、时间间隔，计算无人机在控制坐标系下的速度和角速度（用于外环状态反馈）。
def _estimate_body_velocity(current, dt): 

    prev_position = cascade_state["prev_position"]
    prev_yaw = cascade_state["prev_yaw"]
    if prev_position is None or prev_yaw is None or dt <= 1e-9:
        cascade_state["prev_position"] = (current["x"], current["y"], current["z"])
        cascade_state["prev_yaw"] = current["yaw"]
        cascade_state["vel_est_body"] = [0.0, 0.0, 0.0]
        cascade_state["yaw_rate_est"] = 0.0
        return [0.0, 0.0, 0.0, 0.0]

    vx_world = (current["x"] - prev_position[0]) / dt
    vy_world = (current["y"] - prev_position[1]) / dt
    vz_world = (current["z"] - prev_position[2]) / dt

    cos_yaw = math.cos(current["yaw"])
    sin_yaw = math.sin(current["yaw"])
    vx_body = vx_world * cos_yaw + vy_world * sin_yaw
    vy_body = -vx_world * sin_yaw + vy_world * cos_yaw
    yaw_rate = _wrap_pi(current["yaw"] - prev_yaw) / dt
    # 注释：使用一阶低通滤波器对速度和角速度进行滤波，降低由“差分估计”带来的瞬时波动和测量噪声。
    vel_alpha = 0.35 
    yaw_alpha = 0.30 
    cascade_state["vel_est_body"][0] = vel_alpha * vx_body + (1.0 - vel_alpha) * cascade_state["vel_est_body"][0]
    cascade_state["vel_est_body"][1] = vel_alpha * vy_body + (1.0 - vel_alpha) * cascade_state["vel_est_body"][1]
    cascade_state["vel_est_body"][2] = vel_alpha * vz_world + (1.0 - vel_alpha) * cascade_state["vel_est_body"][2]
    cascade_state["yaw_rate_est"] = yaw_alpha * yaw_rate + (1.0 - yaw_alpha) * cascade_state["yaw_rate_est"]

    cascade_state["prev_position"] = (current["x"], current["y"], current["z"])
    cascade_state["prev_yaw"] = current["yaw"]
    return [
        cascade_state["vel_est_body"][0],
        cascade_state["vel_est_body"][1],
        cascade_state["vel_est_body"][2],
        cascade_state["yaw_rate_est"],
    ]


def _make_params(wind_enabled):
    if wind_enabled:
        # 有风：来自 cascade_wind_best.json 的 params（wind=true，CEM 最优）。
        return {
            "kp_pos_far": [
                1.1120152994811134,
                1.1120152994811134,
                1.15,
                1.15,
            ],
            "kp_pos_near": [
                0.5112147214209553,
                0.5112147214209553,
                0.4628152180204461,
                0.7321278810171701,
            ],
            "ki_pos": [
                0.04671039399091943,
                0.04671039399091943,
                0.06,
                0.02,
            ],
            "kd_pos": [0.0, 0.0, 0.0, 0.0],
            "kp_vel": [
                0.41609970461963186,
                0.41609970461963186,
                0.3915383507371567,
                0.14068305427043298,
            ],
            "ki_vel": [
                0.06323982946263534,
                0.06323982946263534,
                0.05,
                0.01,
            ],
            "vel_ff": [
                0.7545114981830598,
                0.7545114981830598,
                0.7642958329901213,
                0.8455658678917601,
            ],
            "schedule_radius": [
                0.3503098740752093,
                0.3503098740752093,
                0.35,
                0.55,
            ],
            "pos_int_window": [0.55, 0.55, 0.45, 0.35], 
            "pos_int_clamp": [0.35, 0.35, 0.3, 0.25],
            "vel_int_window": [0.45, 0.45, 0.35, 0.3],
            "vel_int_clamp": [0.25, 0.25, 0.22, 0.2],
            "v_ref_cap_far": [0.95, 0.95, 0.55],
            "v_ref_cap_near": [
                0.3912177796704297,
                0.3912177796704297,
                0.1715563599443168,
            ],
            "yaw_ref_cap_far": 0.95,
            "yaw_ref_cap_near": 0.65,
            "cmd_cap_far": [1.0, 1.0, 0.55],
            "cmd_cap_near": [
                0.4400550655608795,
                0.4400550655608795,
                0.1697971603735292,
            ],
            "yaw_cmd_cap_far": 0.95,
            "yaw_cmd_cap_near": 0.557181387265293,
            "max_rate_xyz": [
                4.643498354778622,
                4.643498354778622,
                2.3,
            ],
            "max_rate_yaw": 2.0,
            "hold_err": [0.0018, 0.0018, 0.0018, 0.004],
            "hold_vel": [0.007, 0.007, 0.007, 0.1],
        }

    # 无风：来自 cascade_best.json 的 params（wind=false，CEM 最优）。
    return {
        "kp_pos_far": [
            0.8706689563071079,
            0.8706689563071079,
            0.88,
            1.0,
        ],
        "kp_pos_near": [
            0.75,
            0.75,
            0.7426017812651495,
            0.6305484056210559,
        ],
        "ki_pos": [
            0.07978634455037534,
            0.07978634455037534,
            0.05,
            0.02,
        ],
        "kd_pos": [0.0, 0.0, 0.0, 0.0],
        "kp_vel": [
            0.1784799757411381,
            0.1784799757411381,
            0.10737774805977768,
            0.08266327888507721,
        ],
        "ki_vel": [
            0.05584382906642322,
            0.05584382906642322,
            0.04,
            0.01,
        ],
        "vel_ff": [
            0.9338918030127047,
            0.9338918030127047,
            1.15,
            0.705493435312827,
        ],
        "schedule_radius": [
            0.39176606352526283,
            0.39176606352526283,
            0.3,
            0.45,
        ],
        "pos_int_window": [0.55, 0.55, 0.45, 0.32],
        "pos_int_clamp": [0.32, 0.32, 0.28, 0.24],
        "vel_int_window": [0.42, 0.42, 0.34, 0.28],
        "vel_int_clamp": [0.22, 0.22, 0.2, 0.18],
        "v_ref_cap_far": [0.95, 0.95, 0.5],
        "v_ref_cap_near": [
            0.4104351429740747,
            0.4104351429740747,
            0.32633528631102027,
        ],
        "yaw_ref_cap_far": 0.85,
        "yaw_ref_cap_near": 0.55,
        "cmd_cap_far": [1.0, 1.0, 0.5],
        "cmd_cap_near": [
            0.3670666319548293,
            0.3670666319548293,
            0.12843875444923322,
        ],
        "yaw_cmd_cap_far": 0.85,
        "yaw_cmd_cap_near": 0.6746617068922786,
        "max_rate_xyz": [
            3.798092418213858,
            3.798092418213858,
            2.2,
        ],
        "max_rate_yaw": 1.8,
        "hold_err": [0.0018, 0.0018, 0.0018, 0.003],
        "hold_vel": [0.007, 0.007, 0.007, 0.01],
    }


def _limit_for_axis(axis, abs_err, near_key, far_key, yaw_near_key, yaw_far_key, params):
    if axis < 3:
        return _scheduled_gain(
            abs_err,
            params[near_key][axis],
            params[far_key][axis],
            params["schedule_radius"][axis],
        )
    return _scheduled_gain(
        abs_err,
        params[yaw_near_key],
        params[yaw_far_key],
        params["schedule_radius"][axis],
    )


def _update_integral(int_key, axis, err, rate, dt, window_key, clamp_key, params):
    if abs(err) < params[window_key][axis] and abs(rate) < 0.45:
        cascade_state[int_key][axis] += err * dt
        cascade_state[int_key][axis] = _clamp(
            cascade_state[int_key][axis],
            -params[clamp_key][axis],
            params[clamp_key][axis],
        )
    else:
        cascade_state[int_key][axis] *= 0.90


def _position_loop(axis, err, rate, dt, params):
    abs_err = abs(err)
    _update_integral(
        "pos_integral",
        axis,
        err,
        rate,
        dt,
        "pos_int_window",
        "pos_int_clamp",
        params,
    )

    kp_eff = _scheduled_gain(
        abs_err,
        params["kp_pos_near"][axis],
        params["kp_pos_far"][axis],
        params["schedule_radius"][axis],
    )
    derr = (err - cascade_state["pos_prev_error"][axis]) / dt
    cascade_state["pos_prev_error"][axis] = err

    v_ref = (
        kp_eff * err
        + params["ki_pos"][axis] * cascade_state["pos_integral"][axis]
        + params["kd_pos"][axis] * derr
    )
    cap = _limit_for_axis(
        axis,
        abs_err,
        "v_ref_cap_near",
        "v_ref_cap_far",
        "yaw_ref_cap_near",
        "yaw_ref_cap_far",
        params,
    )
    return _smooth_limit(v_ref, cap)


def _velocity_loop(axis, vel_ref, measured_rate, pos_err, dt, params):
    vel_err = vel_ref - measured_rate
    _update_integral(
        "vel_integral",
        axis,
        vel_err,
        measured_rate,
        dt,
        "vel_int_window",
        "vel_int_clamp",
        params,
    )

    cmd = (
        params["vel_ff"][axis] * vel_ref
        + params["kp_vel"][axis] * vel_err
        + params["ki_vel"][axis] * cascade_state["vel_integral"][axis]
    )
    cap = _limit_for_axis(
        axis,
        abs(pos_err),
        "cmd_cap_near",
        "cmd_cap_far",
        "yaw_cmd_cap_near",
        "yaw_cmd_cap_far",
        params,
    )
    return _smooth_limit(cmd, cap)


def _compute_cascade_commands(errors, rates, dt, params):
    commands = []
    for axis in range(4):
        vel_ref = _position_loop(axis, errors[axis], rates[axis], dt, params)
        cmd = _velocity_loop(axis, vel_ref, rates[axis], errors[axis], dt, params)
        if abs(errors[axis]) < params["hold_err"][axis] and abs(rates[axis]) < params["hold_vel"][axis]:
            cmd = 0.0
            cascade_state["pos_integral"][axis] *= 0.9
            cascade_state["vel_integral"][axis] *= 0.9
        commands.append(cmd)
    return commands


def _apply_slew_rate_limit(commands, dt, params):
    limited = [0.0, 0.0, 0.0, 0.0]
    for axis in range(3):
        max_delta = params["max_rate_xyz"][axis] * dt
        delta = commands[axis] - cascade_state["prev_cmd"][axis]
        limited[axis] = cascade_state["prev_cmd"][axis] + _clamp(delta, -max_delta, max_delta)

    max_delta_yaw = params["max_rate_yaw"] * dt
    delta_yaw = commands[3] - cascade_state["prev_cmd"][3]
    limited[3] = cascade_state["prev_cmd"][3] + _clamp(delta_yaw, -max_delta_yaw, max_delta_yaw)
    cascade_state["prev_cmd"] = list(limited)
    return limited


def _write_flight_log(current, target, err_yaw, final_output):
    """把当前控制周期写入带时间戳的 CSV，便于 plotter.py 分析。"""
    try:
        with open(_FLIGHT_LOG_FILE, "a") as f:
            if f.tell() == 0:
                f.write(
                    "Time,Target_X,Target_Y,Target_Z,Target_Yaw,"
                    "Pos_X,Pos_Y,Pos_Z,Roll,Pitch,Yaw,Err_Yaw,"
                    "Out_Vx,Out_Vy,Out_Vz,Out_YawRate\n"
                )
            f.write(
                f"{cascade_state['sim_time']:.4f},"
                f"{target['x']:.4f},{target['y']:.4f},{target['z']:.4f},"
                f"{_wrap_pi(target['yaw']):.4f},"
                f"{current['x']:.4f},{current['y']:.4f},{current['z']:.4f},"
                f"{current['roll']:.4f},{current['pitch']:.4f},{current['yaw']:.4f},"
                f"{err_yaw:.4f},"
                f"{final_output[0]:.4f},{final_output[1]:.4f},"
                f"{final_output[2]:.4f},{final_output[3]:.4f}\n"
            )
    except Exception:
        pass


def controller(state, target_pos, dt, wind_enabled=False):
    """
    Plan B: controller.py 接口兼容的级联 PID 外环。

    位置环生成速度参考，轻量速度环再根据估计速度修正输出。
    最终仍返回 (vx_body, vy_body, vz_world, yaw_rate)，可直接复制为 controller.py 测试。
    """
    dt = max(dt, 1e-3)
    cascade_state["sim_time"] += dt

    current, target = _parse_state_and_target(state, target_pos)
    errors = _compute_errors(current, target)
    _send_error_plot(errors)
    _reset_if_target_changed(current, target)
    rates = _estimate_body_velocity(current, dt)
    params = _make_params(wind_enabled)

    raw_commands = _compute_cascade_commands(errors, rates, dt, params)
    output_cmds = _apply_slew_rate_limit(raw_commands, dt, params)
    final_output = (output_cmds[0], output_cmds[1], output_cmds[2], output_cmds[3])
    _write_flight_log(current, target, errors[3], final_output)
    return final_output
