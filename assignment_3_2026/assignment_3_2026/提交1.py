# wind_flag = False

import math


# Hard saturation: clamp value to [lower, upper].
def _clamp(value, lower, upper):

    return max(min(value, upper), lower)


# Wrap angle to [-pi, pi] for shortest yaw error (avoid rotating the long way).
# Subtract 2pi while above pi; add 2pi while below -pi.
def _wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


# Smooth saturation using tanh instead of a hard clip to reduce command jumps.
def _smooth_limit(value, limit):
    if limit <= 0.0:
        return 0.0
    return limit * math.tanh(value / limit)


# Gain scheduling: use near_value at small error, blend to far_value as error grows.
def _scheduled_gain(abs_err, near_value, far_value, schedule_radius):
    if schedule_radius <= 1e-9:
        return far_value
    blend = _clamp(abs_err / schedule_radius, 0.0, 1.0)
    return near_value + (far_value - near_value) * blend  # Interpolate gain


# Module-level state persisted across control cycles.
cascade_state = {
    # Position-loop integrator per axis (x, y, z, yaw).
    "pos_integral": [0.0, 0.0, 0.0, 0.0],
    # Previous position error (for derivative term).
    "pos_prev_error": [0.0, 0.0, 0.0, 0.0],
    # Velocity-loop integrator (steady wind / model mismatch).
    "vel_integral": [0.0, 0.0, 0.0, 0.0],
    # Previous command (slew-rate limiting).
    "prev_cmd": [0.0, 0.0, 0.0, 0.0],
    # Previous setpoint tuple to detect waypoint changes.
    "prev_target": None,
    # Previous position for finite-difference velocity.
    "prev_position": None,
    # Previous yaw for yaw-rate estimate.
    "prev_yaw": None,
    # Low-pass filtered velocity in yaw-aligned body frame (x,y,z).
    "vel_est_body": [0.0, 0.0, 0.0],
    # Low-pass filtered yaw rate estimate.
    "yaw_rate_est": 0.0,
}


# Parse state / target arrays into dicts for readability.
def _parse_state_and_target(state, target_pos):
    current = {
        # World position and yaw (indices match coursework interface).
        "x": state[0],
        "y": state[1],
        "z": state[2],
        "yaw": state[5],
    }
    target = {
        # Target world x, y, z, yaw.
        "x": target_pos[0],
        "y": target_pos[1],
        "z": target_pos[2],
        "yaw": target_pos[3],
    }
    return current, target


# Position errors; rotate horizontal errors into yaw-aligned body frame.
def _compute_errors(current, target):
    # World-frame position errors.
    err_x_global = target["x"] - current["x"]
    err_y_global = target["y"] - current["y"]
    err_z_global = target["z"] - current["z"]

    cos_yaw = math.cos(current["yaw"])
    sin_yaw = math.sin(current["yaw"])

    # Body-frame x,y from world errors (yaw-aligned).
    err_x_body = err_x_global * cos_yaw + err_y_global * sin_yaw
    err_y_body = -err_x_global * sin_yaw + err_y_global * cos_yaw
    # Shortest yaw error.
    err_yaw = _wrap_pi(target["yaw"] - current["yaw"])
    return [err_x_body, err_y_body, err_z_global, err_yaw]


# On setpoint change, reset integrators and history so old target does not bleed through.
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


# Finite-difference velocity in world frame, then express x,y in yaw-aligned body frame.
def _estimate_body_velocity(current, dt):
    prev_position = cascade_state["prev_position"]
    prev_yaw = cascade_state["prev_yaw"]
    if prev_position is None or prev_yaw is None or dt <= 1e-9:
        cascade_state["prev_position"] = (current["x"], current["y"], current["z"])
        cascade_state["prev_yaw"] = current["yaw"]
        cascade_state["vel_est_body"] = [0.0, 0.0, 0.0]
        cascade_state["yaw_rate_est"] = 0.0
        return [0.0, 0.0, 0.0, 0.0]

    # World-frame velocities from position difference quotient.
    vx_world = (current["x"] - prev_position[0]) / dt
    vy_world = (current["y"] - prev_position[1]) / dt
    vz_world = (current["z"] - prev_position[2]) / dt

    cos_yaw = math.cos(current["yaw"])
    sin_yaw = math.sin(current["yaw"])
    vx_body = vx_world * cos_yaw + vy_world * sin_yaw
    vy_body = -vx_world * sin_yaw + vy_world * cos_yaw
    yaw_rate = _wrap_pi(current["yaw"] - prev_yaw) / dt

    # First-order low-pass on differentiated signals (larger alpha = trust current sample more).
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


# Select full parameter set from wind_enabled (separate tuned tables for wind / no-wind).
def _make_params(wind_enabled):
    if wind_enabled:
        # Wind-on gains (CEM-tuned, cascade_wind_best.json).
        return {
            # Position P gains (far): stronger when error is large (x, y, z, yaw).
            "kp_pos_far": [
                1.1120152994811134,  # x
                1.1120152994811134,  # y
                1.15,  # z
                1.15,  # yaw
            ],
            "kp_pos_near": [
                0.5112147214209553,
                0.5112147214209553,
                0.4628152180204461,
                0.7321278810171701,
            ],
            # Position integral gain.
            "ki_pos": [
                0.04671039399091943,
                0.04671039399091943,
                0.06,
                0.02,
            ],
            # Position D term (zero here; structure kept for tuning).
            "kd_pos": [0.0, 0.0, 0.0, 0.0],
            # Velocity-loop P gain.
            "kp_vel": [
                0.41609970461963186,
                0.41609970461963186,
                0.3915383507371567,
                0.14068305427043298,
            ],
            # Velocity-loop I gain (wind / drag steady-state).
            "ki_vel": [
                0.06323982946263534,
                0.06323982946263534,
                0.05,
                0.01,
            ],
            # Feedforward from velocity reference to command.
            "vel_ff": [
                0.7545114981830598,
                0.7545114981830598,
                0.7642958329901213,
                0.8455658678917601,
            ],
            # Schedule radius: error scale where near/far gains blend.
            "schedule_radius": [
                0.3503098740752093,
                0.3503098740752093,
                0.35,
                0.55,
            ],
            # Integrate position error only inside this window.
            "pos_int_window": [0.55, 0.55, 0.45, 0.35],
            # Position integrator anti-windup clamp.
            "pos_int_clamp": [0.35, 0.35, 0.3, 0.25],
            # Velocity-error integrator window.
            "vel_int_window": [0.45, 0.45, 0.35, 0.3],
            # Velocity integrator clamp.
            "vel_int_clamp": [0.25, 0.25, 0.22, 0.2],
            # Outer-loop velocity reference cap (far).
            "v_ref_cap_far": [0.95, 0.95, 0.55],
            # Outer-loop velocity reference cap (near).
            "v_ref_cap_near": [
                0.3912177796704297,
                0.3912177796704297,
                0.1715563599443168,
            ],
            "yaw_ref_cap_far": 0.95,
            "yaw_ref_cap_near": 0.65,
            # Inner-loop command cap (far) for x,y,z.
            "cmd_cap_far": [1.0, 1.0, 0.55],
            "cmd_cap_near": [
                0.4400550655608795,
                0.4400550655608795,
                0.1697971603735292,
            ],
            "yaw_cmd_cap_far": 0.95,
            "yaw_cmd_cap_near": 0.557181387265293,
            # Max slew on xyz velocity commands (per second).
            "max_rate_xyz": [
                4.643498354778622,
                4.643498354778622,
                2.3,
            ],
            "max_rate_yaw": 2.0,
            # Hold band: position error threshold to zero command near setpoint.
            "hold_err": [0.0018, 0.0018, 0.0018, 0.004],
            # Hold band: rate threshold alongside hold_err.
            "hold_vel": [0.007, 0.007, 0.007, 0.1],
        }

    # No-wind gains (CEM-tuned, cascade_best.json).
    return {
        "kp_pos_far": [
            0.8706689563071079,  # x
            0.8706689563071079,  # y
            0.88,  # z
            1.0,  # yaw
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


# Axis-dependent cap: first three axes use list params; yaw uses scalar caps.
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


# Anti-windup integrator: accumulate only when error and rate are small; else decay.
def _update_integral(int_key, axis, err, rate, dt, window_key, clamp_key, params):
    if abs(err) < params[window_key][axis] and abs(rate) < 0.45:
        cascade_state[int_key][axis] += err * dt
        cascade_state[int_key][axis] = _clamp(cascade_state[int_key][axis], -params[clamp_key][axis], params[clamp_key][axis])
    else:
        cascade_state[int_key][axis] *= 0.90


# Outer position loop: map position error to velocity setpoint vel_ref.
def _position_loop(axis, err, rate, dt, params):
    abs_err = abs(err)
    _update_integral("pos_integral", axis, err, rate, dt, "pos_int_window", "pos_int_clamp", params)

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


# Inner velocity loop: velocity reference minus estimated rate -> command.
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
    # Inner-loop command: feedforward + P + I on velocity error.
    cmd = (
        params["vel_ff"][axis] * vel_ref
        + params["kp_vel"][axis] * vel_err
        + params["ki_vel"][axis] * cascade_state["vel_integral"][axis]
    )
    # Per-axis scheduled command cap from params; tanh soft limit on cmd.
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


# Cascade outputs for x,y,z,yaw before slew limiting.
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


# Limit per-timestep change in commands (reduce chatter).
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


# Coursework entry: velocity commands (body x,y, world z, yaw rate).
def controller(state, target_pos, dt, wind_enabled=False):
    dt = max(dt, 1e-3)

    current, target = _parse_state_and_target(state, target_pos) # Parse state and target into dictionaries for readability.
    errors = _compute_errors(current, target) # Compute position errors.
    _reset_if_target_changed(current, target) # Reset integrators and history if target changes.
    rates = _estimate_body_velocity(current, dt) # Estimate body velocity.
    params = _make_params(wind_enabled) # Select full parameter set from wind_enabled.

    raw_commands = _compute_cascade_commands(errors, rates, dt, params) # Compute cascade commands.
    output_cmds = _apply_slew_rate_limit(raw_commands, dt, params) # Apply slew rate limit.
    final_output = (output_cmds[0], output_cmds[1], output_cmds[2], output_cmds[3])
    return final_output
