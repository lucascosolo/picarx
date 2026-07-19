#!/usr/bin/env python3
# layer_b/modules/steering_controller.py
"""
Ackermann-aware smooth steering controller (pure math, no I/O).

Replaces the discrete "counter-steer by N degrees when an object looms"
law with a continuous local-arc planner: every tick it turns the fresh
world snapshot into ONE floating-point steering angle plus a recommended
speed, filtered and rate-limited so consecutive commands describe a
fluid arc instead of a staircase. It owns NO hardware and publishes
NOTHING - field_agent feeds it snapshots and publishes its output as
ordinary vetoable picarx/intent/move intents, so the safety daemon's
authority is untouched.

Pipeline per compute_command() call:

  1. Perception: each fresh tracked object becomes (bearing, distance):
       - lateral offset = center_offset / (frame_width/2), same
         convention as field_agent (positive = right of center);
       - distance estimated from its bounding-box area (see AREA
         CALIBRATION below), fused with the ultrasonic when that has a
         fresh reading - the ultrasonic is authoritative at short range
         for anything near the center of frame.
  2. Goal point: objects in the path contribute a signed lateral shift
     away from their side - closer, more central, and approaching
     objects push harder; opposite sides SUM AND CANCEL, so two objects
     flanking a gap yield a near-zero shift and the robot threads
     between them. The shift, placed at a lookahead distance scaled
     with the nearest threat, is the pure-pursuit goal point.
  3. Pure pursuit: alpha = atan2(lateral_shift, lookahead);
     curvature kappa = 2*sin(alpha)/lookahead;
     steering_angle = atan(wheelbase * kappa)  - true Ackermann geometry,
     clamped to the configured steering limit.
  4. Smoothing: an exponential filter plus a hard rate limiter
     (steering_rate_deg_per_sec * dt per call) guarantee the commanded
     angle never jumps; a small deadband suppresses sub-degree jitter.
  5. Speed policy: cruise speed scaled DOWN with commanded curvature
     (tight arc = slow) and with proximity (near obstacle = slow),
     floored at min_speed_factor * cruise.

CALIBRATION KNOBS (config.json, all overridable per-instance in tests):

  kinematics.wheelbase_mm            physical axle-to-axle distance.
                                     Default 95 (PiCar-X chassis);
                                     measure yours - it directly scales
                                     angle-for-curvature.
  kinematics.max_steer_deg           servo's physical steering limit.
  kinematics.steering_rate_deg_per_sec
                                     max commanded slew. Lower = smoother
                                     arcs, higher = snappier response.
  steering.area_distance_k           the AREA CALIBRATION constant:
                                     distance_cm ~= k / sqrt(area_ratio).
                                     A pinhole camera makes an object's
                                     apparent width ~ 1/distance, so area
                                     ~ 1/distance^2. k folds real object
                                     size + focal length into one number,
                                     so it is only ROUGHLY right across
                                     object classes - calibrate by
                                     placing a typical obstacle at a
                                     known distance and reading
                                     area_ratio from the vision topic
                                     (k = distance_cm * sqrt(area)).
                                     The ultrasonic fusion above makes
                                     short-range behaviour insensitive
                                     to this constant.
  steering.clearance_m               preferred lateral clearance when
                                     passing an object (the minimum goal
                                     shift once anything qualifies).
  steering.cruise_speed              straight-path speed (matches the
                                     explore cruise value).
  steering.curve_slowdown_gain       how aggressively speed drops with
                                     steering angle (0 = never slow,
                                     1 = full-lock means min speed).

Everything else (cone width, minimum area, urgency multiplier) mirrors
the constants of the discrete law it replaces, so behaviour degrades
gracefully to familiar territory if the tuning is off.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import robot_config


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


class SteeringController:
    def __init__(self, config=None):
        cfg = config if config is not None else robot_config
        # --- kinematics (physical robot properties) ---
        self.wheelbase_m = float(cfg.get("kinematics", "wheelbase_mm", 95)) / 1000.0
        self.max_steer_deg = float(cfg.get("kinematics", "max_steer_deg", 30))
        self.steering_rate = float(cfg.get(
            "kinematics", "steering_rate_deg_per_sec", 60))
        # --- avoidance tuning ---
        self.avoid_max_deg = min(self.max_steer_deg, 22.0)  # stay under the +/-30 emergency reflexes
        self.area_distance_k = float(cfg.get("steering", "area_distance_k", 35.0))
        self.clearance_m = float(cfg.get("steering", "clearance_m", 0.15))
        self.cruise_speed = float(cfg.get("steering", "cruise_speed", 25))
        self.curve_gain = float(cfg.get("steering", "curve_slowdown_gain", 0.9))
        self.min_speed_factor = 0.25
        # Perception gates - mirror the discrete law's semantics.
        self.min_area = 0.06            # ignore specks unless approaching
        self.cone_frac = 0.75           # |offset| beyond this isn't in the path
        self.approach_boost = 1.5       # urgency multiplier for approaching objects
        # Goal-point shaping.
        self.shift_gain_m = 0.55        # extra lateral shift per unit of threat weight
        self.max_shift_m = 0.40
        self.lookahead_scale = 0.6      # Ld = scale * nearest distance, clamped:
        self.lookahead_min_m = 0.25
        self.lookahead_max_m = 0.80
        # Distance model bounds + fusion.
        self.dist_min_cm, self.dist_max_cm = 10.0, 400.0
        self.prox_near_cm, self.prox_far_cm = 20.0, 150.0   # threat-weight ramp
        self.ultra_trust_cm = 60.0      # fresh ultrasonic below this is authoritative
        self.ultra_center_frac = 0.5    # ...for objects this close to center
        # Speed-vs-proximity ramp.
        self.slow_floor_cm, self.slow_start_cm = 15.0, 80.0
        # Smoothing.
        self.ema_alpha = 0.5
        self.deadband_deg = 1.5
        self.dt_min, self.dt_max = 0.02, 0.5
        # Filter state.
        self._angle = 0.0               # current filtered command (deg)
        self._last_ts = None

    # ---------- perception ----------

    def _estimate_distance_cm(self, obj, ultra_cm):
        """Rough distance from bounding-box area (see AREA CALIBRATION in
        the module docstring), overridden by a fresh short ultrasonic
        reading for near-center objects - at short range the sonar is
        measuring the thing the camera is looking at."""
        area = max(1e-4, float(obj.get("area_ratio") or 0.0))
        d = _clamp(self.area_distance_k / math.sqrt(area),
                   self.dist_min_cm, self.dist_max_cm)
        frame_w = obj.get("frame_width") or 0
        if ultra_cm is not None and ultra_cm < self.ultra_trust_cm and frame_w > 0:
            offset_frac = obj.get("center_offset", 0) / (frame_w / 2.0)
            if abs(offset_frac) < self.ultra_center_frac:
                d = min(d, ultra_cm)
        return d

    def _relevant_objects(self, snapshot):
        """(objects, ultra_cm): each object annotated with offset_frac,
        dist_cm and threat weight; None-safe and staleness-safe."""
        objects = (snapshot or {}).get("objects") or {}
        ultra_cm = None
        if (snapshot and snapshot.get("distance_cm") is not None
                and not snapshot.get("distance_stale", True)
                and snapshot["distance_cm"] > 0):
            ultra_cm = float(snapshot["distance_cm"])
        if objects.get("stale", True):
            return [], ultra_cm
        out = []
        for obj in objects.get("items", []):
            frame_w = obj.get("frame_width") or 0
            if frame_w <= 0:
                continue
            area = obj.get("area_ratio") or 0.0
            approaching = bool(obj.get("approaching"))
            if area < self.min_area and not approaching:
                continue
            offset_frac = obj.get("center_offset", 0) / (frame_w / 2.0)
            if abs(offset_frac) > self.cone_frac:
                continue
            dist_cm = self._estimate_distance_cm(obj, ultra_cm)
            centrality = 1.0 - abs(offset_frac)
            proximity = _clamp(
                (self.prox_far_cm - dist_cm) / (self.prox_far_cm - self.prox_near_cm),
                0.0, 1.0)
            weight = centrality * proximity
            if approaching:
                # Closing fast = urgent whatever its current apparent size.
                weight = max(weight * self.approach_boost, 0.5)
            if weight <= 0.0:
                continue
            out.append({"label": obj.get("label", "something"),
                        "offset_frac": offset_frac, "dist_cm": dist_cm,
                        "weight": weight, "approaching": approaching})
        return out, ultra_cm

    # ---------- pure pursuit ----------

    def _raw_target_deg(self, objs, nearest_cm):
        """Unfiltered pure-pursuit steering angle for this snapshot."""
        shift = 0.0
        for o in objs:
            if o["offset_frac"] == 0:
                continue  # dead ahead has no side to prefer
            side = 1.0 if o["offset_frac"] > 0 else -1.0
            shift += -side * (self.clearance_m + self.shift_gain_m * o["weight"])
        shift = _clamp(shift, -self.max_shift_m, self.max_shift_m)
        if shift == 0.0:
            return 0.0
        lookahead = _clamp((nearest_cm / 100.0) * self.lookahead_scale,
                           self.lookahead_min_m, self.lookahead_max_m)
        alpha = math.atan2(shift, lookahead)
        kappa = 2.0 * math.sin(alpha) / lookahead
        angle = math.degrees(math.atan(self.wheelbase_m * kappa))
        return _clamp(angle, -self.avoid_max_deg, self.avoid_max_deg)

    def _slew(self, target_deg, dt):
        """Exponential smoothing + hard rate limit toward target_deg.
        Guarantees |change per call| <= steering_rate * dt."""
        step = (target_deg - self._angle) * self.ema_alpha
        max_step = self.steering_rate * dt
        self._angle += _clamp(step, -max_step, max_step)
        return self._angle

    @staticmethod
    def _reason(objs, angle):
        left = sorted({o["label"] for o in objs if o["offset_frac"] < 0})
        right = sorted({o["label"] for o in objs if o["offset_frac"] > 0})
        if left and right and abs(angle) < 4.0:
            return (f"threading the gap between {', '.join(left)} (left) "
                    f"and {', '.join(right)} (right)")
        labels = ", ".join(sorted({o["label"] for o in objs}))
        if abs(angle) < 1e-6:
            return f"slowing for {labels} dead ahead"
        return f"bending {'left' if angle < 0 else 'right'} around {labels}"

    # ---------- public API ----------

    def compute_command(self, snapshot, now=None):
        """One control tick. Returns
        {"steering_angle_deg": float, "speed": float, "reason": str,
         "active": bool, "labels": [...], "nearest_cm": float|None}.
        active=False means nothing qualifies for avoidance (the caller
        should run its normal cruise/wander logic); the angle still
        decays smoothly back toward straight in that state so a
        re-activation never jumps from a stale value."""
        now = now if now is not None else time.time()
        dt = self.dt_max if self._last_ts is None else _clamp(
            now - self._last_ts, self.dt_min, self.dt_max)
        self._last_ts = now

        objs, ultra_cm = self._relevant_objects(snapshot)
        candidates = [o["dist_cm"] for o in objs]
        if ultra_cm is not None:
            candidates.append(ultra_cm)
        nearest_cm = min(candidates) if candidates else None

        if not objs:
            self._slew(0.0, dt)
            angle = 0.0 if abs(self._angle) < self.deadband_deg else self._angle
            return {"steering_angle_deg": float(angle), "speed": float(self.cruise_speed),
                    "reason": "clear path", "active": False, "labels": [],
                    "nearest_cm": nearest_cm}

        raw = self._raw_target_deg(objs, nearest_cm)
        filtered = self._slew(raw, dt)
        angle = 0.0 if abs(filtered) < self.deadband_deg else filtered

        curve_factor = max(self.min_speed_factor,
                           1.0 - self.curve_gain * abs(filtered) / self.max_steer_deg)
        speed = self.cruise_speed * curve_factor
        if nearest_cm is not None:
            prox_factor = _clamp(
                (nearest_cm - self.slow_floor_cm) / (self.slow_start_cm - self.slow_floor_cm),
                self.min_speed_factor, 1.0)
            speed = min(speed, self.cruise_speed * prox_factor)

        return {"steering_angle_deg": float(angle), "speed": float(round(speed, 2)),
                "reason": self._reason(objs, angle), "active": True,
                "labels": sorted({o["label"] for o in objs}),
                "nearest_cm": nearest_cm}
