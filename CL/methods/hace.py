from __future__ import annotations

from typing import Dict, Tuple

from CL.rl.sac import SAC


class HACE_SAC(SAC):
    """Homeostatic Actor-Critic (HACE-style) reward shaping.

    Adds an internal reward based on *drive reduction* toward a health setpoint:

        r_int = scale * (|h_prev - h*| - |h_now - h*|)

    so the agent is rewarded for reducing deviation from the setpoint.
    """

    def __init__(
        self,
        hace_health_setpoint: float = 100.0,
        hace_internal_reward_scale: float = 0.01,
        **vanilla_sac_kwargs,
    ):
        super().__init__(**vanilla_sac_kwargs)
        self.hace_health_setpoint = float(hace_health_setpoint)
        self.hace_internal_reward_scale = float(hace_internal_reward_scale)
        self._prev_health = None

    def on_env_reset(self, info: Dict) -> None:
        self._prev_health = self._read_health()

    def shape_reward(self, reward: float, info: Dict) -> Tuple[float, float]:
        health_now = self._read_health()
        if health_now is None:
            return reward, 0.0

        if self._prev_health is None:
            self._prev_health = health_now
            return reward, 0.0

        drive_prev = abs(float(self._prev_health) - self.hace_health_setpoint)
        drive_now = abs(float(health_now) - self.hace_health_setpoint)

        internal_reward = self.hace_internal_reward_scale * (drive_prev - drive_now)
        self._prev_health = health_now

        return reward + internal_reward, internal_reward

    def _read_health(self):
        # VizDoom is an optional dependency in some installs; keep this robust.
        try:
            from vizdoom import GameVariable
        except Exception:
            return None

        try:
            active_env = self.env.get_active_env()
            game = getattr(active_env, "game", None)
            if game is None:
                return None
            return float(game.get_game_variable(GameVariable.HEALTH))
        except Exception:
            return None

