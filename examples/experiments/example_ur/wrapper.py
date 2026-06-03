from ur_env.envs.ur_env import UREnv


class URExampleEnv(UREnv):
    """Reach-only example task on UR.

    Reset uses the UR server's joint reset target instead of Cartesian
    RESET_POSE. The joint reset path is implemented with RTDE moveJ, matching
    the hardware path that has already been validated.
    """
    pass

    # def go_to_reset(self, joint_reset=False):
    #     if hasattr(self, "velocity_stop"):
    #         self.velocity_stop()

    #     response = self.session.post(self.url + "jointreset")
    #     response.raise_for_status()

    #     self._update_currpos()
    #     self.nextpos = self.currpos.copy()
    #     print(f"[example_ur] Joint reset complete: {response.text}")
