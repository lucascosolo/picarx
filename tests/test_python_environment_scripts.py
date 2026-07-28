import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PythonEnvironmentScriptTest(unittest.TestCase):
    def _read(self, name):
        with open(os.path.join(ROOT, name), encoding="utf-8") as stream:
            return stream.read()

    def test_repair_uses_sunfounder_robot_hat_and_checks_api(self):
        script = self._read("repair_python_environment.sh")
        self.assertIn(
            "git+https://github.com/sunfounder/robot-hat.git@2.5.x", script)
        self.assertNotIn('"robot_hat|robot-hat"', script)
        self.assertIn(
            "from robot_hat import ADC, Pin, PWM, Servo, fileDB, Grayscale_Module, Ultrasonic",
            script)

    def test_legacy_setup_uses_same_robot_hat_source(self):
        script = self._read("setup_python.sh")
        self.assertIn(
            "git+https://github.com/sunfounder/robot-hat.git@2.5.x", script)
        self.assertNotIn('"robot_hat|robot-hat"', script)

    def test_arm_setup_selects_pi_mediapipe_runtime(self):
        for name in ("repair_python_environment.sh", "setup_python.sh"):
            script = self._read(name)
            self.assertIn('PICARX_MEDIAPIPE_PACKAGE', script)
            self.assertIn('mediapipe-rpi4', script)


if __name__ == "__main__":
    unittest.main()
