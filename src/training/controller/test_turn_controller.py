import training.simrunner as sr


def _description() -> dict:
    return {
        "description": "Check how to turn a robot 180 deg",
    }


class TestTurnController(sr.Controller):
    count = 0

    def take_step(self, sensor: sr.CombiSensor) -> sr.DiffDriveValues:
        if self.count < 80:
            self.count += 1
            return sr.DiffDriveValues(-1, 1)
        return sr.DiffDriveValues(0, 0)

    def name(self) -> str:
        return "test_turn"

    def description(self) -> dict:
        return _description()
