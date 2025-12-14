from cog import BasePredictor, Input, Path
import subprocess
import os

class Predictor(BasePredictor):

    def setup(self):
        os.makedirs("outputs", exist_ok=True)

    def predict(
        self,
        prompt: str = Input(description="Description de la scène"),
        frames: int = Input(default=16),
    ) -> Path:

        output_video = "outputs/result.mp4"

        subprocess.run([
            "python",
            "run.py",
            "--prompt", prompt,
            "--frames", str(frames),
            "--output", output_video
        ], check=True)

        return Path(output_video)
