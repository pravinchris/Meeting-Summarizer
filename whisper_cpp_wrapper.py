import os
import subprocess

def run_whisper_cpp(audio_path, model_path="./models/ggml-small.en.bin"):
    output_path = "output.txt"
    command = [
        "./main", "--model", model_path,
        "--file", audio_path,
        "--output-txt",
        "--output-file", output_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return f.read()
    return ""
