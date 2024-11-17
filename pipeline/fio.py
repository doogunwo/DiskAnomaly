import subprocess
import os
import random
import time

def select_fio(folder: str) -> str:
    """
    Randomly select a .fio script from the specified folder.
    """
    try:
        fio_files = [f for f in os.listdir(folder) if f.endswith(".fio")]
        if not fio_files:
            raise FileNotFoundError("No .fio files found in the specified folder.")
        selected_script = random.choice(fio_files)
        return os.path.join(folder, selected_script)
    except Exception as e:
        print(f"Error selecting Fio script: {e}")
        return None
    
def run_fio(jobfile: str):
    """

    """
    try:
        print(f"Running Fio job: {jobfile}")

        proc = subprocess.Popen(
            ["fio", "--output-format=json", jobfile],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    except Exception as e:
        print(f"Error running Fio: {e}")