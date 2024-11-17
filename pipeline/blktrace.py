import subprocess

def run_blktrace(device: str):
    """
    Run blktrace and blkparse on the specified device and yield parsed lines.
    """
    try:
        # Start blktrace and pipe to blkparse
        blktrace_proc = subprocess.Popen(
            ["sudo", "blktrace", "-d", device, "-o", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        blkparse_proc = subprocess.Popen(
            ["sudo", "blkparse", "-"],
            stdin=blktrace_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # Enable text mode for easier parsing
        )

        # Process blkparse output line by line
        for line in blkparse_proc.stdout:
            yield line.strip()

        # Ensure processes are terminated
        blkparse_proc.wait()
        blktrace_proc.terminate()

    except Exception as e:
        print(f"Error running blktrace: {e}")

