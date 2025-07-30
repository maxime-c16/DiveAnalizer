#!/usr/bin/env python3
import os
import sys
import subprocess

# Set environment variables to suppress logs
env = os.environ.copy()
env.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'GLOG_minloglevel': '3',
    'TF_ENABLE_ONEDNN_OPTS': '0'
})

# Run the actual script with stderr filtered
if __name__ == "__main__":
    # Run the actual slAIcer.py script
    cmd = [sys.executable, "slAIcer_core.py"] + sys.argv[1:]

    # Filter out the specific log messages we don't want
    process = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    # Real-time output filtering
    while True:
        output = process.stderr.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            # Filter out the unwanted log messages
            if not any(phrase in output for phrase in [
                'GL version:', 'INFO: Created TensorFlow', 'WARNING: All log messages before absl'
            ]):
                sys.stderr.write(output)
                sys.stderr.flush()

    # Get stdout
    stdout, _ = process.communicate()
    if stdout:
        sys.stdout.write(stdout)
        sys.stdout.flush()

    sys.exit(process.returncode)
