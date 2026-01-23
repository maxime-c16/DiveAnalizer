Automation for headless acceptance test

Prerequisites
- Python 3.8+
- `playwright` Python package

Install:

```bash
python -m pip install --user playwright
python -m playwright install chromium
```

Usage

Run the shell wrapper (one-liner) which starts the CLI, waits for the server, then runs the headless browser automation:

```bash
bash scripts/run_accept_and_close.sh /path/to/input_video.MOV
```

The automation will attempt to click the "Accept & Close" button and wait for the server to shut down.

Logs and output
- Console output is printed to stdout. The Playwright script will print any console errors it observes.
