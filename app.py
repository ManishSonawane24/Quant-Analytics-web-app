import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None


def start_backend():
    """
    Start the FastAPI backend using uvicorn in a subprocess.

    We run it on port 8000 so the Streamlit frontend can talk to it via HTTP.
    """
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.routes:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ]
    )


def start_frontend():
    """
    Start the Streamlit dashboard in a subprocess.

    It will open a browser window on port 8501 by default.
    """
    # Disable Streamlit telemetry/analytics to prevent the Segment.com error
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    env["STREAMLIT_SERVER_HEADLESS"] = "false"
    
    frontend_path = Path(__file__).parent / "frontend" / "dashboard.py"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(frontend_path),
            "--server.port",
            "8501",
            "--server.headless",
            "false",
        ],
        env=env,
    )


def wait_for_backend(max_wait=30):
    """Wait for backend to be ready by checking the health endpoint."""
    if requests is None:
        # Fallback to simple sleep if requests not available
        time.sleep(5)
        return True
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get("http://localhost:8000/health", timeout=2)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    print("Starting FastAPI backend...")
    backend_proc = start_backend()
    
    # Wait for backend to be ready before launching the UI
    print("Waiting for backend to be ready...")
    if wait_for_backend():
        print("✓ Backend is ready!")
    else:
        print("⚠ Warning: Backend may not be fully ready, but continuing anyway...")
    
    print("Starting Streamlit frontend...")
    frontend_proc = start_frontend()

    try:
        # Wait for either process to exit; if one exits, terminate the other.
        while True:
            backend_ret = backend_proc.poll()
            frontend_ret = frontend_proc.poll()
            if backend_ret is not None or frontend_ret is not None:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (backend_proc, frontend_proc):
            if proc and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()


