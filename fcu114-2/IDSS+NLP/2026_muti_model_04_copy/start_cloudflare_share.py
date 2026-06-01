"""Start Streamlit and Cloudflare Tunnel for temporary sharing.

Run:
    python start_cloudflare_share.py

The public URL is written to cloudflare_share_url.txt.
Keep this process running while teammates use the site.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
URL_FILE = ROOT / "cloudflare_share_url.txt"
STREAMLIT_LOG = ROOT / "streamlit_share.log"
CLOUDFLARED_LOG = ROOT / "cloudflared_share.log"


def wait_localhost(timeout: int = 90) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen("http://127.0.0.1:8501", timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def main() -> int:
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        winget_packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        matches = list(winget_packages.glob("Cloudflare.cloudflared_*/*cloudflared.exe"))
        if matches:
            cloudflared = str(matches[0])
    if not cloudflared:
        print("cloudflared not found. Install it with: winget install Cloudflare.cloudflared")
        return 1

    URL_FILE.write_text("", encoding="utf-8")
    streamlit_log = STREAMLIT_LOG.open("w", encoding="utf-8")
    tunnel_log = CLOUDFLARED_LOG.open("w", encoding="utf-8")

    streamlit = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ],
        cwd=ROOT,
        stdout=streamlit_log,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if not wait_localhost():
        print("Streamlit did not become ready. See streamlit_share.log.")
        streamlit.terminate()
        return 1

    tunnel = subprocess.Popen(
        [cloudflared, "tunnel", "--url", "http://127.0.0.1:8501"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")
    public_url = ""

    try:
        assert tunnel.stdout is not None
        for line in tunnel.stdout:
            tunnel_log.write(line)
            tunnel_log.flush()
            match = pattern.search(line)
            if match and not public_url:
                public_url = match.group(0)
                URL_FILE.write_text(public_url, encoding="utf-8")
                print(f"Public URL: {public_url}", flush=True)

            if streamlit.poll() is not None:
                print("Streamlit stopped. See streamlit_share.log.")
                return 1
            if tunnel.poll() is not None:
                print("Cloudflared stopped. See cloudflared_share.log.")
                return 1
    except KeyboardInterrupt:
        print("Stopping sharing server...")
    finally:
        for proc in (tunnel, streamlit):
            if proc.poll() is None:
                proc.terminate()
        streamlit_log.close()
        tunnel_log.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
