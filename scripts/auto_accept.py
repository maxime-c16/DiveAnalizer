#!/usr/bin/env python3
"""
Automate a Chromium visit to the review gallery and click the "Accept & Close" button.

Usage:
  python scripts/auto_accept.py http://localhost:8765/dives/review_gallery.html

Requires: `playwright` (install and run `python -m playwright install chromium` once).
"""
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765/dives/review_gallery.html"

def server_is_up(u):
    try:
        urllib.request.urlopen(u, timeout=1)
        return True
    except Exception:
        return False

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page()

        errors = []

        def on_console(msg):
            t = msg.type
            text = msg.text()
            line = f"[console:{t}] {text}"
            if t == "error" or "SyntaxError" in text:
                errors.append(line)
            print(line)

        page.on("console", on_console)
        page.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))

        page.goto(url, wait_until="networkidle", timeout=30000)

        # Try a few selector strategies to find the Accept & Close control
        selectors = [
            "text=Accept & Close",
            "text=Accept",
            "button:has-text(\"Accept & Close\")",
            "button:has-text(\"Accept\")",
            ".accept-close",
            "#accept-close",
        ]

        clicked = False
        for sel in selectors:
            try:
                page.wait_for_selector(sel, timeout=2500)
                page.click(sel)
                clicked = True
                print(f"Clicked selector: {sel}")
                break
            except Exception:
                continue

        if not clicked:
            # Fallback: click first button containing 'accept' text (case-insensitive)
            try:
                page.evaluate(
                    "() => { const b = Array.from(document.querySelectorAll('button')).find(x=>/accept/i.test(x.innerText)); if(b) b.click(); }"
                )
                print("Clicked fallback accept button via evaluate()")
            except Exception:
                print("Could not locate Accept button")

        # Allow client to send delete/shutdown
        time.sleep(2)

        # Wait for the server to go down (shutdown endpoint should stop it)
        for _ in range(60):
            if not server_is_up(url):
                print("Server appears to have shut down")
                break
            time.sleep(0.5)

        browser.close()

        if errors:
            print("Captured errors:\n" + "\n".join(errors))
            sys.exit(2)

        print("Automation finished successfully")

if __name__ == '__main__':
    main()
