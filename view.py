#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view.py has been unified into app.py!

All functionality (flipbook player, novel_images browser, video maker, and aspect ratio routing)
is now available directly in app.py.

Running view.py launches the unified app.py server on port 7865.
"""

if __name__ == "__main__":
    print("\n[*] view.py is now merged into app.py.")
    print("[*] Starting unified Media Studio on http://127.0.0.1:7865 ...\n")
    from app import app
    app.run(host="0.0.0.0", port=7865, debug=False, threaded=True)
