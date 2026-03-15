#!/usr/bin/env bash
set -e
echo "==> Setting up Day 155 environment"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
python - << 'PYEOF'
import os
try:
    import urllib.request
    import cv2
    import numpy as np
except ImportError:
    pass
if not os.path.exists("sample.png"):
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; Day155-CV/1.0)"})
        urllib.request.urlretrieve(req, "sample.png")
        print("  Sample image downloaded: sample.png")
    except Exception as e:
        print("  Download failed (%s), creating local sample.png" % type(e).__name__)
        img = np.zeros((280, 280, 3), dtype=np.uint8)
        img[:140, :140] = [66, 135, 245]
        img[:140, 140:] = [245, 66, 66]
        img[140:, :140] = [66, 245, 66]
        img[140:, 140:] = [200, 200, 200]
        cv2.imwrite("sample.png", img)
        print("  Sample image created: sample.png")
else:
    print("  Sample image already present.")
PYEOF
echo "==> Setup complete. Activate with: source venv/bin/activate"
