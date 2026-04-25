import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = os.path.join(BASE_DIR, "PalletSiteWithlaser.py")

subprocess.run([sys.executable, "-m", "streamlit", "run", app])