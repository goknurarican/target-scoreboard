#!/usr/bin/env python3
import os
import sys

# Test all imports
try:
    import streamlit as st
    print("✅ Streamlit OK")
except ImportError as e:
    print(f"❌ Streamlit: {e}")

try:
    import requests
    print("✅ Requests OK")
except ImportError as e:
    print(f"❌ Requests: {e}")

try:
    import pandas as pd
    print("✅ Pandas OK")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import plotly
    print("✅ Plotly OK")
except ImportError as e:
    print(f"❌ Plotly: {e}")

print("\nPython executable:", sys.executable)
print("Working directory:", os.getcwd())
