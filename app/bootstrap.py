import sys
from pathlib import Path

# Path to /app
APP_DIR = Path(__file__).resolve().parent

# Path to project root (parent of /app)
PROJECT_ROOT = APP_DIR.parent

# Add project root to Python path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
