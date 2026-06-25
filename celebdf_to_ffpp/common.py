from pathlib import Path
import sys


WORKFLOW_DIR = Path(__file__).resolve().parent
REPO_ROOT = WORKFLOW_DIR.parent
DEFAULT_CONFIG_PATH = WORKFLOW_DIR / "config.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
