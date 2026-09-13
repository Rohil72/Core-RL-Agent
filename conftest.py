import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

collect_ignore = ["test_output.txt", "test_result.log"]
