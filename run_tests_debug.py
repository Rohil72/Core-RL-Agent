import pytest
import sys

if __name__ == "__main__":
    with open("test_result.log", "w", encoding="utf-8") as f:
        # Redirect stdout/stderr
        sys.stdout = f
        sys.stderr = f
        sys.exit(pytest.main(["-v", "tests/"]))
