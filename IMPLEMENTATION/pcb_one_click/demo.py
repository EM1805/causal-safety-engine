"""PCB demo runner.

Run with:
  python demo.py

(or double-click demo.bat on Windows).

This runs the full pipeline and writes outputs to ./out.
"""

import os
import subprocess
import sys


def main() -> int:
    # Ensure relative paths work even when launched from an IDE
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    print("\n=== PCB Demo ===")
    print("Folder:", root)

    cli = os.path.join(root, "pcb_cli.py")
    data = os.path.join(root, "data.csv")

    # Skip Level 3.2 by default because it requires logged trials.
    cmd = [sys.executable, cli, "run", "--data", data, "--skip-32"]
    print("Running:", " ".join(cmd))

    subprocess.check_call(cmd)

    print("\nDone. Outputs are in ./out\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
