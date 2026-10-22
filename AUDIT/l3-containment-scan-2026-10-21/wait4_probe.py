import os
import subprocess
import sys

child = "b = bytearray(b'\\x01') * (400 << 20)"
code = (
    "import subprocess, sys, time\n"
    f"subprocess.run([sys.executable, '-c', {child!r}])\n"
    "time.sleep(0.2)\n"
)
for label, snippet in (("idle root only", "import time; time.sleep(0.3)"), ("root + 400 MiB grandchild", code)):
    p = subprocess.Popen([sys.executable, "-c", snippet])
    _, status, ru = os.wait4(p.pid, 0)
    print(f"{label}: wait4 ru_maxrss {ru.ru_maxrss / (1 << 20):.0f} MiB (macOS reports bytes)")
