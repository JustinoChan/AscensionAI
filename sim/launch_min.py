import subprocess, sys, os
p = subprocess.Popen([sys.executable, "min_child.py"], stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, cwd=os.getcwd())
out, err = p.communicate(timeout=60)
print("rc", p.returncode)
print("STDERR:\n" + err.decode()[:2000])
