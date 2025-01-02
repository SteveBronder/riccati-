# %%
import time
import subprocess

N = 1000
l = (100,)
eps = 1e-12
epsh = 1e-13
m = 32
bash_cmd = f"for i in {{1..1000}} do ./brenner237 {100} {eps} {epsh} {m} done"
start = time.time_ns()
for i in range(N):
    res = subprocess.run(bash_cmd, capture_output=True, shell=True)

end = time.time_ns()
runtime = (end - start) * 1e-9 / N

print(runtime, "s")
