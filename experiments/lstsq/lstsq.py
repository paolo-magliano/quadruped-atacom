import torch
import matplotlib.pyplot as plt
import time
import tqdm
import os

torch.manual_seed(0)

path = os.path.dirname(os.path.realpath(__file__))

# Load matrixes
# A shape (batch, n, m) B shape (batch, n) -> x shape (batch, m)
# Where batch is 4096, n is 24, m is 36
A = torch.load(os.path.join(path, "A.pt"))
B = torch.load(os.path.join(path, "B.pt"))

times = []

# Solution of the problem:
solution = torch.linalg.lstsq(A, B).solution

# Test all the batch sizes until max_batch_size (do not set over 4096)
max_batch_size = 512

# Measure the time for every batch size, from 1 to max_batch_size
for i in tqdm.tqdm(range(1, max_batch_size)):
    start = time.time()
    x = torch.linalg.lstsq(A[:i], B[:i]).solution
    assert torch.allclose(x, solution[:i])
    times.append(time.time() - start)

# Plot the results
plt.figure()
plt.plot(times, label='lstsq')
plt.legend()
plt.xlabel('Batch size')
plt.ylabel('Time (s)')

plt.savefig('plot/lstsq/lstsq.png')
