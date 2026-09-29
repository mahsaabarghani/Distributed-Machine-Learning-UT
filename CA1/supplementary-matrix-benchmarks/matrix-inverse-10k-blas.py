import numpy as np
import time

A = np.random.rand(10000, 10000)

start_time = time.time()
A_inv = np.linalg.inv(A)
end_time = time.time()

print(f"Matrix inversion of size 20000x20000 with BLAS took {end_time - start_time:.2f} seconds.")