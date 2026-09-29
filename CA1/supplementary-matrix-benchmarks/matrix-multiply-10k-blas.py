import numpy as np
import time

def benchmark_matrix_multiplication(size):
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    start_time = time.time()
    C = np.dot(A, B)
    end_time = time.time()

    print(f"Matrix multiplication of size {size}x{size} with BLAS took {end_time - start_time:.2f} seconds.")

benchmark_matrix_multiplication(10000)