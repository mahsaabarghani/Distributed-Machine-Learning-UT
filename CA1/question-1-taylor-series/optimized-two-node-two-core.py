from mpi4py import MPI
import time
import math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


n = 5000

chunk_size = n // size
remain = n % size

if rank < remain:
    start = rank * (chunk_size + 1)
    end = start + chunk_size
else:
    start = rank * chunk_size + remain
    end = start + chunk_size - 1

def taylor(start, end):
    sumation = 0
    # if k ==0 :
    a_firstValue = 1
    c_firstValue = 2
    d_firstValue = 1
    
    for k in range(start, end + 1):
        if k>0:
            a_firstValue *= (2 * k) * ((2 * k) + 1)
            c_firstValue *= 8
            d_firstValue  *= k
        sumation += a_firstValue / (c_firstValue * (d_firstValue**2))

    return sumation

startTime = time.time()
sub_result = taylor(start, end)
total_result = comm.reduce(sub_result, op=MPI.SUM, root=0)
endTime = time.time()

if rank == 0:
    print(f"Rank {rank}: Total result with n={n}: {total_result}")
    print('\n')
    print(f"Rank {rank}: Total time: {endTime - startTime} seconds")


print('\n')
print(f"Rank: {rank} time: {endTime - startTime} seconds")
