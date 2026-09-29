import time
import numpy as np


myMatrix1 = np.random.rand(10000 , 10000)
myMatrix2 = np.random.rand(10000 , 10000)


startTime = time.time()
multiple = np.dot(myMatrix1, myMatrix2)
endTime = time.time()

print(f'total time = {endTime - startTime} ')