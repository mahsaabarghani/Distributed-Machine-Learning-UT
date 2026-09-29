import time
import numpy as np


myMatrix1 = np.random.rand(20000 , 20000)
myMatrix2 = np.random.rand(20000 , 20000)


startTime = time.time()
multiple = np.dot(myMatrix1, myMatrix2)
endTime = time.time()

print(f'total time = {endTime - startTime} ')