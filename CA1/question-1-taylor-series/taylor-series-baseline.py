import time
import math

def taylor(n = 5000):
    sumation = 0
    for k in range(n):
        a = math.factorial((2*k)+1)
        c = 2 ** ((3*k)+1)
        d = (math.factorial(k)) **2
        b = c * d
        sumation += a / b
    
    return sumation

startTime = time.time()
callTaylor = taylor()
endTime = time.time()

print( f'result = {callTaylor}' )
print( f'total time = {endTime - startTime}')
