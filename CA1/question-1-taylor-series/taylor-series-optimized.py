import time

startTime = time.time()
def taylor(n = 5000):
    sumation = 0
    # if k ==0 :
    a_firstValue = 1
    c_firstValue = 2
    d_firstValue = 1
    
    for k in range(n):
        if k>0:
            a_firstValue *= (2 * k) * ((2 * k) + 1)
            c_firstValue *= 8
            d_firstValue  *= k
        sumation += a_firstValue / (c_firstValue * (d_firstValue**2))

    return sumation

res = taylor()
endTime = time.time()
print( f' result  = {res}')
print(f'total time = {endTime - startTime}')
    