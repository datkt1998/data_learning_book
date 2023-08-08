# BAI TOAN TIM SO NGUYEN TO
# su dung numba
from numba import njit
import math
import timeit

def runtime(func):
    def func_wrapper(x):
        start = timeit.default_timer()
        res = func(x)
        stop = timeit.default_timer()
        print("Run with numba: ",stop - start, "(seconds)")
        return res
    return func_wrapper

@njit(fastmath=True, cache=True)
def is_prime(number):
    if number == 2:
        return True
    if number <= 1 or not number % 2:
        return False
    max_range = int(math.sqrt(number)) + 1
    for div in range(3, max_range, 2):
        if not number % div:
            return False
    return True

@runtime
@njit(fastmath=True, cache=True)
def run_program(max_number):
    for number in range(max_number):
        is_prime(number)
        
if __name__ == '__main__':
    run_program(5000000)