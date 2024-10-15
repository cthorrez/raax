import timeit
import numpy as np

def time_function(func, label, runs=5):
    times = []
    result = None
    
    for _ in range(runs):
        start = timeit.default_timer()
        result = func()  # Capture the return value of the function
        end = timeit.default_timer()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_dev = np.std(times)
    print(f"{label} average time: {avg_time:.5f} (s) ± {std_dev:.5f} (s)")
    
    return result  # Return the result of the function