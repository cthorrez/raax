import timeit
import numpy as np

def time_function(func, label, runs=5):
    times = np.zeros(shape=(runs,))
    result = None
    
    for idx in range(runs):
        start = timeit.default_timer()
        result = func()  # Capture the return value of the function
        end = timeit.default_timer()
        times[idx] = end - start
      
    # drop the first run to exclude compile time of jitted implementation
    if runs > 1:
        times = times[1:]
    avg_time = np.mean(times)
    std_dev = np.std(times)
    print(f"{label} average time: {avg_time:.5f} (s) ± {std_dev:.5f} (s)")
    
    return result  # Return the result of the function