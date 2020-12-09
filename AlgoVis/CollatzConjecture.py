'''
Algorithm Visualisation for rthe Collatz Conjecture
'''

# Imports
import matplotlib.pyplot as plt

# Main Functions
# Algorithm Functions
def CollatzConjecture_Converge(startVal, max_iters=-1):
    values = [startVal]
    curVal = startVal
    n_iters = 0
    while(curVal > 1):
        if curVal % 2 == 0:
            curVal = int(round(curVal / 2))
        else:
            curVal = 3*curVal + 1

        values.append(curVal)

        n_iters += 1
        if n_iters > max_iters and max_iters > -1:
            break

    return values

# Visualisation Functions
def CollatzConjecture_Visualise(values):
    print("No of iters:", len(values)-1)
    values_str = []
    for v in values:
        values_str.append(str(v))
    print("Trace:", ' '.join(values_str))
    plt.plot(list(range(1, len(values)+1)), values)
    plt.scatter(list(range(1, len(values)+1)), values)
    plt.show()

# Driver Code
# Params
startValue = 10
# Params

# RunCode
trace = CollatzConjecture_Converge(startValue)
CollatzConjecture_Visualise(trace)