import numpy as np
import math

def softmax(L):
    l_exps = np.exp(L)
    l_exps_sum = sum(l_exps)
    result = []
    for i in l_exps:
        result.append(1.0 * i/ l_exps_sum)
    return result

x = np.arange(6).reshape(2,3)

print(x / 256.0)

import math; print(math.e)























# print(np.float_([2,3,4]))
    # print(-np.sum([2,3,4]))
    # softmax([5,6,7])
    # print(math.exp(7))