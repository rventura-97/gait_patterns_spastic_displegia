# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import os
from itertools import compress
from collections import Counter
from matplotlib import pyplot as plt
from scipy.signal import periodogram


# %%




# %%
x = np.arange(0,100)
y = 5 + np.sin(0.1*x) + 0.1*np.cos(2*x)

plt.plot(x,y)
plt.show()

# %%
f, pxx = periodogram(y, fs=1, nfft=10)
plt.plot(f, pxx)
plt.show()