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
data_agg_legs = pd.read_csv("dados_agregados_pernas.csv",\
                index_col=[0,1,2,3,4,5,6])  

sample_groups = data_agg_legs.groupby(level=[0,1,2])

data_samples = [sample_groups.get_group(g) for g in sample_groups.groups]
sample_classes = [data_samples[g].index.get_level_values(6)[0] for g in range(0,len(data_samples))]



# %%
x = np.arange(0,100)
y = 5 + np.sin(0.1*x) + 0.1*np.cos(2*x)

plt.plot(x,y)
plt.show()

# %%
f, pxx = periodogram(y, fs=1, nfft=10)
plt.plot(f, pxx)
plt.show()