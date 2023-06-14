# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import os
from itertools import compress
from collections import Counter

# %%
data_class = pd.read_excel("dados_individuos.xlsx")
data_agg = [None]*len(os.listdir("Dados normativos"))*6

i = 0
for d in os.listdir("Dados normativos"):
    for f in os.listdir("Dados normativos/"+d):
        with open("Dados normativos/{}/{}".format(d,f), "r") as f:
            lines = f.readlines()
            
        data = pd.DataFrame([lines[i].split()[1:] for i in range(5,len(lines))]).transpose()

        measurements = lines[0].lstrip("\t").rstrip("\n").split("\t")
        joints = lines[1].lstrip("\t").rstrip("\n").split("\t")

        data["Joint"] = joints
        data["Measurement"] = measurements
        data["Patient"] = d
        data["File"] = f.name.split("/")[-1]
        del measurements, joints

        data_agg[i] = data
        i += 1
        print("Processed file {} / {}".format(str(i), str(len(data_agg))))


data_agg = pd.concat(data_agg)

data_agg = data_agg.merge(data_class.rename(columns={"code":"Patient"}))