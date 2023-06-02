# -*- coding: utf-8 -*-
# %%
import pandas as pd


# %%
with open("Dados normativos/CR001/Norm Joint Angles_X.txt", "r") as f:
    lines = f.readlines()
data = pd.DataFrame([lines[i].split()[1:] for i in range(5,len(lines))]).transpose()
data = data.astype("float")

measurements = lines[0].split()
measurements = [var for var in measurements if var != "Dev"]
joints = lines[1].split()

data["Joint"] = joints
data["Measurement"] = measurements
del measurements, joints

#data.set_index(["Joint","Measurement"], inplace=True)