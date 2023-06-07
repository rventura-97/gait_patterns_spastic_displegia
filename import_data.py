# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np

# %%
with open("Dados normativos/CR001/Norm Joint Moments_X.txt", "r") as f:
    lines = f.readlines()
data = pd.DataFrame([lines[i].split()[1:] for i in range(5,len(lines))]).transpose()
data = data.astype("float")

measurements = lines[0].split()
measurements = [var for var in measurements if var != "Dev"]
joints = lines[1].split()

data["Joint"] = joints
data["Measurement"] = measurements
del measurements, joints

data = data[np.logical_and(data.Measurement!="Mean", data.Measurement!="Std")]

data = pd.concat((data,data.Measurement.str.split("_", expand=True)[[3,5]].\
       rename(columns={3:"Patient",5:"Gait"})),axis=1)
    
data.drop(columns=["Measurement"], inplace=True)

data["Side"] = data.Joint.str[0]

data.Joint.replace({"L_PELVIS_ANG_curto":"PELVIS",\
                    "R_PELVIS_ANG_curto":"PELVIS",\
                    "LHIP_ANG":"HIP",\
                    "RHIP_ANG":"HIP",\
                    "LKNEE_ANG":"KNEE",\
                    "RKNEE_ANG":"KNEE",\
                    "LVFT_ANG":"ANKLE",\
                    "RVFT_ANG":"ANKLE",\
                    "LHIP_MF_N":"HIP",\
                    "RHIP_MF_N":"HIP",\
                    "LKNEE_MF_N":"KNEE",\
                    "RKNEE_MF_N":"KNEE",\
                    "LANKLE_MF_N":"ANKLE",\
                    "RANKLE_MF_N":"ANKLE"},\
                    inplace=True)
    
if "Angles" in f.name:
    data["Measure"] = "Angle"

if "Moments" in f.name:
    data["Measure"] = "Moment"
