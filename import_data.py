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
            
        data = pd.DataFrame([lines[i].lstrip("\t").rstrip("\n").\
               split("\t")[1:] for i in range(5,len(lines))]).transpose()
        #data = data.astype("float")

        measurements = lines[0].lstrip("\t").rstrip("\n").split("\t")
        joints = lines[1].lstrip("\t").rstrip("\n").split("\t")
        
        # measurements_idxs = ["BAREFOOT" in var for var in col_heads]
        # measurements = list(compress(col_heads, measurements_idxs))
        # data = data.iloc[measurements_idxs,:]
        # joints = lines[1].split()
        # joints = list(compress(joints,measurements_idxs))

        data["Joint"] = joints
        data["Measurement"] = measurements
        data["Patient"] = d
        data["File"] = f.name.split("/")[-1]
        del measurements, joints

        

        # data = pd.concat((data,data.Measurement.str.split("_", expand=True)[[3,5]].\
        #        rename(columns={3:"Patient",5:"Gait"})),axis=1)
            
        # data.drop(columns=["Measurement"], inplace=True)

        # data["Side"] = data.Joint.str[0]
        # data["Axis"] = f.name.split(".txt")[0].split("_")[1]

        # data.Joint.replace({"L_PELVIS_ANG_curto":"PELVIS",\
        #                     "R_PELVIS_ANG_curto":"PELVIS",\
        #                     "LHIP_ANG":"HIP",\
        #                     "RHIP_ANG":"HIP",\
        #                     "LKNEE_ANG":"KNEE",\
        #                     "RKNEE_ANG":"KNEE",\
        #                     "LVFT_ANG":"ANKLE",\
        #                     "RVFT_ANG":"ANKLE",\
        #                     "LHIP_MF_N":"HIP",\
        #                     "RHIP_MF_N":"HIP",\
        #                     "LKNEE_MF_N":"KNEE",\
        #                     "RKNEE_MF_N":"KNEE",\
        #                     "LANKLE_MF_N":"ANKLE",\
        #                     "RANKLE_MF_N":"ANKLE"},\
        #                     inplace=True)
           
        # indiv_data = data_class[data_class.code == data.Patient.iloc[0]]
        # weight = indiv_data.weight.values[0]
            
        # if "Angles" in f.name:
        #     data["Measure"] = "Angle"

        # if "Moments" in f.name:
        #     data["Measure"] = "Moment"    
        #     data.iloc[:,0:101] = data.iloc[:,0:101]/(weight*9.8)
                
        #####################################################################    
            
        # if data.Side.values[0] == "L":
        #     data["Class"] = indiv_data.left_leg.values[0]
        #     data["Comments"] = indiv_data.left_leg_comments.values[0]
            
        # if data.Side.values[0] == "R":
        #     data["Class"] = indiv_data.right_leg.values[0]
        #     data["Comments"] = indiv_data.right_leg_comments.values[0]
            
        ######################################################################
            
        # if data.Class.isnull().all() == True:
        #     data["Class"] = "Normal"
            
        # if data.Comments.isnull().all() == True:
        #     data["Comments"] = "None"
        
        # data_agg[i] = data.groupby(["Patient","Joint","Gait","Side","Axis",\
        #               "Measure","Class","Comments"]).mean().reset_index()
        
        data_agg[i] = data
        i += 1
        print("Processed file {} / {}".format(str(i), str(len(data_agg))))


data_agg = pd.concat(data_agg)

data_agg = data_agg.merge(data_class.rename(columns={"code":"Patient"}))

data_agg["Axis"] = data_agg.Joint.str.slice(0,1)

# data_agg.set_index(["Patient","Joint","Gait","Side","Axis",\
#               "Measure","Class","Comments"], drop=True, inplace=True)
    
# data_agg = data_agg.astype(float)

data_agg.to_csv("dados_agregados.csv")

# %%