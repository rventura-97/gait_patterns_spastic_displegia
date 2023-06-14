# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import os
from itertools import compress
from collections import Counter
from matplotlib import pyplot as plt

# %%
data_class = pd.read_excel("dados_individuos.xlsx")
data_agg = [None]*len(os.listdir("Dados normativos"))*6

i = 0
for d in os.listdir("Dados normativos"):
    for f in os.listdir("Dados normativos/"+d):
        with open("Dados normativos/{}/{}".format(d,f), "r") as f:
            lines = f.readlines()
            
        # data = pd.DataFrame([lines[i].lstrip("\t").rstrip("\n").\
        #        split("\t")[1:] for i in range(5,len(lines))]).transpose()
            
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


data_agg["Axis"] = data_agg.File.str.slice(-5,-4)
data_agg.loc[data_agg.File.str.contains("Angles"),"Measure"] = "Angle"
data_agg.loc[data_agg.File.str.contains("Moments"),"Measure"] = "Moment"
data_agg.drop(columns=["File"], inplace=True)

data_agg["Side"] = data_agg.Joint.str.slice(0,1)


data_agg = data_agg[np.logical_and(data_agg.Measurement!="Mean",\
            data_agg.Measurement!="Std Dev")].reset_index(drop=True)

    
data_agg.left_leg.fillna("Normal",inplace=True)
data_agg.right_leg.fillna("Normal",inplace=True)

temp=data_agg.loc[data_agg[data_agg.Side=="L"].index,"left_leg"]
data_agg.loc[temp.index,"Class"] = temp

temp=data_agg.loc[data_agg[data_agg.Side=="R"].index,"right_leg"]
data_agg.loc[temp.index,"Class"] = temp

temp=data_agg.loc[data_agg[data_agg.Side=="L"].index,"left_leg_comments"]
data_agg.loc[temp.index,"Comments"] = temp

temp=data_agg.loc[data_agg[data_agg.Side=="R"].index,"right_leg_comments"]
data_agg.loc[temp.index,"Comments"] = temp

data_agg.drop(columns=["left_leg","right_leg","left_leg_comments","right_leg_comments"], inplace=True)

data_agg = data_agg[data_agg.Comments.isnull()].reset_index(drop=True)
data_agg.drop(columns=["Comments"], inplace=True)


# data_agg = data_agg[-data_agg.Joint.str.contains("|".join(["R_PELVIS_ANG_curto","R_PELVIS_ANG_longo",\
#                                                            "L_PELVIS_ANG_curto","L_PELVIS_ANG_longo",\
#                                                            "RFT_PROG_ANG_curto","RFT_PROG_ANG_longo",\
#                                                            "LFT_PROG_ANG_curto","LFT_PROG_ANG_longo",\
#                                                            "LFT_PROG_ANG","RFT_PROG_ANG"]))].reset_index(drop=True)
    
# data_agg.Joint.replace({"LHIP_ANG":"Hip",\
#                         "LKNEE_ANG":"Knee",\
#                         "LVFT_ANG":"Ankle",\
#                         "RHIP_ANG":"Hip",\
#                         "RKNEE_ANG":"Knee",\
#                         "RVFT_ANG":"Ankle",\
#                         "LHIP_MF_N":"Hip",\
#                         "LKNEE_MF_N":"Knee",\
#                         "LANKLE_MF_N":"Ankle",\
#                         "RHIP_MF_N":"Hip",\
#                         "RKNEE_MF_N":"Knee",\
#                         "RANKLE_MF_N":"Ankle",\
#                         "L_PELVIS_ANG":"Pelvis",\
#                         "R_PELVIS_ANG":"Pelvis"},inplace=True)
    

# data_agg.loc[data_agg[data_agg.Measurement.str.contains("Mean")].index,"Test"] = "Mean"   
# data_agg.loc[data_agg[data_agg.Measurement.str.contains("Std Dev")].index,"Test"] = "STD"   

temp = data_agg[data_agg.Measurement.str.contains("GAIT0")].Measurement.str.split("BAREFOOT_",expand=True)[1].str.split("_",expand=True)[0]
data_agg.loc[temp.index,"Test"] = temp

temp = data_agg[data_agg.Measurement.str.contains("WALK0")].Measurement.str.split("BAREFOOT_",expand=True)[1].str.split("_",expand=True)[0]
data_agg.loc[temp.index,"Test"] = temp

temp = data_agg[data_agg.Measurement.str.contains("GAIT_")].Measurement.str.split("BAREFOOT_",expand=True)[1].str.split("_CINE",expand=True)[0]
data_agg.loc[temp.index,"Test"] = temp
data_agg.drop(columns=["Measurement"],inplace=True)

# temp = data_agg[data_agg.Measure=="Moment"].weight
# data_agg.loc[temp.index,data_agg.columns[0:101]] = data_agg.loc[temp.index,data_agg.columns[0:101]].astype(float).divide(temp*9.8,axis='index').values
data_agg.drop(columns=["weight"],inplace=True)
del temp

data_agg.set_index(['Patient','Test','Side','Measure','Joint','Axis','Class'],inplace=True)
data_agg = data_agg.astype(float)

data_agg = data_agg.groupby(level=[0,1,2,3,4,5,6]).mean()

data_agg.to_csv("dados_agregados.csv")


# %%
means_by_joint_axis = data_agg.groupby(level=[2,3,4,5,6]).mean()
means_by_joint_axis_normal = means_by_joint_axis[means_by_joint_axis.index.get_level_values(4)=="Normal"]


