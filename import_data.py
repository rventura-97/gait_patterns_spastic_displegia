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


temp = data_agg[-data_agg.Class.str.contains("Normal")*data_agg.Joint.str.contains("FT_PROG_ANG")*data_agg.Axis.str.contains("Z")]
data_agg.loc[temp.index,"Axis"] = "Y"


data_agg = data_agg[data_agg.Joint.str.contains("|".join(["LFT_PROG_ANG_curto","RFT_PROG_ANG_curto",\
                                                          "LFT_PROG_ANG_longo","RFT_PROG_ANG_longo",\
                                                            "LFT_PROG_ANG","RFT_PROG_ANG",\
                                                            "LVFT_ANG","RVFT_ANG",\
                                                            "LHIP_ANG","RHIP_ANG",\
                                                            "LKNEE_ANG","RKNEE_ANG",\
                                                            "L_PELVIS_ANG_curto","R_PELVIS_ANG_curto",\
                                                            "L_PELVIS_ANG_longo","R_PELVIS_ANG_longo",\
                                                            "L_PELVIS_ANG","R_PELVIS_ANG",\
                                                            "LANKLE_MF_N","RANKLE_MF_N",\
                                                            "LKNEE_MF_N","RKNEE_MF_N",\
                                                            "LHIP_MF_N","RHIP_MF_N"]))].reset_index(drop=True)
    
data_agg.Joint.replace({"LFT_PROG_ANG_curto":"Ankle",\
                        "RFT_PROG_ANG_curto":"Ankle",\
                        "LFT_PROG_ANG_longo":"Ankle",\
                        "RFT_PROG_ANG_longo":"Ankle",\
                        "LFT_PROG_ANG":"Ankle",\
                        "RFT_PROG_ANG":"Ankle",\
                        "LVFT_ANG":"Ankle",\
                        "RVFT_ANG":"Ankle",\
                        "LHIP_ANG":"Hip",\
                        "RHIP_ANG":"Hip",\
                        "LKNEE_ANG":"Knee",\
                        "RKNEE_ANG":"Knee",\
                        "L_PELVIS_ANG_curto":"Pelvis",\
                        "R_PELVIS_ANG_curto":"Pelvis",\
                        "L_PELVIS_ANG_longo":"Pelvis",\
                        "R_PELVIS_ANG_longo":"Pelvis",\
                        "L_PELVIS_ANG":"Pelvis",\
                        "R_PELVIS_ANG":"Pelvis",\
                        "LANKLE_MF_N":"Ankle",\
                        "RANKLE_MF_N":"Ankle",\
                        "LKNEE_MF_N":"Knee",\
                        "RKNEE_MF_N":"Knee",\
                        "LHIP_MF_N":"Hip",\
                        "RHIP_MF_N":"Hip"},inplace=True)
    



temp = data_agg[data_agg.Measurement.str.contains("GAIT0")].Measurement.str.split("BAREFOOT_",expand=True)[1].str.split("_",expand=True)[0]
data_agg.loc[temp.index,"Test"] = temp

temp = data_agg[data_agg.Measurement.str.contains("WALK0")].Measurement.str.split("BAREFOOT_",expand=True)[1].str.split("_",expand=True)[0]
data_agg.loc[temp.index,"Test"] = temp

temp = data_agg[data_agg.Measurement.str.contains("GAIT_")].Measurement.str.split("BAREFOOT_",expand=True)[1].str.split("_CINE",expand=True)[0]
data_agg.loc[temp.index,"Test"] = temp
data_agg.drop(columns=["Measurement"],inplace=True)

temp = data_agg[data_agg.Measure=="Moment"].weight
data_agg.loc[temp.index,data_agg.columns[0:101]] = data_agg.loc[temp.index,data_agg.columns[0:101]].astype(float).divide(temp*9.8,axis='index').values
data_agg.drop(columns=["weight"],inplace=True)
del temp

data_agg.set_index(['Patient','Test','Side','Measure','Joint','Axis','Class'],inplace=True)
data_agg = data_agg.astype(float)

data_agg = data_agg.groupby(level=[0,1,2,3,4,5,6]).mean()

data_agg.to_csv("dados_agregados_totais.csv")

        
unique_leg_gaits = data_agg.groupby(level=[0,1,2]).size()

valid_leg_gait_vars = pd.DataFrame([["Angle","Ankle","X"],["Angle","Ankle","Y"],\
                                    ["Angle","Ankle","Z"],["Angle","Hip","X"],\
                                    ["Angle","Hip","Y"],["Angle","Hip","Z"],\
                                    ["Angle","Knee","X"],["Angle","Knee","Y"],\
                                    ["Angle","Knee","Z"],["Angle","Pelvis","X"],\
                                    ["Angle","Pelvis","Y"],["Angle","Pelvis","Z"],\
                                    ["Moment","Ankle","X"],["Moment","Ankle","Y"],\
                                    ["Moment","Ankle","Z"],["Moment","Hip","X"],\
                                    ["Moment","Hip","Y"],["Moment","Hip","Z"],\
                                    ["Moment","Knee","X"],["Moment","Knee","Y"],\
                                    ["Moment","Knee","Z"]],\
                                   columns=["Measure","Joint","Axis"])
    
data_agg_legs = [None]*unique_leg_gaits.shape[0]

for i in range(0, unique_leg_gaits.size):
    leg_gait_data = data_agg[(data_agg.index.get_level_values(0) == unique_leg_gaits.index[i][0])*\
                             (data_agg.index.get_level_values(1) == unique_leg_gaits.index[i][1])*\
                             (data_agg.index.get_level_values(2) == unique_leg_gaits.index[i][2])]
    
    if pd.merge(leg_gait_data.index.to_frame().iloc[:,[3,4,5]].\
       reset_index(drop=True),valid_leg_gait_vars,\
       how="inner").shape[0] == 21:
        
        data_agg_legs[i] = leg_gait_data
        
        
data_agg_legs = pd.concat(data_agg_legs)

data_agg_legs.to_csv("dados_agregados_pernas.csv")   

data_agg_means = data_agg_legs.groupby(level=[2,3,4,5,6]).mean()   
data_agg_means.to_csv("dados_agregados_medias.csv")         

# %%

# grafico das contagens por classe

# %%   
plt.figure(dpi=300)
                                             
plot_data_cycles = data_agg_legs[(data_agg_legs.index.get_level_values(2)=="L")*\
                                 (data_agg_legs.index.get_level_values(3)=="Angle")*\
                                 (data_agg_legs.index.get_level_values(4)=="Ankle")*\
                                 (data_agg_legs.index.get_level_values(5)=="X")*\
                                 (data_agg_legs.index.get_level_values(6)=="Normal")]                        
plot_data_mean = data_agg_means[(data_agg_means.index.get_level_values(0)=="L")*\
                                 (data_agg_means.index.get_level_values(1)=="Angle")*\
                                 (data_agg_means.index.get_level_values(2)=="Ankle")*\
                                 (data_agg_means.index.get_level_values(3)=="X")*\
                                 (data_agg_means.index.get_level_values(4)=="Normal")]

    
plt.plot(plot_data_cycles.values.transpose(),color="gray",alpha=0.2)
plt.plot(plot_data_mean.values.transpose(),color="k",linestyle="dashed")
plt.xlabel("Gait Cycle (%)")
plt.ylabel("Degrees")
plt.show()
plt.savefig('filename.png')

# %%   
plt.figure(dpi=300)
                                             
plot_data_cycles = data_agg_legs[(data_agg_legs.index.get_level_values(2)=="L")*\
                                 (data_agg_legs.index.get_level_values(3)=="Angle")*\
                                 (data_agg_legs.index.get_level_values(4)=="Ankle")*\
                                 (data_agg_legs.index.get_level_values(5)=="X")*\
                                 (data_agg_legs.index.get_level_values(6)=="Crouch Gait")]                        
plot_data_mean = data_agg_means[(data_agg_means.index.get_level_values(0)=="L")*\
                                 (data_agg_means.index.get_level_values(1)=="Angle")*\
                                 (data_agg_means.index.get_level_values(2)=="Ankle")*\
                                 (data_agg_means.index.get_level_values(3)=="X")*\
                                 (data_agg_means.index.get_level_values(4)=="Crouch Gait")]

    
plt.plot(plot_data_cycles.values.transpose(),color="gray",alpha=0.2)
plt.plot(plot_data_mean.values.transpose(),color="k",linestyle="dashed")
plt.xlabel("Gait Cycle (%)")
plt.ylabel("Degrees")
plt.show()
plt.savefig('filename.png')



                                     
# %%
data_var_means = data_agg.groupby(level=[2,3,4,5,6]).mean()
for i in range(0,data_var_means.shape[0]):
    if data_var_means.index.get_level_values(4)[i] == "Normal":
        plt.plot(data_var_means.iloc[i,:])
        plt.xlabel("Cycle")
        plt.title(" ".join(data_var_means.index[i][0:4]))
        plt.show()