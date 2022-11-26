

import numpy as np
import pandas as pd

passband2name = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}
passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                 3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}


dataPath = 'ANTARES_NEW.csv'

df_all = pd.read_csv(dataPath)
df_all = df_all.drop('Unnamed: 0', 1)

obj_names = df_all['object_id'].unique()

df_all.loc[df_all.obj_type == 'SN Ia', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-91T', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-pec', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Iax', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-91bg', 'obj_type'] = 1
df_all.loc[df_all.obj_type == 'SN Ia-CSM', 'obj_type'] = 1
df_all.loc[df_all.obj_type != 1, 'obj_type'] = 0


newL = []
for pb in df_all["passband"]:
    newL.append(passband2lam[pb])

df_all["Processed_Pass_Band"] = newL

df_all.to_csv("processedAntaresData.csv", index=False)