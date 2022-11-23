

import numpy as np
import pandas as pd


passband2name = {0: 'u', 1: 'g', 2: 'r', 3: 'i', 4: 'z', 5: 'y'}
passband2lam  = {0: np.log10(3751.36), 1: np.log10(4741.64), 2: np.log10(6173.23), 
                    3: np.log10(7501.62), 4: np.log10(8679.19), 5: np.log10(9711.53)}


def getobject(data, object_id):
    anobject = data[data.object_id == object_id]
    return anobject


def getpassband(anobject, passband):
    light_curve = anobject[anobject.passband == passband]
    return light_curve


def goodobject(anobject):
    good = 1
    
    # remove all objects with negative flux values
    if anobject['flux'].values.min() < 0:
        good = 0
    
    # keep only objects with at least 10 observations in at least 3 passbands # We may have some changes on these choices. 
    count = 0
    for passband in range(6):
        if len(getpassband(anobject, passband)) < 10:
            count += 1
    if count > 3:
        good = 0
        
    # keep only objects without large breaks in observations # not finalised at this point
    anobject = anobject.sort_values('mjd')
    mjd = anobject['mjd'].values
    if np.diff(mjd, 1).max() > 50:
        good = 0
    
    return good


if __name__ == "__main__":
    outputFilePath = 'gooddata.csv'
    dataPath = 'plasticc_train_lightcurves.csv.gz'
    metaDataPath = 'plasticc_train_metadata.csv.gz'
    
    data = pd.read_csv(dataPath)
    metadata = pd.read_csv(metaDataPath)

    data = data[data.detected_bool == 1]
    object_ids = np.unique(data.object_id)

    outDf = pd.DataFrame(columns=data.columns)
    for i, val in enumerate(object_ids[::50]):
        anobject = getobject(data, val) 
        if not goodobject(anobject):
            continue
    
        outDf = pd.concat([outDf, anobject])
        outDf.reset_index(drop = True, inplace = True)
        
    outDf.to_csv(outputFilePath, index = False)