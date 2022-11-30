import pandas as pd
import matplotlib.pyplot as plt


def get_object(data, object_id):
    anobject = data[data.object_id == object_id]
    return anobject

def get_passband(anobject, passband):
    light_curve = anobject[anobject.passband == passband]
    return light_curve

def plot_light_curves(anobject, predDf, passband2name, title=''):
    anobject = anobject.sort_values('mjd')
    plt.figure(figsize=(9, 5.5))
    for passband in range(len(passband2name)):
        light_curve = get_passband(anobject, passband)
        plt.scatter(light_curve['mjd'].values, light_curve['flux'].values)
    
    plt.plot(predDf['augTimestamps'].values, predDf['predFluxPass1'].values, label=passband2name[0])
    plt.plot(predDf['augTimestamps'].values, predDf['predFluxPass2'].values, label=passband2name[1])

    plt.xlabel('Modified Julian Date')
    plt.xticks()
    plt.ylabel('Flux')
    plt.yticks()
    plt.legend()
    plt.title(title)


def plotLightCurve(object_, data, predFlux, aug_timestamp, passband2name, title):
    '''
    Parameters:
    object_: Name of the object for which plot is to be made
    data: DataFrame containing mjd, Actual Flux, processed passband
    predFlux: Predicted flux array
    passband2name: Similar to {0: 'g', 1: 'r'}
    '''
    anobject = get_object(data, object_)

    nAugTimeStamps = len(aug_timestamp)
    predDf = pd.DataFrame(
        {"predFluxPass1": predFlux[:nAugTimeStamps], 
         "predFluxPass2": predFlux[-nAugTimeStamps:],
         "augTimestamps": aug_timestamp
         })
    
    plot_light_curves(anobject, predDf, passband2name, title)
    plt.show()

'''calling code'''

#     passband2name = {0: 'g', 1: 'r'}
#     title = ''
#     # the length of predicted flux should twice that of aug timestamp, because we do it for two passbands
#     df is the filtered data frame containing only rows corresponding to object_ from ANTARES_NEW.CSV
#     plotLightCurve(object_, df, pred_flux, aug_timestamp, passband2name, title) 
