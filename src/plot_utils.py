import pandas as pd
import matplotlib.pyplot as plt

# extracting rows corresponding to object_id
def get_object(data, object_id): 
    anobject = data[data.object_id == object_id]
    return anobject


# extracting rows corresponding to passband
def get_passband(anobject, passband):
    light_curve = anobject[anobject.passband == passband]
    return light_curve

    
# Plotting light curves
def plot_light_curves(anobject, predDf, passband2name, filesavepath, title=''):
    anobject = anobject.sort_values('mjd') #sorting on basis of MJD(timestamp)
    plt.figure(figsize=(9, 5.5)) 
    # running loop for each passband 
    for passband in range(len(passband2name)):
        light_curve = get_passband(anobject, passband)
        #defining colour and label for passband 0 and 1
        if passband == 0:
          color = 'b'
          label= 'DATA: PB=g' 
        else: 
          color = 'g'
          label= 'DATA: PB=r'
        plt.scatter(light_curve['mjd'].values, light_curve['flux'].values, color=color, label=label)
    
    plt.plot(predDf['augTimestamps'].values, predDf['predFluxPass1'].values, label='NF: PB=g', color='b')
    plt.plot(predDf['augTimestamps'].values, predDf['predFluxPass2'].values, label='NF: PB=r', color='g')    
    plt.xlabel('Modified Julian Date')
    plt.xticks()
    plt.ylabel('Flux')
    plt.yticks()
    plt.legend()
    plt.title(title)
    plt.savefig(filesavepath)
    plt.clf()   # Clear figure
    plt.close() # Close a figure window


def plotLightCurve(object_, data, predFlux, aug_timestamp, passband2name, title=''):
    '''
    Parameters:
    object_: Name of the object for which plot is to be made
    data: Original DataFrame (Antares_new.csv) containing mjd, Actual Flux, processed passband
    predFlux: Predicted flux array
    aug_timestamp : timestamp corresponding to predicted flux
    passband2name: Similar to {0: 'g', 1: 'r'}
    title: title of the plot
    '''
    anobject = get_object(data, object_)

    nAugTimeStamps = len(aug_timestamp)
    predDf = pd.DataFrame(
        {"predFluxPass1": predFlux[:nAugTimeStamps], 
         "predFluxPass2": predFlux[-nAugTimeStamps:],
         "augTimestamps": aug_timestamp
         })
    title = 'flux vs timestamp '+ object_ 
    filesavepath = 'Nf_run_plots/Light_Curve_Flux_NF_'+object_+'.jpg'
    plot_light_curves(anobject, predDf, passband2name, filesavepath, title)
   
    #plt.clf()

'''calling code'''

#     passband2name = {0: 'g', 1: 'r'}
#     title = ''
#     # the length of predicted flux should twice that of aug timestamp, because we do it for two passbands
#     df is the filtered data frame containing only rows corresponding to object_ from ANTARES_NEW.CSV
#     plotLightCurve(object_, df, pred_flux, aug_timestamp, passband2name, title) 
