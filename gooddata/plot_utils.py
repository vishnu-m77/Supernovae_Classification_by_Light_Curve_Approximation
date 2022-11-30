import matplotlib.pyplot as plt

def get_object(data, object_id):
    anobject = data[data.object_id == object_id]
    return anobject

def get_passband(anobject, passband):
    light_curve = anobject[anobject.passband == passband]
    return light_curve

def plot_light_curves(anobject, passband2name, title=''):
    anobject = anobject.sort_values('mjd')
    plt.figure(figsize=(9, 5.5))
    for passband in range(len(passband2name)):
        light_curve = get_passband(anobject, passband)
        plt.scatter(light_curve['mjd'].values, light_curve['flux'].values)
        plt.plot(light_curve['mjd'].values, light_curve['predFlux'].values, label=passband2name[passband])
    plt.xlabel('Modified Julian Date')
    plt.xticks()
    plt.ylabel('Flux')
    plt.yticks()
    plt.legend()
    plt.title(title)


def plotLightCurve(object_, data, predFlux, passband2name, title):
    '''
    Parameters:
    object_: Name of the object for which plot is to be made
    data: DataFrame containing mjd, Actual Flux, processed passband
    predFlux: Predicted flux array
    passband2name: Similar to {0: 'g', 1: 'r'}
    '''
    anobject = get_object(data, object_)
    anobject["predFlux"] = predFlux
    plot_light_curves(anobject, passband2name, title)
    plt.show()


if __name__ == "__main__":
    object_ = 'ZTF20adaduxg'
    passband2name = {0: 'g', 1: 'r'}
    title = ''
    # df_obj_filt is original dataframe containing mjd, Actual Flux, processed passband
    # length of df_obj_filt == len(predFlux)
    # Tested with if (i+1)%5==0:
    # to match the length of predicted values with length of other parameters
    plotLightCurve(object_, df_obj_filt, predFlux, passband2name, title)