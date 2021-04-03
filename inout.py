import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm
import pandas as pd
import wget


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
def checkfolders(ROOT_DIR = ROOT_DIR):
    PATH_unsup = ROOT_DIR + '/trained_models/unsupervised/'
    PATH_sup = ROOT_DIR + '/trained_models/supervised/'
    
    if not os.path.exists(PATH_unsup):
       os.makedirs(PATH_unsup)

    if not os.path.exists(PATH_sup):
       os.makedirs(PATH_sup)
       
    return ROOT_DIR, PATH_unsup, PATH_sup   


def get_dataframe(country, begin_date, average, ROOT_DIR = ROOT_DIR): 

    try:
        countries_to_fit = pd.read_csv(ROOT_DIR + '/real_data/countries.csv')
    except:
    # url of the raw csv dataset
        urls = [
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
            ]
    
        [wget.download(url, out= ROOT_DIR + 'real_data') for url in urls]
           
    confirmed = pd.read_csv('real_data/time_series_covid19_confirmed_global.csv')
    deaths = pd.read_csv('real_data/time_series_covid19_deaths_global.csv')
    recovered = pd.read_csv('real_data/time_series_covid19_recovered_global.csv')
        
    dates = confirmed.columns[4:]
        
    confirmed_long = confirmed.melt(
            id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
            value_vars=dates,
            var_name='Date',
            value_name='Confirmed'
        )
    
    dates = deaths.columns[4:]
        
    deaths_long = deaths.melt(
            id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
            value_vars=dates,
            var_name='Date',
            value_name='Deaths'
        )
    
    
    dates = recovered.columns[4:]
    
    recovered_long = recovered.melt(
            id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
            value_vars=dates,
            var_name='Date',
            value_name='Recovered'
        )
    
    
    # Merging confirmed_long and deaths_long
    full_table = confirmed_long.merge(
            right=deaths_long,
            how='left',
            on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']
        )
    # Merging full_table and recovered_long
    full_table = full_table.merge(
            right=recovered_long,
            how='left',
            on=['Province/State', 'Country/Region', 'Date', 'Lat', 'Long']
        )
    
    full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']
    full_grouped = full_table.groupby(['Date', 'Country/Region'])[[
            'Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()
    
   
    full_grouped['Date']= pd.to_datetime(full_grouped['Date'], format = '%m/%d/%y')
    full_grouped.to_csv('real_data/countries.csv')
   
    country_df = full_grouped[(full_grouped['Country/Region'] == country) & (full_grouped['Date'] > begin_date) ]
    country_df.reset_index()
    country_df_avg = country_df.resample(average, on='Date').mean()
    infected = np.array(country_df_avg['Active'])
    removed = np.array(country_df_avg['Deaths']) + np.array(country_df_avg['Recovered'])

    return infected, removed
    
def printLoss(loss, runTime, model_name, ROOT_DIR = ROOT_DIR):
    np.savetxt(ROOT_DIR + '/trained_models/Unsupervised/{}'.format(model_name), loss)
    #print('Training time (minutes):', runTime/60)
    print('Final training loss: ',  loss[-1] )
    plt.figure()
    plt.loglog(loss,'-b',alpha=0.975);
    plt.tight_layout()
    plt.ylabel('Loss');plt.xlabel('t')

    plt.savefig('Loss_history.png')
    plt.close()


def printSIRsolution(t_net, sTest, iTest, rTest, sExact, iExact, rExact, beta, gamma):
    sTest = sTest.detach().numpy()
    iTest = iTest.detach().numpy()
    rTest = rTest.detach().numpy()
    lineW = 2
    
    plt.plot(t_net, sTest,'-g', label='susceptible', linewidth=lineW);
    plt.plot(t_net, iTest,'-b', label='infective',linewidth=lineW, alpha=.5);
    plt.plot(t_net, rTest,'-y', label='removed',linewidth=lineW, alpha=.5)
    plt.plot(t_net, sExact,'--g', label='susceptible', linewidth=lineW);
    plt.plot(t_net, iExact,'--b', label='infective',linewidth=lineW, alpha=.5);
    plt.plot(t_net, rExact,'--y', label='removed',linewidth=lineW, alpha=.5)
    plt.savefig('solution.png')
    plt.title('beta = {beta}, gamma = {gamma} '.format(beta = beta, gamma = gamma))
    plt.legend()
    #plt.tight_layout()
    plt.close()
       
    
    
def printGroundThruth(t_net, x_exact, xTest,  xdot_exact, xdotTest):
    lineW = 4 # Line thickness
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    plt.plot(t_net, x_exact,'-g', label='Ground Truth', linewidth=lineW);
    plt.plot(t_net, xTest,'--b', label='Network',linewidth=lineW, alpha=.5);
    plt.ylabel('x(t)');plt.xlabel('t')
    plt.legend()
    
    plt.subplot(2,2,2)
    plt.plot(t_net, xdot_exact,'-g', label='Ground Truth', linewidth=lineW);
    plt.plot(t_net, xdotTest,'--b', label='Network',linewidth=lineW, alpha=.5);
    plt.ylabel('dx\dt');plt.xlabel('t')
    plt.legend()
    
    plt.savefig('simpleExp.png')
    plt.tight_layout()
    plt.close()

def print_scatter(Losses):

    X, Y, Z = Losses[:, 0], Losses[:, 1], Losses[:, 2]
    area = 20.0
    plt.scatter(X,Y,edgecolors='none',s=area,c=Z,
                norm=LogNorm())
    plt.colorbar()
    plt.savefig('scatter_loss.png')
    plt.tight_layout()
    plt.close()
    
    
