import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LogNorm
import pandas as pd
import wget
import shutil
import datetime as dt
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def checkfolders(ROOT_DIR = ROOT_DIR):
    PATH_plots = ROOT_DIR + '/plots/'
    PATH_trained = ROOT_DIR + '/trained_models/'
    PATH_checkpoint = ROOT_DIR + '/plots/Checkpoint'  
    PATH_real = ROOT_DIR +  '/real_data/' 
             
    if not os.path.exists(PATH_plots):
       os.makedirs(PATH_plots)

    if os.path.exists(PATH_checkpoint):
       shutil.rmtree(PATH_checkpoint)
       os.makedirs(PATH_checkpoint)
    else:
       os.makedirs(PATH_checkpoint)
       
    if not os.path.exists(PATH_trained):
       os.makedirs(PATH_trained)
  
    if not os.path.exists(PATH_real):
       os.makedirs(PATH_real)
  
    return ROOT_DIR 

country_populations = {'Italy': 60460000,
                                  'Spain': 46940000,
                                  'Greece': 10720000,
                                  'France': 66990000,
                                  'Germany': 83020000,
                                  'Switzerland': 8570000,
                                  'United Kingdom': 66650000,
                                  'Russia': 146793000,
                                  'US': 328200000,
                                  'Sweden': 10230000,
                                  'New Zealand': 4886000,
                                  'Israel' : 9053000,
                                  'Chile' : 18950000,
                                  'Hungary': 9773000,
                                  'United Arab Emirates' : 9771000,
                                  'Uruguay': 3462000,
                                  'Turkey' : 82000000,
                                  'Netherlands' : 17280000,
                                  'Argentina' : 44940000}
                                  
def get_dataframe(country, begin_date, end_date, recovered_mode = None, moving_average = True,  lam = 0.95, ROOT_DIR = ROOT_DIR): 
    
    begin_date = datetime.strptime(begin_date, '%m/%d/%y')
    end_date = datetime.strptime(end_date, '%m/%d/%y')
    Npop = country_populations[country]
    
    try:
        countries_to_fit = pd.read_csv(ROOT_DIR + '/real_data/CSSE_data_countries.csv')
    except:
    # url of the raw csv datasets
        urls = [
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
                'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
                'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
            ]
    
        [wget.download(url, out= ROOT_DIR + '/real_data/') for url in urls]
           
    confirmed = pd.read_csv('real_data/time_series_covid19_confirmed_global.csv')
    deaths = pd.read_csv('real_data/time_series_covid19_deaths_global.csv')
    recovered = pd.read_csv('real_data/time_series_covid19_recovered_global.csv')
    owid_df = pd.read_csv('real_data/owid-covid-data.csv')   
     
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
    
    owid_df = owid_df.rename(columns = {'location': 'Country/Region','date' : 'Date'})
   
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
    full_grouped = full_table.groupby(['Date', 'Country/Region'])[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum().reset_index()
    
    full_grouped['Date']= pd.to_datetime(full_grouped['Date'], format = '%m/%d/%y')  
    full_grouped = full_grouped.sort_values(by='Date',ascending=True)                                                                             
    full_grouped.to_csv(ROOT_DIR + '/real_data/CSSE_data_countries.csv')
 
    owid_df['Date'] = pd.to_datetime(owid_df['Date'])
    
    # JHU dataframe, sometimes the 'removed' population is missing
    if recovered_mode == 'retarded':
       # if the recovered population is missing from the data, we use the confirmed population shifted back in time by 14 days instead
       country_df = full_grouped[(full_grouped['Country/Region'] == country)]
       country_df = country_df.reset_index(drop=True)     
       country_df = country_df.sort_values(by='Date',ascending=True)  
       country_df['Removed'] = country_df['Confirmed'].shift(-14)
       country_df['Active'] = country_df['Removed'] - country_df['Confirmed']

       if moving_average:
           begin_date_moving = begin_date - dt.timedelta(days = 6)
           country_df = country_df[(country_df['Date'] >= begin_date_moving) & (country_df['Date'] <=  end_date)] 
           country_df.loc[: , 'Active'] = country_df.loc[: , 'Active'].rolling(window=7).mean() 
           country_df = country_df.sort_values(by='Date',ascending=True)                                                                               
           country_df = country_df.reset_index(drop=True)
           country_df = country_df.loc[6:]
       else:
           country_df = country_df[(country_df['Date'] >= begin_date) & (country_df['Date'] <= end_date)]
           country_df = country_df.sort_values(by='Date',ascending=True)                                                                               
           country_df = country_df.reset_index(drop=True)     

    else: 

       if moving_average:
          begin_date_moving = begin_date - dt.timedelta(days = 6)
          country_df = full_grouped[(full_grouped['Country/Region'] == country) & (full_grouped['Date'] >= begin_date_moving) & (full_grouped['Date'] <=  end_date)] 
          country_df.loc[:,'Active'] = country_df.loc[:,'Active'].rolling(window=7).mean() 
          country_df = country_df.reset_index(drop=True)
          country_df = country_df.loc[6:]
       else:
          country_df = full_grouped[(full_grouped['Country/Region'] == country) & (full_grouped['Date'] >= begin_date) & (full_grouped['Date'] <= end_date)]    
          country_df = country_df.sort_values(by='Date',ascending=True)                                                                                    
          country_df = country_df.reset_index(drop=True)                                                             

    # owid dataframe to retrieve the vaccinated population
    if moving_average:
       begin_date_moving = begin_date - dt.timedelta(days = 6)
       owid_country_df = owid_df[(owid_df['Country/Region'] == country) & (owid_df['Date'] >= begin_date_moving) & (owid_df['Date'] <= end_date)]
       owid_country_df.loc[:,'people_vaccinated'] = owid_country_df.loc[:,'people_vaccinated'].rolling(window=7).mean() 
       owid_country_df.loc[:,'people_fully_vaccinated'] = owid_country_df.loc[:,'people_fully_vaccinated'].rolling(window=7).mean()  
       owid_country_df = owid_country_df.sort_values(by='Date',ascending=True)                                                                                    
       owid_country_df = owid_country_df.reset_index(drop=True)
       owid_country_df = owid_country_df.loc[6:]   
          
    else:    
       owid_country_df = owid_df[(owid_df['Country/Region'] == country) & (owid_df['Date'] >= begin_date) & (owid_df['Date'] <= end_date)  ]   
       owid_country_df = owid_country_df.reset_index(drop=True)     
                                                                                                                                                                                        
    vaccinated_owid = owid_country_df.loc[:,['Date','people_vaccinated', 'people_fully_vaccinated']]
    vaccinated_owid[['people_vaccinated','people_fully_vaccinated']] = vaccinated_owid[['people_vaccinated', 'people_fully_vaccinated']].fillna(0) 
    vaccinated_owid = vaccinated_owid.reset_index(drop = True)                                                                                                                         

    # the 'vaccinated' population of the SAIVR model contains people who are vaccinated but still not immune                                   
    vaccinated_removed = vaccinated_owid.loc[:, ['Date','people_fully_vaccinated']]
    people_vaccinated = vaccinated_owid.loc[:, 'people_vaccinated'] - vaccinated_owid.loc[:, 'people_fully_vaccinated']
    vaccinated = [vaccinated_owid.loc[:, ['Date']], people_vaccinated]
    vaccinated = pd.concat(vaccinated, axis = 1)
    vaccinated = vaccinated.rename(columns = {0 : 'vaccinated'})
                                                                                                              
    infected = np.array(country_df['Active'])
    vaccinated = np.array(vaccinated['vaccinated']) * lam # rescaled by the average vaccine efficiency
    vaccinated_full = np.array(vaccinated_removed['people_fully_vaccinated']) * lam 
    
    if recovered_mode == 'retarded':
       removed = np.array(country_df['Removed']) + vaccinated_full + np.array(country_df['Recovered'])
    else:
       removed = np.array(country_df['Recovered']) + vaccinated_full + np.array(country_df['Deaths'])                                                                                                           

    time_series_dict = {}
    time_sequence = np.linspace(0., len(infected), len(infected))
    i_real = infected.reshape(-1,1)/Npop
    v_real = vaccinated.reshape(-1,1)/Npop
    v_full_real = vaccinated_full.reshape(-1,1)/Npop
    r_real = removed.reshape(-1,1)/Npop 
    s_real = 1. - r_real - v_full_real
    
    plt.plot(time_sequence, i_real,'-r', label='infective');
    plt.title('{} real data'.format(country))
    plt.legend()
    plt.savefig('plots/Infected_real_data_{}.png'.format(country))
    plt.close()
    
    plt.plot(time_sequence, v_real,'-b', label='people_vaccinated');
    plt.plot(time_sequence, v_full_real,'-k', label='people_fully_vaccinated');
    plt.title('{} real data'.format(country))
    plt.legend()
    plt.savefig('plots/Vaccinated_real_data_{}.png'.format(country))
    plt.close()
        
    plt.plot(time_sequence, r_real,'-k', label='Deaths + Recovered + Fully Vaccinated');
    plt.title('{} real data'.format(country))
    plt.legend()
    plt.savefig('plots/known_Removed_real_data_{}.png'.format(country))
    plt.close()
    
    for j,t in enumerate(time_sequence):
        time_series_dict[t] = [float(s_real[j]), float(i_real[j]), float(v_real[j]), float(r_real[j])]

    return time_series_dict
    
    
def printLoss(loss, runTime, model_name, ROOT_DIR = ROOT_DIR):
    np.savetxt(ROOT_DIR + '/plots/Loss_history_{}'.format(model_name), loss)
    print('Final training loss: ',  loss[-1] )
    plt.figure()
    plt.loglog(loss,'-b',alpha=0.975);
    plt.tight_layout()
    plt.ylabel('Loss');plt.xlabel('t')

    plt.savefig('plots/Loss_history.png')
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
    plt.savefig('plots/solution.png')
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
    
    plt.savefig('plots/simpleExp.png')
    plt.tight_layout()
    plt.close()

def print_scatter(Losses):

    X, Y, Z = Losses[:, 0], Losses[:, 1], Losses[:, 2]
    area = 20.0
    plt.scatter(X,Y,edgecolors='none',s=area,c=Z,
                norm=LogNorm())
    plt.colorbar()
    plt.savefig('plots/scatter_loss.png')
    plt.tight_layout()
    plt.close()
    
    
