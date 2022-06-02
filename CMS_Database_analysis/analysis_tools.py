import query_data_from_DB as bd
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pretty_html_table import build_table
import warnings
from scipy.interpolate import CubicSpline
import test as t



pd.options.mode.chained_assignment = None

def make_list(filename):

    with open(filename) as f:

        list_with_values = json.load(f)

    split_data = []
    
    for j in list_with_values:
        split_data.append(j.split(','))

    return split_data




def make_dataframe(file_prefix, sql_parameter, headers):

   filename = '{}_data/{}.json'.format(file_prefix, sql_parameter)

   datalist = make_list(filename)
   df = pd.DataFrame(datalist, columns=headers)
   if sql_parameter !='runs' and file_prefix=='SQC':
       # drop data from older sensors
       
       df = df.loc[(~df['Sensor'].str.contains('HPK'))]
       df = df.loc[~df['Sensor'].str.contains('33234')]

   return df



def filter_dataframe(df, parameter, lower_limit, upper_limit):

   df[parameter] = pd.to_numeric(df[parameter]).abs()

   df = df.loc[(df[parameter]<float(upper_limit)) & (df[parameter]>float(lower_limit))]

   return df




def find_wafer_type(df, header):

    column_name = '{}_type'.format(header)

    df[column_name] = np.nan

    df[column_name] = df[column_name].mask(df[header].str.contains('2-S'), '2-S').mask(df[header].str.contains('PSS'), 'PSS').mask(df[header].str.contains('PSP'), 'PSP')

    return df



def find_batch(df, header):

    df['Batch'] = np.nan

    df['Batch'] = (df[header].str.split('_')).str[0]
    df['Batch'] = pd.to_numeric(df['Batch'])

    df = df.sort_values(by=['Batch'])

    df['Batch'] = df['Batch'].astype(str)

    return df




def find_location_SQC(df_location, df_data, header):

   
    merged_df = pd.merge(df_data, df_location, how='outer', on=['Run_number'])

    merged_df = merged_df.dropna(subset=[header, 'Run_number'])
    

    return merged_df




def find_location_PQC(df):

   
    df['Location'] = np.nan
    df['Location'] = df['Location'].mask(df['File'].str.contains('HPK_'), 'Perugia')

    df['Location'] = df['Location'].mask(df['File'].str.contains('Test'), 'Brown').mask(df['File'].str.contains('Demokritos'), 'Demokritos').mask(df['File'].str.contains('.json'), 'HEPHY')

    return df




def merge_dataframes(parameter_df, metadata_df, volts, parameter):


    parameter_df['Condition_number'] = pd.to_numeric(parameter_df['Condition_number'])

    metadata_df['Condition_number'] = pd.to_numeric(metadata_df['Condition_number'])

    parameter_df = parameter_df.sort_values(by=['Condition_number'])
    metadata_df = metadata_df.sort_values(by = ['Condition_number'])
    

    
    parameter_df['Condition_number'] = parameter_df['Condition_number']- 1


 
    merged_df = pd.merge(metadata_df, parameter_df[['Condition_number', str(volts), str(parameter)]], on='Condition_number', how='left')

    merged_df = merged_df.dropna(subset=[parameter])
    

    return merged_df




def sqc_sequence(df, runs_df):


    df = find_wafer_type(df, 'Sensor')
    df = find_batch(df, 'Sensor')
    df = find_location_SQC(runs_df, df, 'Sensor')

    return df




def pqc_sequence(df, metadata_df, header, parameter):

    list_of_attributes = [find_batch, find_wafer_type]

     
    df = merge_dataframes(df, metadata_df, parameter)


    df = find_location_PQC(df)
    
    for attr in list_of_attributes:
          df = attr(df, header)

    
    df[parameter] = pd.to_numeric(df[parameter])
    

    df = df.dropna(subset=[parameter])


    return df




def make_html_table(df, color, name):


    html_table = build_table(df, color, text_align='center')
    with open('html_tables/{}.html'.format(name), 'w') as f:
       # f.write('<html><body><h1> {} outside of specifications <font color = #4000FF></font></h1>\n</body></html>')
       # f.write('\n')
        f.write(html_table)




def find_SQC_values_out_of_specs(df, parameter, config_file):

   
    list_with_condition_larger_than = ['Cac', 'Rint']

    if parameter in list_with_condition_larger_than:

        bad_strips_df = df.loc[df[parameter]<config_file['SQC_parameters'][parameter]['Limit']]

    elif parameter== 'Rpoly':

        bad_strips_df = df.loc[~df[parameter].between(config_file['SQC_parameters'][parameter]['Limit']-0.5, config_file['SQC_parameters'][parameter]['Limit']+0.5)]

    else:

        bad_strips_df = df.loc[df[parameter].abs()>config_file['SQC_parameters'][parameter]['Limit']]

    return bad_strips_df




def find_outliers(df, config_file, parameter):


    upper_limit = config_file['up']
    lower_limit = config_file['low']


    return upper_limit, lower_limit




def plot_distribution(df, config_file, parameter, plot_type):

    from matplotlib.patches import Rectangle

    df[parameter]= df[parameter].abs()  
    low, upper = find_outliers(df, config_file, parameter)

    fig, ax = plt.subplots()
    

    hist, bins, patches = plt.hist(df[parameter], bins=80, range=(low, upper), color='blue', edgecolor='black', linewidth=1.2)
    patches[0].set_height(patches[0].get_height() + df.loc[df[parameter]<=low].shape[0])
    patches[0].set_facecolor('green')
    patches[-1].set_height(patches[-1].get_height() + df.loc[df[parameter]>=upper].shape[0])
    patches[-1].set_facecolor('yellow')

    ax.set_xlabe('{}'.format(config_file), fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of {}'.format(y_label), fontsize=12, fontweight='bold')
    extra = Rectange((0,0), 1, 1, fc='q', fill=False, edgecolor='none', linewidth=0)
    #label = 'Specs:

   # leg = plt.legend([extra], [label], frameon=False, loc='best')

    plt.savefig('figures/{}_{}/pdf'.format(parameter, plot_type))




def plot_time_evolution(df, parameter, measured_entity, ylabel):


    batch = list(dict.fromkeys(df['Batch']))


    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Batch', y=parameter, hue=measured_entity, s=25, linewidth=0, ax=ax)
    plt.xticks(batch, batch, rotation=90, va='top', fontsize=6)
    plt.locator_params(axis='x', nbins=len(batch)/4)
    ax.set(xlabel=None)
    if parameter == 'vdp_bulk' or parameter=='Diode_bulk':
        plt.ylim(0,11)
        plt.axhline(y=3.5, linestyle='dashed', color='black')
   # plt.axvline(x=16, linestyle='dashed', color='black')
    
    plt.ylabel(ylabel)
    plt.legend(loc='best')

    plt.savefig('figures/{}_evolution.pdf'.format(parameter))
    

