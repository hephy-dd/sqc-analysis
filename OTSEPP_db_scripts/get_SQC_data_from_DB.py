import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import argparse



parameters = ['IV', 'CV', 'Istrip', 'Rpoly', 'Cac', 'Idiel', 'Cint', 'Rint']



def read_yml_configuration(yml_file):

  with open(yml_file, 'r') as f:
     conf = yaml.load(f, Loader=yaml.FullLoader)

  return conf





def query_data_from_DB(query, table_header_list, sqltable_prefix, sensor_list, parameter):

  
  print('Querying {} data from CMS database'.format(parameter))

  error_message = 'ERROR: Exception'
  data = [] 

  for sensor in sensor_list:
      
      final_query = query + ' ' +  "where {}.SENSOR = '{}'".format(sqltable_prefix, sensor)
      
      p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
           '{}'.format(final_query)], capture_output=True)

      answer = p1.stdout.decode().splitlines()
      
      if len(answer)>2: # check if query does not return an empty-of-data list
          data.extend(answer[1:])
      
      
      if error_message in answer:

         print('Query of {} data incomplete due to timeout error!'.format(parameter))

      else:
          print('Query of {} data is successful'.format(parameter))
  
  return data

  
 

def process_list_with_data(data):

   split_data = [] 
    
   for i in data:
      split_data.append(i.split(','))
      
    
   return split_data
   



def make_dataframe(data, headers):

  list_of_data = process_list_with_data(data)  
  
  df = pd.DataFrame(list_of_data, columns = headers)
   
  df = df.dropna()

  df = df.loc[df['TYPE']=='SQC'] # only SQC data, dataframe contains also HPK data

  print(df) 
  
  return df
  


def plot_data(df, y_label, headers,  parameter):

    
    df[headers[1]] = pd.to_numeric(df[headers[1]]).abs() # volts or strip number

    df[headers[2]] = pd.to_numeric(df[headers[2]]).abs() # SQC parameter of interest, e.g Rint, Itotal
    
    fig, ax = plt.subplots()
    for key, group in df.groupby(['SENSOR']):
        ax.plot(group[headers[1]], group[headers[2]], label=key)


    plt.ylabel('{}'.format(y_label))
    plt.legend(loc='best')
    plt.savefig('plots_from_DB/{}.png'.format(parameter))

    plt.clf()



def parse_args():

   parser = argparse.ArgumentParser()
   parser.add_argument('-s', '--sensorlist', default = [], nargs='+',  help = 'give list of sensors to be analysed')
   return parser.parse_args()



def analyse_data(sqc_parameters):


  args = parse_args()
  sensor_list = args.sensorlist

  print('The list of the sensors to be analysed is: \n {}'.format(sensor_list))


  for i in parameters:
    
    query = sqc_parameters[i]['query']
    headers = sqc_parameters[i]['table_headers']
    sql_table = sqc_parameters[i]['sql_table_prefix']
    answer_from_DB = query_data_from_DB(query, headers, sql_table, sensor_list, i)

    df = make_dataframe(answer_from_DB, headers)
    plot_data(df, sqc_parameters[i]['label'], headers, i)



def run():

  try:
    os.makedirs('plots_from_DB')
  except FileExistsError:
     print('Directory plots_from_DB already exists')

  configuration = read_yml_configuration('SQC_parameters_DB.yml')
  sqc_parameters = configuration['SQC_parameters']

  analyse_data(sqc_parameters) 
 



if __name__=="__main__":

    run()   
  
 
