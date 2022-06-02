import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import yaml
import os




parameters = ['IV', 'CV', 'Istrip', 'Rpoly', 'Cac', 'Idiel', 'Cint', 'Rint']



def read_yml_configuration(yml_file):

  with open(yml_file, 'r') as f:
     conf = yaml.load(f, Loader=yaml.FullLoader)

  return conf





def query_data_from_DB(table_header_list, sql_table_prefix, parameter):

  
  print('The query of {} data from the CMS DB is gonna take a while'.format(parameter))

  error_message = 'ERROR: Exception'

  if parameter == "IV":
       
       p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
                      "select d.{}, d.{}, d.{}, d.{}, d.{} from trker_cmsr.tracker_sensor_{}_v d".format(*table_header_list, sql_table_prefix)], capture_output=True)

  else:
       p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
                      "select d.{}, d.{}, d.{}, d.{} from trker_cmsr.tracker_sensor_{}_v d".format(*table_header_list, sql_table_prefix)], capture_output=True)

  answer = p1.stdout.decode().splitlines()

  if error_message in answer:

      print('Query of {} data incomplete due to timeout error!'.format(parameter))

  else:
      print('Query of {} Data is complete'.format(parameter))
  
  return answer


 

def query_bad_strips_from_DB():
    
  print('Querying bad strips information from the CMS DB')

  p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
                      "select d.PART_NAME_LABEL, d.KIND_OF_FAILURE_ID, d.STRIP_NUMBER  from trker_cmsr.c8200 d"], capture_output=True)

  answer = p1.stdout.decode().splitlines()
  
  
  print('Query of bad strips information is complete')
  return answer


  
def query_runs_table_from_DB():


  print('Querying the run information from the CMS DB')

  p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
                      "select r.LOCATION, r.RUN_TYPE, r.RUN_NUMBER  from trker_cmsr.runs r"], capture_output=True)

  answer = p1.stdout.decode("latin-1").splitlines()
  
  print('Query of run information is complete')

  return answer




def query_SQC_summary_from_DB(table_sensor_type):
    

  print('Querying SQC summary information from the CMS DB')

  p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
                      "select d.NAME_LABEL, d.BATCH_NUMBER, d.DESCRIPTION, d.ASENSOR_TYPE, d.ASTATUS, d.ASENSOR_KNOWN_PROBLEM  from trker_cmsr.{} d".format(table_sensor_type)], capture_output=True)

  answer = p1.stdout.decode().splitlines()
  
  
  print('Query of SQC summary information is complete')
  return answer

 
  
def save_DB_table_as_json(answer_from_DB, filename):

  with open('data/{}.json'.format(filename), 'w') as f:
    json.dump(answer_from_DB[1:], f)




    
def make_list_from_json(file):

   with open(file, 'r') as file:
       data = json.load(file)
       
   return data[1:]

   
   

def process_list_with_data(data):

   split_data = [] 
    
   for i in data:
      split_data.append(i.split(','))
      
    
   return split_data
   



def make_dataframe(data):

  
  df = pd.DataFrame(split_data, columns=['ID_Number', 'Barcode', 'Volts', 'Cap'])
   
  
  return df
  



def generate_json_with_data(sqc_parameters):

  summary_tables = {'2-S': 'p1120', 'PS-s': 'p1160'}


  for i in parameters:
    
      headers = sqc_parameters[i]['table_headers']
      sql_table = sqc_parameters[i]['sql_table_prefix']
      answer_from_DB = query_data_from_DB(headers, sql_table, i)
      save_DB_table_as_json(answer_from_DB, i)

  runs_answer = query_runs_table_from_DB() 
  save_DB_table_as_json(runs_answer, 'runs')

  bad_strips_answer = query_bad_strips_from_DB()

  save_DB_table_as_json(bad_strips_answer, 'bad_strips')
 

  for sensor_type, table in summary_tables.items():

      summary_table_answer = query_SQC_summary_from_DB(table)

      save_DB_table_as_json(summary_table_answer, 'summary_{}_sensors'.format(sensor_type))




def run():

 
  try:
    os.mkdir("data")

  except FileExistsError:
    print("Directory data/ already exists")



  configuration = read_yml_configuration('SQC_parameters_DB.yml')
  sqc_parameters = configuration['SQC_parameters']

  generate_json_with_data(sqc_parameters)
  
 



if __name__=="__main__":

    run()   
  
 
