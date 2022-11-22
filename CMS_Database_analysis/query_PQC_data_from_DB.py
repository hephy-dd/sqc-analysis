import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import yaml
import os




parameters = ['IV', 'CV', 'FET']




def read_yml_configuration(yml_file):

  with open(yml_file, 'r') as f:
     conf = yaml.load(f, Loader=yaml.FullLoader)

  return conf





def query_data_from_DB(query,  parameter):

  
  print('The query of {} PQC data from the CMS DB is gonna take a while'.format(parameter))

      
  p1 = subprocess.run(['python3', 'rhapi.py', '-n', '--login', '-all', '--url=https://cmsdca.cern.ch/trk_rhapi',
                      "{}".format(query)], capture_output=True)
  
       
  answer = p1.stdout.decode().splitlines()
  print('Query of {} PQC data is complete'.format(parameter))
  
  return answer


   
 
  
def save_DB_table_as_json(answer_from_DB, filename):

  with open('PQC_data/{}.json'.format(filename), 'w') as file:
    json.dump(answer_from_DB[1:], file)




    
def make_list_from_json(file):

   with open(file, 'r') as file:
       data = json.load(file)
       
   return data

   
   

def process_list_with_data(data):

   split_data = [] 
    
   for i in data:
      split_data.append(i.split(','))
      
    
   return split_data
    



def generate_json_with_data(pqc_parameters):

  for i in parameters:
    
      query = pqc_parameters[str(i)]['query']
      
      answer_from_DB = query_data_from_DB(query, str(i))
      save_DB_table_as_json(answer_from_DB, str(i))



def run():

 
  try:
    os.mkdir("PQC_data")

  except FileExistsError:
    print("Directory data/ already exists")



  configuration = read_yml_configuration('PQC_parameters_DB.yml')
  pqc_parameters = configuration['PQC_tables']

  generate_json_with_data(pqc_parameters)
  
 



if __name__=="__main__":

    run()   
  
 
