import pandas as pd
import numpy as np
import yaml
from lxml.html import parse
import urllib3 
from pretty_html_table import build_table
import requests
from bs4 import BeautifulSoup
import glob
import argparse
import json
import dicttoxml
import webbrowser
import base64
import os




#### This script works only internally at HEPHY or via HEPHY VPN!



def read_yaml_file(yml_file):
    with open(yml_file, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=str, required=True)
    return parser.parse_args()



def find_wafer_id(halfmoons):

      split_string = [x for x in halfmoons[0].split('_')]
      wafer_id = split_string[3]
    
      return wafer_id   
  
  
def generate_html(dataframe: pd.DataFrame):
    # get the table HTML from the dataframe
    table_html = dataframe.to_html(table_id="table")
    # construct the complete HTML with jQuery Data tables
    # You can disable paging or enable y scrolling on lines 20 and 21 respectively
    html = f"""
    <html>
    <header>
        <link href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css" rel="stylesheet">
    </header>
    <body>
    {table_html}
    <script src="https://code.jquery.com/jquery-3.6.0.slim.min.js" integrity="sha256-u7e5khyithlIdTpu22PHhENmPcRdFiHRjhAuHcs05RI=" crossorigin="anonymous"></script>
    <script type="text/javascript" src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <script>
        $(document).ready( function () {{
            $('#table').DataTable({{
                // paging: false,    
                // scrollY: 400,
            }});
        }});
    </script>
    </body>
    </html>
    """
    # return the html
    return html
 

 
def make_html_table(dictionary, batch):
       
                  
                  
      with open('PQC_results_{}.html'.format(batch), 'w') as f:
          f.write('<html><body> <h1>PQC tables <font color = #4000FF></font></h1>\n</body></html>')
          f.write('\n')
          
          for key, df in dictionary.items():
              for col in df:
                  df[col] = df[col].replace(np.nan, '---') 
              
             
              
              f.write('\n')
              f.write('<html><body> <h1>{} <font color = #4000FF></font></h1>\n</body></html>'.format(key))
              f.write('\n')
              html_table = build_table(df, 'blue_light', font_family='Open Sans, sans-serif', text_align= 'center', font_size='small')
              f.write(html_table)   
              f.write('\n')
      
             

def main():

  arg = parse_args()
  batch = os.path.basename(os.path.dirname(arg.batch)).split('_')[1]
  print(batch)

  PQC_parameters = read_yaml_file('PQC_parameters.yml')



  dict_with_parameters_medians = {par : []  for flute in PQC_parameters['Parameters'] for par in PQC_parameters['Parameters'][flute]}
  for flute in PQC_parameters['Parameters']: 
      for par in PQC_parameters['Parameters'][flute]:
        dict_with_parameters_medians['{}_err'.format(par)]=[]
  dict_with_parameters_medians['batch'] = []
  dict_with_parameters_medians['ID'] = []

  http = urllib3.PoolManager()

 
  #response = http.request('GET', 'http://heros.local.hephy.at/pqc-results/INCOMING/analysis_{}/results.html'.format(batch)) 

# parsed = BeautifulSoup(response.data.decode('utf-8'), features="lxml")
  with open('{}results.html'.format(arg.batch), 'r') as f:

    contents = f.read()

    parsed =  BeautifulSoup(contents, features="lxml")



  dataframe_collection = {}

  for number, table in enumerate(parsed.find_all('table')):

    def tableDataText(table): #neeed   
      """Parses a html segment started with tag <table> followed 
      by multiple <tr> (table rows) and inner <td> (table data) tags. 
      It returns a list of rows with inner columns. 
      Accepts only one <th> (table header/data) in the first row.
      """
      def rowgetDataText(tr, coltag='td'): # td (data) or th (header)       
         return [td.get_text(strip=True) for td in tr.find_all(coltag)]  
      rows = []
      strings = []
    
      trs = table.find_all('tr')
      headerow = rowgetDataText(trs[0], 'th')
      if headerow: # if there is a header row include first
          rows.append(headerow)
          trs = trs[1:]
      for tr in trs: # for every table row
        
          rows.append(rowgetDataText(tr, 'td') ) # data row 
          strings.append(rowgetDataText(tr, 'th') )
    
      def sort_out_empty_lines(list1, list2):
         
         halfmoons =[]
         data = []
         for i in list1:
           if len(i)>=1:
             if 'HPK_'in i[0]:
               halfmoons.append(i[0])    
    
         for j in list2[:-7]:
           
               if j == list2[0]:
                  j.pop(0)
        
               if len(j)>=2:  # a dummy way to get rid of the empty lines
                  data.append(j)      
         
                
         return halfmoons, data
     
     
      halfmoons, data = sort_out_empty_lines(strings, rows)     
      return data, halfmoons
   

   
    data_table, halfmoons = tableDataText(table)
    
       
    colnames=['failed']
    wafer_id = find_wafer_id(halfmoons)
      
    df = pd.DataFrame(halfmoons, columns =['Halfmoon'])
    
    df2 = pd.DataFrame(data_table[1:], columns = data_table[0])
    if 'i300' in list(df2.columns):
        df2 = df2.drop(df2.columns[[1]], axis=1) 
    
    df = df.join(df2)
 
    medians = {}
    stdev = {}
  
    for col in df:
      if col=='Halfmoon':
       
        medians[col] = 'Median'
        stdev[col] = 'Standard Deviation'
      
      else:
     
        df[col] = df[col].replace('---', np.nan)
        df[col] = df[col].replace('failed', np.nan)
        df[col] = pd.to_numeric(df[col])
        df[col] = df[col].round(decimals=2)
      
     
        medians[col] =round(df[col].median(),2)
        stdev[col] = round(df[col].std(), 2)
      
    units = []
    configuration = []
    
    for col in df: 

      if col =='Halfmoon':
        halfmoon_temp = df[col].values
        halfmoon_temp = ['_'.join(i.split('_')[1:]) for i in halfmoon_temp]
        halfmoon_temp = [hf.replace("VPX", "") for hf in halfmoon_temp]
        
        df[col] = halfmoon_temp
        units.append('')
        configuration.append('')

     
      else:
    
         df.rename({col: PQC_parameters['Plot']['parameters'][col][0]}, axis='columns', inplace=True)
    
         medians[PQC_parameters['Plot']['parameters'][col][0]] = medians.pop(col)
         stdev[PQC_parameters['Plot']['parameters'][col][0]] = stdev.pop(col)

         units.append(PQC_parameters['Plot']['parameters'][col][2])
         configuration.append(PQC_parameters['Plot']['parameters'][col][1])

  
  
    df = df.append(medians, ignore_index=True) 
    df = df.append(stdev, ignore_index=True)
    
    df.columns = pd.MultiIndex.from_tuples(list(zip(df.columns, configuration, units)))  
    df = df.reset_index(drop=True)

    html = generate_html(df)

    dataframe_collection['Flute {}'.format(number+1)] = df

  
  make_html_table(dataframe_collection, batch)
  


if __name__ == "__main__":
     main()
