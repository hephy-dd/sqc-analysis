import SQC_analysis_tools as sqc
import matplotlib.pyplot as plt
import glob as glob
import os
import argparse
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages




color = ['red', 'blue', 'black', 'gold', 'purple', 'green', 'violet', 'peru', 'lightskyblue',
         'pink', 'slategray', 'lime', 'lightgreen', 'cyan', 'brown', 'darkcyan',  'lightgray', 'tan', 'thistle' ]




def IV_analysis(file):
     
     # reads an IVCV file and generates a dataframe with IV data. Should work also for HPK data
     
     if 'HPK' in file:
         start_line = 23
         
     else:
         start_line = 9

     df = sqc.make_Dataframe_IVCV(file, start_line)
     voltage = df['Voltage [V]'].abs()
     current = df['current [A]'].abs()
    
         
     return voltage, current


def CV_analysis(file):

    # reads an IVCV file and generates a dataframe with CV data. Should work also for HPK data
     
    if 'HPK' in file:
         start_line = 76
         
    else:
         start_line = 9

    df = sqc.make_Dataframe_IVCV(file, start_line)
    voltage = df['Voltage [V]'].abs()
    cap = df['capacitance [F]'].abs()
     
  
    c_inv = 1./cap**2
  
    return voltage, c_inv

 
  
def stripscan_analysis(filename, parameter):

     # reads an Stripscan file and generates a dataframe with all data. Returns only the strip-parameter 

     if '2-S' in filename:
        sensor_id = '2-S'
     else:
        sensor_id = 'PSS'

     df = sqc.make_Dataframe_Stripscan(parameter, filename, sensor_id)

    
     strip = df['Pad']
     parameter = df[parameter]

     return strip, parameter 
  
  

def overlay_plots(files, key, values):

    # itterates over all input files and produces the plots.
    # key-values refer to the dictionary produced from the configuration file
        
    color_index =-1
    flag = True
   
    for file in files:
      
       if not "HPK" in file:
           lbl='-'.join(os.path.basename(file).split('-')[:1 if 'PSS' in file else 2])
           lbl='_'.join(lbl.split('_')[1:])
           lbl=lbl.split('.')[0]
       else:
          lbl = os.path.basename(file)  
    
    
   
       if "IVC" in file or "HPK" in file:
                 
           x,y = globals()['{}_analysis'.format(key)](file)
            
       elif "Str" in file:  
           
           x, y = stripscan_analysis(file, values['variables'][1])
         
         
       y = y*(float(values['units_conversion']))
       color_index +=1

       sqc.plot_graph(abs(x), abs(y), color[color_index], lbl, values['title'], values['variables'][0] + ' ' , values['variables'][1] + ' ' +'[{}]'.format(values['units']))
        
       
       if (y.abs()).max() > (50*y.abs()).median() and "Str" in file:
         
          plt.yscale('log')
          
    
       
     
def find_most_recent_file(files):

   # in case of remeasurements, it returns the file of the latest date

   df = pd.DataFrame()
   df['File'] = files
   
   df['Sensor'] = ['-'.join(os.path.basename(file).split('-')[:1 if 'PSS' in file else 2]) for file in files]
   df['Date'] = ['-'.join(os.path.basename(f).split('-')[-5:]).split('.')[0] for f in files]
   
   df = df.sort_values('Date').drop_duplicates('Sensor',keep='last')
   
   most_recent_files = [f for f in files if f in df['File'].values]
   
   return most_recent_files



def make_pdf_with_plots(files, measurement_config, pdf_name):

  # produces the pdfs with the plots
  
  
  with PdfPages('{}.pdf'.format(pdf_name)) as pdf:
      for key,values in measurement_config.items():
         
         scale_factor = values['units_conversion'] # to scale the parameter to the typical order of magnitude
         
         fig = plt.figure()
         overlay_plots(files, key, values) 
         pdf.savefig(fig)
         

  
  
def build_slack_message(dictionary, sensor):

   # build the message to send to slack channel!
   
   main_message = 'SQC measurement of sensor {} is done!'.format(sensor)
   additional_message = ''
   
   for key, value in dictionary.items():
        if not value:
            additional_message = 'The {} measurement had at least one outlier, please check the plot'.format(key)
            main_message = main_message + "\n" + additional_message
        
   return main_message
    
    
    

def parse_args():

    parser = argparse.ArgumentParser(epilog = 'a path to directory containing ascii files to be analysed')
    parser.add_argument('path')
   # parser.add_argument('sensor')
    return parser.parse_args()




def main():


    args = parse_args()

    filename = glob.glob(f'{args.path}/*/*.txt', recursive=True) 
    
    most_recent_files = find_most_recent_file(filename)
   
  
    IVCV_files = [f for f in most_recent_files if "IVC" in f]
    Stripscan_files = [f for f in most_recent_files if "Str" in f]
    HPK_files = [f for f in filename if "HPK" in f]
    
    
    IV_config = sqc.read_config()['IVCV_Parameters']
  
    Stripscan_config = sqc.read_config()['Strip_Parameters']
    
    if len(IVCV_files)>=1:
         make_pdf_with_plots(IVCV_files, IV_config, "IVCV")
        
    if len(Stripscan_files)>=1:
         make_pdf_with_plots(Stripscan_files, Stripscan_config, "Stripscan")
    
    
    #message = build_slack_message(dictionary_with_flags, args.sensor)
   # sqc.send_slack_message(message)
    
   


if __name__ == "__main__":
    main()
