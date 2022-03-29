import subprocess
import query_data_from_DB as db
import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pretty_html_table import build_table
import warnings
from scipy.interpolate import CubicSpline

pd.options.mode.chained_assignment = None


class runs:

    def __init__(self):

        pass


    def make_runs_list(self):

        with open('data_old/runs.json') as f:

            runs_list = json.load(f)

        splt = []
        for j in runs_list:

            splt.append(j.split(','))

        return splt



    def runs_dataframe(self):

      runs_list = self.make_runs_list()
      runs_df = pd.DataFrame(runs_list, columns= ['Location', 'Type', 'Run_number'])

      runs_df = runs_df.loc[runs_df['Type']!='PQC']
      runs_df = runs_df.drop_duplicates(subset=['Run_number', 'Location'], keep='first')
 

      return runs_df


    
##################################################################################################################
##################################################################################################################
#########################                 IV                ######################################################
##################################################################################################################
##################################################################################################################




class IV(runs):


    def __init__(self, iv):

        self.iv = iv
        
        self.config_file = db.read_yml_configuration('SQC_parameters_DB.yml')

        

    def make_list(self):

        ## opens and reads the json file and creates a list of data

        with open('data_old/IV.json') as f:

            iv_list = json.load(f)

        
        splt =[]
        for j in iv_list:

            splt.append(j.split(','))

        return splt


        
    def make_dataframe(self):
       
        ## constructs a pandas dataframe from the json data 

        lista = self.make_list()
        df = pd.DataFrame(lista, columns = ['Sensor', 'Volts', self.iv, 'Temp', 'Run_number'])
        df = df.loc[~df['Sensor'].str.contains('HPK')]
        df = df.loc[~df['Sensor'].str.contains('33234')]
        df = df.drop_duplicates(subset=['Run_number', self.iv], keep='first')
    
        return df


    def filter_dataframe(self, df, x1, x2):

        ## filters the dataframe and keeps only data which correspond to the given voltage
         
        df['Volts']= pd.to_numeric(df['Volts']).abs()
    
        
        df = df.loc[(df['Volts']<float(x2)) & (df['Volts']>float(x1))]

        df['IV'] = pd.to_numeric(df['IV']).abs()


        return df



    def find_sensor_type(self, df):

        ## extracts the type of the sensor (2-S, PSS, PSP) from the barcode label 

        df['Sensor_type'] = np.nan
        df['Sensor_type'] = df['Sensor_type'].mask(df['Sensor'].str.contains('2-S'), '2-S').mask(df['Sensor'].str.contains('PSS'), 'PSS').mask(df['Sensor'].str.contains('PSP'), 'PSP')

    

        return df


    def find_batch(self, df):

        ## extracts the batch number from the barcode label

        df['Batch'] = np.nan

        df['Batch'] = (df['Sensor'].str.split('_')).str[0]
        df['Batch'] = pd.to_numeric(df['Batch'])

        df = df.sort_values(by=['Batch'])
        df['Batch'] = df['Batch'].astype(str)

        return df



    def scale_current(self, df):
       
        ##scales the current to 21oC

        df['Temp'] = pd.to_numeric(df['Temp'])

        df['Temp']  = df['Temp'].mask(df['Temp'].isnull()==True, float(25)) # assigns 25oC to HPK data

    
        df['I_scaled'] = (df['IV']*21)/df['Temp']

        return df



    def make_table_with_outliers(self, df):


        outliers_df = df.loc[df['IV']>self.config_file['SQC_parameters']['IV']['Limit']]

    


        return outliers_df





    def plot_i600_over_time(self, df):

        ## plots the scaled current at -600V over batch number

        batch = list(dict.fromkeys(df['Batch']))

        fig, ax = plt.subplots()
        sns.scatterplot(data = df, x='Batch', y='I_scaled', hue='Sensor_type', s=25, linewidth=0, ax=ax)
    
        plt.xticks(batch, batch, rotation=90, va='top', fontsize = 6)
        plt.locator_params(axis='x', nbins=len(batch)/4)
        ax.set(xlabel=None)

        plt.axvline(x=22, linestyle='dashed', color='black')
        plt.text(26, 1000, 'Production Period', style ='italic')
        plt.yscale('log')

        plt.figtext(0.5, 0.01, r'Time evolution $\longrightarrow$', fontsize=10, fontweight='bold')
        ax.set_ylabel('Total Current@-600V [nA]', fontsize = 10, fontweight='bold')

        plt.legend(loc='best')
        #plt.show()
        plt.savefig('figures/i600_type.png')

                        


    
    def run(self):

        self.df = self.make_dataframe()
        i600_df = self.filter_dataframe(self.df, 599, 601)
        df_final = self.find_sensor_type(i600_df)    
        df_final = self.find_batch(df_final)
        df_final = self.scale_current(df_final)
        
        self.make_table_with_outliers(df_final)

        self.plot_i600_over_time(df_final)
        
        return df_final


##############################################################################################################################################
##############################################################################################################################################
###############################################                CV                #############################################################
##############################################################################################################################################
##############################################################################################################################################


class CV(runs):


    def __init__(self, cv):

        self.cv = cv
        
        self.config_file = db.read_yml_configuration('SQC_parameters_DB.yml')

        

    def make_list(self):

        ## opens and reads the json file and creates a list of data

        with open('data/CV.json') as f:

            cv_list = json.load(f)

        
        splt =[]
        for j in cv_list:

            splt.append(j.split(','))

        return splt


        



    def analyse_cv(self, v, c, cut_param=0.004, debug=False):

      # init
      v_dep2 = -1

      c = [1/i**2 for i in c]
      print(c)
      #c = [1/c**2 for i in c]

      #print(v)
      #print(c)

      # get spline fit, requires strictlty increasing array
      y_norm = c/np.max(c)
      x_norm = np.arange(len(y_norm))
      spl = CubicSpline(x_norm, y_norm)
      spl_dev = spl(x_norm, 1)


      # get regions for indexing
      idx_rise = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) > cut_param) ]
      idx_const = [ i for i in range(len(spl_dev)) if (abs(spl_dev[i]) < cut_param) ]

      with warnings.catch_warnings():
         warnings.filterwarnings('error')

         try:
            v_rise = v[ idx_rise[-6]:idx_rise[-1]+1 ]
            v_const = v[ idx_const[1]:idx_const[-1]+1 ]
            c_rise = c[ idx_rise[-6]:idx_rise[-1]+1 ]
            c_const = c[ idx_const[1]:idx_const[-1]+1 ]

            # line fits to each region
            a_rise, b_rise = np.polyfit(v_rise, c_rise, 1)
            a_const, b_const = np.polyfit(v_const, c_const, 1)

            # full depletion via intersection
            v_dep2 = (b_const - b_rise) / (a_rise - a_const)


         except (ValueError, TypeError, IndexError):

            print("The array seems empty. Try changing the cut_param parameter.")

      return  v_dep2





    def make_dataframe(self):
       
        ## constructs a pandas dataframe from the json data 

        lista = self.make_list()
        
        
        df = pd.DataFrame(lista, columns = ['Sensor', 'Volts', self.cv, 'Run_number'])
        df = df.loc[~df['Sensor'].str.contains('HPK')]
        df = df.loc[~df['Sensor'].str.contains('33234')]
        df = df.drop_duplicates(subset=['Run_number', self.cv], keep='first')
        df = df.dropna(subset=[self.cv]) 
        df['Volts'] = pd.to_numeric(df['Volts']).abs()
       # df = df.loc[df['Volts']>0]
        try:
            df[self.cv] = pd.to_numeric(df[self.cv], errors='coerce')
        except Exception as err:
            print(err)

    
        df = df.groupby('Run_number')
        dataframe = [group for _, group in df]
        for i in dataframe:
           
            vfd = self.analyse_cv(i['Volts'].values, i[self.cv].values)

            print(vfd)
        #df['Vfd'] = list(map(self.analyse_cv, df['Volts'], df[self.cv]))
       
    
        #volts = df['Volts'].values
        #capacitance = df['iv.cv'].values

       # for num,val in enumerate(capacitance):
        #    vfd = self.analyse_cv(

       # return df




########################################################################################################################################################################################
########################################################################################################################################################################################
##################################################                Strip parameters               #######################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################




class strip_parameter(runs):



  def __init__(self, parameter):

       self.parameter = parameter
       self.config_file = db.read_yml_configuration('SQC_parameters_DB.yml')




  def make_list(self):


       with open('data_old/{}.json'.format(self.parameter)) as f:

            parameter_list = json.load(f)

        
       split_data =[]
       for j in parameter_list:

            split_data.append(j.split(','))

       return split_data



  def make_dataframe(self):

      parameter_list = self.make_list()

      df = pd.DataFrame(parameter_list, columns = ['Sensor', 'Strip', self.parameter, 'Run_number'])
      df = df.loc[~df['Sensor'].str.contains('HPK')]
      df = df.loc[~df['Sensor'].str.contains('33234')]

      
              
      return df



  def find_sensor_type(self, df):

        ## extracts the type of the sensor (2-S, PSS, PSP) from the barcode label 

        df['Sensor_type'] = np.nan
        df['Sensor_type'] = df['Sensor_type'].mask(df['Sensor'].str.contains('2-S'), '2-S').mask(df['Sensor'].str.contains('PSS'), 'PSS').mask(df['Sensor'].str.contains('PSP'), 'PSP')

        return df




  def find_batch(self, df):

        ## extracts the batch number from the barcode label

        df['Batch'] = np.nan

        df['Batch'] = (df['Sensor'].str.split('_')).str[0]
        df['Batch'] = pd.to_numeric(df['Batch'])

        df = df.sort_values(by=['Batch'])
        df['Batch'] = df['Batch'].astype(str)

        return df




  def find_bad_strip(self, df):
      
    #df = self.normalize_data(df)
    
      
    list_larger_than_condition = ['Cac', 'Rint']
            
    if self.parameter in list_larger_than_condition:
          bad_strips_df = df.loc[df[self.parameter]<self.config_file['SQC_parameters'][self.parameter]['Limit']]
     
    elif self.parameter=='Rpoly':
         
          bad_strips_df = df.loc[~df[self.parameter].between(self.config_file['SQC_parameters'][self.parameter]['Limit']-0.5, self.config_file['SQC_parameters'][self.parameter]['Limit']+0.5)]

    else:

          bad_strips_df = df.loc[df[self.parameter]>self.config_file['SQC_parameters'][self.parameter]['Limit']]

 
    
    
    if not bad_strips_df.empty:
          
          html_table_green_light = build_table(bad_strips_df, 'green_light', text_align='center')
          with open('html_tables/{}_table_with_strips_outside_specs.html'.format(self.parameter), 'w') as f:
                 f.write('<html><body> <h1>Strips outside of specifications <font color = #4000FF></font></h1>\n</body></html>')
                 f.write('\n')
                 f.write(html_table_green_light)
        







  def dataframe_with_median(self, df):

      df[self.parameter] = pd.to_numeric(df[self.parameter]).abs()
      
      
      df = df.groupby(['Sensor', 'Run_number', 'Batch', 'Sensor_type', 'Location']).agg(parameter=(self.parameter, 'median'))
      df = df.reset_index()
      df = df.sort_values(by=['Batch'], ascending=True)
    
      df= df.rename(columns={'parameter': self.parameter})

      
      return df



  def find_location(self, df):

     runs_df = self.runs_dataframe()

     merged_df = pd.merge(df, runs_df, how='outer', on=['Run_number'])
     merged_df = merged_df.dropna(subset=['Sensor', 'Run_number'])
     #pd.set_option('display.max_rows', merged_df.shape[0] +1)
     

     return merged_df




  def normalize_data(self, df):

   
     
     df[self.parameter] = pd.to_numeric(df[self.parameter])

     df[self.parameter] = np.where(df['Sensor_type']=='2-S', df[self.parameter]/self.config_file['SQC_parameters'][self.parameter]['factor_2-S'], df[self.parameter])

     df[self.parameter] = np.where(df['Sensor_type']=='PSS', df[self.parameter]/self.config_file['SQC_parameters'][self.parameter]['factor_PSS'], df[self.parameter])

     if self.parameter=='Idiel':

        df =  self.correct_idiel_only_hephy_data(df)
   
     return df



  def correct_idiel_only_hephy_data(self, df):

      
      wrong_idiel = ['35953', '35715', '35717', '35718']

      df[self.parameter] = df[self.parameter].mask(df['Batch'].apply(lambda x: x in wrong_idiel), df[self.parameter]/1000)
      
      
      return df






  def make_html_table_with_outliers(self, df):

      
      list_larger_than_condition = ['Cac', 'Rint']
            
      if self.parameter in list_larger_than_condition:
          outliers_df = df.loc[df[self.parameter]<self.config_file['SQC_parameters'][self.parameter]['Limit']]
     
      elif self.parameter=='Rpoly':
         
          outliers_df = df.loc[~df[self.parameter].between(self.config_file['SQC_parameters'][self.parameter]['Limit']-0.5, self.config_file['SQC_parameters'][self.parameter]['Limit']+0.5)]

      else:

          outliers_df = df.loc[df[self.parameter]>self.config_file['SQC_parameters'][self.parameter]['Limit']]



      if not outliers_df.empty:
          
          html_table_blue_light = build_table(outliers_df, 'blue_light', text_align='center')
          with open('html_tables/{}_table_with_sensors_outside_specs.html'.format(self.parameter), 'w') as f:
                 f.write('<html><body> <h1>Sensors outside of specifications <font color = #4000FF></font></h1>\n</body></html>')
                 f.write('\n')
                 f.write(html_table_blue_light)
        
         



  def plot_distribution(self, df):

    fig, ax = plt.subplots()

    plt.hist(df[self.parameter], bins=30, color='blue', edgecolor='black', linewidth=1.2)
    ax.set_xlabel('{}'.format(self.config_file['SQC_parameters'][self.parameter]['label']), fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of sensors', fontsize= 10, fontweight='bold')
    
    plt.savefig('figures/{}.png'.format(self.parameter))




  def run(self):

      attributes_list = [self.find_sensor_type, self.find_batch, self.find_location, self.normalize_data]
    
      df = self.make_dataframe()
      #df = self.find_sensor_type(df)
      #df = self.find_batch(df)
      
      for attr in attributes_list:
          
          df = attr(df)
    

      self.find_bad_strip(df)
      df = self.dataframe_with_median(df)

     

      print(df) 
      self.make_html_table_with_outliers(df)
      self.plot_distribution(df)






try:
    os.mkdir("figures")

except FileExistsError:
    print("Directory figures/ already exists")


try: 
    os.mkdir("html_tables")

except FileExistsError:
    print("Directory html_tables/ already exists")




parameters =  ['IV','Istrip', 'Rpoly', 'Cac', 'Idiel', 'Cint', 'Rint']

for i in parameters:

  if i=='IV':
     p = IV(i)
     p.run()
     print('###############################################')
  elif i=='CV':
     p = CV(i)
     p.make_dataframe()
     print('###############################################')
  else:
     p = strip_parameter(i)
     p.run()
     print('##############################################')

