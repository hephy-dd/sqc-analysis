import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import query_PQC_data_from_DB as db
import seaborn as sns
from analysis_tools import PQC_tools
from pretty_html_table import build_table
from matplotlib.patches import Rectangle





########################################################################################################################################################################################
########################################################################################################################################################################################
#################################################          IV parameters          ######################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

class van_der_pauw(PQC_tools):

   def __init__(self, iv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv




   def run(self):

       
       sheet_resistances = ['pstop_vdp', 'strip_vdp', 'poly_vdp']

       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


       # differentiate dataframes by looping over sheet resistances
       for sheet in sheet_resistances:
    
         
           df_sheet = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters'][sheet]['sql_label'])]
         
           df_sheet = df_sheet.rename(columns={'R_sheet': sheet})
           
           df_sheet = df_sheet[['Halfmoon', sheet, 'Flute', 'Parameter', 'Config', 'Position', 'File']]
           
           df_sheet = PQC_tools.pqc_sequence(self, df_sheet, sheet)
       
           PQC_tools.plot_time_evolution(self, df_sheet, sheet)

        
       
#########       

class linewidth(PQC_tools):



   def __init__(self, iv):

       self.metadata = 'metadata'
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      
       
       linewidth_structures = ['linewidth_strip', 'linewidth_pstop'] 

     
       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])



       for line in linewidth_structures:
             
           df_lwth = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters'][line]['sql_label'])]

           df_lwth = df_lwth.rename(columns={'Linewidth': line})
           

           df_lwth = PQC_tools.pqc_sequence(self, df_lwth, line)
           

           PQC_tools.plot_time_evolution(self, df_lwth, line)



###########

class rcont_strip(PQC_tools):



   def __init__(self, iv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      

      df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])


     
      df_rcont = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['rcont_strip']['sql_label'])]
      
      df_rcont = PQC_tools.pqc_sequence(self, df_rcont, 'rcont_strip')

      PQC_tools.plot_time_evolution(self, df_rcont, 'rcont_strip')


##########

class R_poly(PQC_tools):



   def __init__(self, iv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      

      df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])



 
      df_rpoly = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Rpoly']['sql_label'])]
     
      df_rpoly = PQC_tools.pqc_sequence(self, df_rpoly, 'Rpoly')

      
      PQC_tools.plot_time_evolution(self, df_rpoly, 'Rpoly')


###########

class vdp_bulk(PQC_tools):



   def __init__(self, iv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      


       df = PQC_tools.make_dataframe(self,  self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])
 
             
       df_vdp_bulk = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['vdp_bulk']['sql_label'])]
           

       df_vdp_bulk = PQC_tools.pqc_sequence(self, df_vdp_bulk, 'vdp_bulk')
           
          
       df_vdp_bulk= df_vdp_bulk.loc[df_vdp_bulk['vdp_bulk']<self.config_file['PQC_parameters']['vdp_bulk']['upper']]


       df_vdp_bulk= df_vdp_bulk.loc[df_vdp_bulk['vdp_bulk']>self.config_file['PQC_parameters']['vdp_bulk']['lower']]

       #####
       df_standard = df_vdp_bulk.loc[df_vdp_bulk['Config'].str.contains('Standard')]
       df_standard = df_standard.drop_duplicates(subset=['Halfmoon', 'vdp_bulk'])

       df_perugia = df_standard.loc[df_standard['Location'].str.contains('Perugia')]
  
       df_standard =df_standard.loc[~df_standard['Location'].str.contains('Perugia')]
       #####

       
       df_2 = df_standard.loc[df_standard['Halfmoon'].str.contains('WW')]
    
       PQC_tools.plot_time_evolution(self, df_standard, 'vdp_bulk') #self.config_file['PQC_parameters']['vdp_bulk']['ylabel']


       return df_standard


#############

class I_surf(PQC_tools):



   def __init__(self, iv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])

           
       df_isurf = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Isurf']['sql_label'])]
           
       df_isurf = PQC_tools.pqc_sequence(self, df_isurf, 'Isurf')
                   

       PQC_tools.plot_time_evolution(self, df_isurf, 'Isurf')



############

class S0(PQC_tools):



   def __init__(self, iv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])

             
       df_s0 = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['S0']['sql_label'])]
           

       df_s0 = PQC_tools.pqc_sequence(self, df_s0, 'S0')
                   

       PQC_tools.plot_time_evolution(self, df_s0, 'S0')


###########

class Oxide_bd(PQC_tools):


   def __init__(self, iv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.iv = iv



   def run(self):

      

       df = PQC_tools.make_dataframe(self, self.iv, self.config_file['PQC_tables'][self.iv]['dataframe_headers'])

             
       df_oxide_bd = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Oxide_Vbd']['sql_label'])]
           

       df_oxide_bd = PQC_tools.pqc_sequence(self, df_oxide_bd, 'Oxide_Vbd')
                   

       PQC_tools.plot_time_evolution(self, df_oxide_bd, 'Oxide_Vbd')




############################################################################################################
############################################################################################################
################################## CV ######################################################################

class Vfb(PQC_tools):



   def __init__(self, cv):

     
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])


             
       df_vfb = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Vfb']['sql_label'])]
           

       df_vfb = PQC_tools.pqc_sequence(self, df_vfb,  'Vfb')
           

       PQC_tools.plot_time_evolution(self, df_vfb, 'Vfb')

       return df_vfb


class Nox(PQC_tools):



   def __init__(self, cv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])


             
       df_nox = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Nox']['sql_label'])]
           
       
       df_nox = PQC_tools.pqc_sequence(self, df_nox, 'Nox')
     
       
       PQC_tools.plot_time_evolution(self, df_nox, 'Nox')      
     



     
class Tox(PQC_tools):



   def __init__(self, cv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])

             
       df_tox = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Tox']['sql_label'])]
           

       df_tox = PQC_tools.pqc_sequence(self, df_tox, 'Tox')
           
       
       PQC_tools.plot_time_evolution(self, df_tox, 'Tox')       
 


 
class Dox(PQC_tools):



   def __init__(self, cv):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])

             
       df_dox = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Dox']['sql_label'])]
           

       df_dox = PQC_tools.pqc_sequence(self, df_dox, 'Dox')
           
       
       PQC_tools.plot_time_evolution(self, df_dox, 'Dox')
       
       return df_dox
       
              
       
class Diode_bulk(PQC_tools):



   def __init__(self, cv):

     
       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.cv = cv




   def find_bulk_resistivity(self, df):

       thickness = 290*1e-4
       e0 = 8.85*1e-14
       e_r = 11.68
       mu_h = 450 

       df['Diode_bulk'] = (thickness*thickness)/(2*e0*e_r*mu_h*df['Diode_Vfd']*1000)

       return df

      


   def run(self):

      
       df = PQC_tools.make_dataframe(self, self.cv, self.config_file['PQC_tables'][self.cv]['dataframe_headers'])
      

       df_bulk = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Diode_bulk']['sql_label'])]
       df_bulk = PQC_tools.pqc_sequence(self, df_bulk, 'Diode_bulk')
       
        

       df_bulk = df_bulk.drop_duplicates(subset=['Halfmoon', 'Diode_bulk'])
       
       df_perugia = df_bulk.loc[df_bulk['Location'].str.contains('Perugia')]
       df_diode_bulk2 = df_bulk.loc[~df_bulk['Location'].str.contains('Perugia')]
       
   

       PQC_tools.plot_time_evolution(self, df_diode_bulk2, 'Diode_bulk') 
       
       
       return df_diode_bulk2
       


  

###########################################################################################################
###########################################################################################################
########################## FET ############################################################################

  
            
class Vth(PQC_tools):



   def __init__(self, fet):

       self.config_file = db.read_yml_configuration('PQC_parameters_DB.yml')
       self.fet = fet



   def run(self):

      


       df = PQC_tools.make_dataframe(self, self.fet, self.config_file['PQC_tables'][self.fet]['dataframe_headers'])

            
       df_fet = df.loc[df['Parameter'].str.contains(self.config_file['PQC_parameters']['Vth']['sql_label'])]
           

       df_fet = PQC_tools.pqc_sequence(self, df_fet, 'Vth')
           
       
       PQC_tools.plot_time_evolution(self, df_fet, 'Vth')   
        
       return df_fet 

#############################################################################################################################################################################################
#############################################################################################################################################################################################
######################################################################


def run_PQC_sequence():
  class_dictionary = { 'IV': ['van_der_pauw', 'linewidth', 'I_surf', 'S0', 'vdp_bulk', 'R_poly', 'rcont_strip', 'Oxide_bd'], 'CV': ['Vfb', 'Nox', 'Tox', 'Dox', 'Diode_bulk'], 'FET': ['Vth']} 
  df_list=[]


  for  class_ in class_dictionary.keys():
    for structure in class_dictionary[class_]:
          df = pd.DataFrame()
          v = globals()[structure](str(class_))
     
          df = v.run()
          print('The analysis of {} data is successful'.format(structure))
          print('######################################')
        
          
        
if __name__ == "__main__":
  run_PQC_sequence()
