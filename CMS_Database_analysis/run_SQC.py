import numpy as np
import pandas as pd
from SQC_analysis_db import IV, CV, strip_parameter, bad_Strips, overview
import os
import seaborn as sns
import query_data_from_DB as db
import matplotlib.pyplot as plt
from pretty_html_table import build_table
from matplotlib.patches import Rectangle
import argparse

parameters = ['IV', 'CV', 'Istrip', 'Rpoly', 'Cac', 'Idiel', 'Cint', 'Rint'] 

try:
    os.makedirs('Figures/SQC')
except FileExistsError:
    print('Directory Figures/SQC already exists')


try:
    os.makedirs('html_tables/SQC')

except FileExistsError:
    print('Directory html_tables/SQC/ already exists')


##########################################################################################################################


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('expert')
    return parser.parse_args()



def run_SQC_analysis():

 #args = parse_args()


 for i in parameters:

    if i=='IV':
        p=IV(i)
        df_iv = p.run()
      
        
        print('{} analysis is done'.format(i))
        print('##############################################################')

    elif i=='CV':
        p = CV(i)
        df_cv = p.run()
    
        print('{} analysis is done'.format(i))
        print('##############################################################')

    else:
        p = strip_parameter(i)
        df_strip = p.run()
    
        print('{} analysis is done'.format(i))
        print('###############################################################')

 v = bad_Strips()
 bad_strips_df = v.run()

 bad_summary = {'bad_strips': bad_strips_df}

 n = overview()
 overview_df = n.run()
 


if __name__ == "__main__":
    
    run_SQC_analysis()