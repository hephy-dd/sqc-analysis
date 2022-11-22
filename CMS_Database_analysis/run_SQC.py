import numpy as np
import pandas as pd
from SQC_analysis_db import IV, CV, strip_parameter, bad_Strips, SQC_summary
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pretty_html_table import build_table
from pandasgui import show
import argparse

parameters = ['CV', 'Istrip', 'Rpoly', 'Cac', 'Idiel', 'Cint', 'Rint'] # 'IV', 'CV',

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
 dataset = {}

 for i in parameters:

    if i=='IV':
        p=IV(i)
        df_iv = p.run()
        data_iv = {i: df_iv}
        dataset.update(data_iv)
        
        n = SQC_summary(df_iv)
        summary_list_df = n.run()
        for en,df in enumerate(summary_list_df):
             data_summary = {'summary_{}'.format(en): df}
             dataset.update(data_summary)
        
        print('{} analysis is done'.format(i))
        print('##############################################################')

    elif i=='CV':
        p = CV(i)
        df_cv = p.run()
        data_cv = {i: df_cv}
        dataset.update(data_cv)
        
        print('{} analysis is done'.format(i))
        print('##############################################################')

    else:
        p = strip_parameter(i)
        df_strip = p.run()
        data_strip = {i: df_strip}
        dataset.update(data_strip)
        
        print('{} analysis is done'.format(i))
        print('###############################################################')

 v = bad_Strips()
 bad_df = v.run()

 bad_summary = {'bad_strips': bad_df}
 dataset.update(bad_summary)

 #if args.expert == 'expert':
  #   show(**dataset)



if __name__ == "__main__":
    
    run_SQC_analysis()