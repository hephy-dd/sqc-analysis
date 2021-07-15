import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import glob
from HPK_data_analysis import make_dictionary_with_currents, find_date
from datetime import datetime


def find_sensor_ID(files):

    for file in files:

        if 'PSS' in file:
            id = 'PSS'
        elif '2-S' in file:
            id = '2-S'
        else:
            id = 'PS-P'

    return id

def clear_nan(array):

   array = [ x for x in array if not np.isnan(x)]
   return array



def MAD(list, list_median):


     list_mad = np.median(np.abs(((list - list_median))))
     return list_mad


def make_plot(df):

    fig, ax = plt.subplots()
    index = 0
    colors = ['blue', 'red', 'green']
    for key, group in df.groupby(['ID']):
        ax = group.plot(ax=ax, kind='scatter', x='Date', y='I600', yerr='Stdev', c=colors[index], label=key)
        index += 1

    plt.grid()
    plt.title("Time evolution of I600 - HPK data")
    plt.axvline(x=datetime(2020, 12, 1), color='black', label='Production Period')
    plt.ylabel('Median Current@600V [nA]')
    plt.tick_params(axis='x', labelsize=7)
    plt.xticks(rotation=90)
    plt.legend(loc='best')
    plt.show()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    return parser.parse_args()


def main():

        dates=[]
        i600 =[]
        id_list=[]
        error = []
        args = parse_args()

        for subdirs, dirs, files in os.walk(args.path):

            filepath = subdirs + os.sep
            glob_files = glob.glob(filepath + '*.txt')


            if len(files)>2:

              sensor_id = find_sensor_ID(glob_files)
              id_list.append(sensor_id)
              date = str(find_date(glob_files[2]))
              date = str(date.split('\n')[0])
              dates.append(datetime.strptime(date, "%d-%b-%y"))


              i_dict, batch, total_bad_strips, ratio_list, i600_list = make_dictionary_with_currents(glob_files)
              i600_list = clear_nan(i600_list)
              i600.append(np.median(i600_list))

              error.append(MAD(i600_list, np.median(i600_list)))

        temp_dictionary = {'Date': dates, 'I600': i600, 'Stdev': error, 'ID': id_list}
        print(i600)
        print(id_list)
        print(error)
        print(dates)
        df = pd.DataFrame(temp_dictionary, columns = ['Date', 'I600', 'Stdev', 'ID'])
        df = df.sort_values(by='Date')

        make_plot(df)





if __name__ == "__main__":
        main()