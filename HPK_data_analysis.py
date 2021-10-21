import pandas as pd
import os
import sys
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings


colors = ['red', 'gold', 'darkgreen', 'deepskyblue', 'black', 'blue', 'm', 'grey', 'coral', 'violet',
          'salmon', 'saddlebrown', 'lime', 'cyan']

marker_style = ['o', '^', '*', '+', 'x', '.']


def make_dataframe_from_ascii(datafile, skip, max_row, quantity):

    data = np.genfromtxt(datafile, skip_header=skip, max_rows=max_row)
    df = pd.DataFrame(data, columns= ['Voltage', quantity])
    return df


def find_date(datafile):
    
    f = open(datafile)
    lines = f.readlines()
    date = lines[6][6:]

    return date

def find_avg_Rpoly(datafile):

    if not 'PSP_' in datafile:
        f = open(datafile, encoding='Latin-1')
        lines = f.readlines()
       
        rpoly = lines[17][44:48]
        if len(rpoly)>1:
          return rpoly
        else:
          return 0
    else:
        return 0


def find_batch_number(dirs):
    
   for d in dirs:
        print(dirs)
        #the following process is based on HPK standard ascii file names. If the latter changes then we need to modify the lines below

        #batch_1 = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(d)))[0].split('_')[2:3])
        #batch_2 = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(d)))[0].split('_')[4:])
        #batch = batch_1 + '_' + batch_2  # this is actually the batch number
        
        return d


def find_bad_strips(datafile):

    f = open(datafile)
    lines = f.readlines()
    bad_strips = lines[8][12] # finds the string which corresponds to # of bad strips and assigns it to a parameter

    if bad_strips=='-':
        bad_strips=0

    return bad_strips


def check_current(df):

    i600 = df['Current'].loc[df['Voltage'] == 600].values[0]
    i800 = df['Current'].loc[df['Voltage'] == 800].values[0]
    i1000 = df['Current'].loc[df['Voltage'] == 1000].values[0]
    if not np.isnan(i800):
        ratio = i800/i600
    else:
        ratio = 0

    #returns the currents in the scale of nA
    return i600*1e9, i800*1e9, i1000*1e9, ratio



def plot_distribution(data, xlabel, batch):

    plt.hist(data, bins=50, label = batch)
    plt.xlabel('{} '.format(xlabel))
    plt.ylabel('Number of Sensors')
    plt.title("Distribution of {}".format(xlabel))
    plt.legend()



def plot_scatter(x, y, color, label, title, bad_strips, sensors_reached_compliance, sensors_with_large_ratio):
    plt.scatter(x,y, color=color, label = label)
    plt.title("Batch {} HPK measured currents".format(title))
    plt.ylabel('Current [nA]')
    plt.tick_params(axis='x', labelsize=7)
    plt.xticks(rotation=90, ha='right')
    plt.legend(loc='best', fontsize=7)
    annotate = 'Total bad strips : {} \nSensors with breakdown: {} \nSensors with i800/i600 beyond the limit: {} '.format(bad_strips, sensors_reached_compliance, sensors_with_large_ratio)
    plt.annotate(annotate, (0.25, 0.65), xycoords='figure fraction', color='black')


def plot_vfd(x, y, title):

    plt.scatter(x, y, color='red')
    plt.title("Batch {} Extracted Vfd from HPK data".format(title))
    plt.ylabel('Full Depletion Voltage [V]')
    plt.tick_params(axis='x', labelsize=7)
    plt.xticks(rotation=90, ha='right')
    

def plot_graph(x, y, yaxis, batch, marker, label, color):


    plt.plot(x, y, marker=marker, color = color, markersize=5, label=label)
    plt.title('HPK data Batch {}'.format(batch), fontname="Times New Roman", fontsize=16, fontweight='bold')
    plt.xlabel('Voltage [V]', fontsize=12)
    plt.ylabel('{}'.format(yaxis), fontsize=12)
    #plt.yscale('log')
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=12)
    plt.legend(loc='best', fontsize=5, ncol=2)


def color_map():
    #Creates a list of colors which can be used in a plot with overlaid curves
    # not used at this script, but can be useful if you want to analyse the HPK data

    colors = []

    evenly_spaced_interval = np.linspace(0, 1, 50)
    for x in evenly_spaced_interval:

           colors.append(plt.cm.Set1(x))

    return colors


def make_dictionary_with_currents(files):
    # This function takes a list of ascii files as an input and generates
    # a dictionary which contains the HPK currents @600,800,1000V

    i_dict = {}
    batch = 0
    total_bad_strips=0
    ratio_dict={}
    i600_list =[]

    for f in files:

        #the following process is based on HPK standard ascii file names. If the latter changes then we need to modify the lines below
        
        batch_1 = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(f)))[0].split('_')[2:3])
        batch_2 = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(f)))[0].split('_')[4:])
        
        batch = batch_1 + '_' + batch_2  # this is actually the batch number
        sensor = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(f)))[0].split('_')[3:4]) # this is the sensor ID (just the number for visualisation purposes)

        try:
            df = make_dataframe_from_ascii(f, 23, 51, 'Current')
            i600, i800, i1000, ratio = check_current(df)

            ratio_dict.update({sensor: ratio})
            i_dict.update({sensor: [i600, i800, i1000]})
            total_bad_strips += int(find_bad_strips(f))

            i600_list.append(i600)



        except Exception as e:
            print(e)

    i_dict = dict(sorted(i_dict.items()))

    return i_dict, batch, total_bad_strips, ratio_dict, i600_list



def plot_IVCV(files):
    # This function takes a list of ascii files as an input and generates
    # a dataframe which conta

    batch = index = marker_index = 0
    capacitance_dict={}
    current_dict = {}
    vfd_dict = {}


    for f in files:

       
        #the following process is based on HPK standard ascii file names. If the latter changes then we need to modify the lines below
        os.path.basename(os.path.normpath(f))
        batch_1 = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(f)))[0].split('_')[2:3])
        batch_2 = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(f)))[0].split('_')[4:])
        
        batch = batch_1 + '_' + batch_2  # this is actually the batch ID
        sensor = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(f)))[0].split('_')[3:4]) # this is the sensor ID (just the number for visualisation purposes)

        try:
            df1 = make_dataframe_from_ascii(f, 23, 51, 'Current')
            df2 = make_dataframe_from_ascii(f, 76, 40, 'Capacitance')

            current_dict.update({sensor: (df1['Current']*1e9).values})
            capacitance_dict.update({sensor: (1/df2['Capacitance']**2).values})

            current_dict = dict(sorted(current_dict.items()))
            capacitance_dict = dict(sorted(capacitance_dict.items()))

            iv_voltages = df1['Voltage'].values
            cv_voltages= df2['Voltage'].values
            
            v_fd = analyse_cv(cv_voltages, (1/df2['Capacitance']**2).values)
            vfd_dict.update({sensor: v_fd})

        except Exception as e:
            print(e)

    for sensor,current in current_dict.items():

         plot_graph(iv_voltages, current, 'Current [nA]', batch, marker_style[marker_index],  color= colors[index], label=sensor)
         index += 1
         if index==len(colors):
             index = 0
             marker_index +=1

    plt.savefig("IV_{}.png".format(batch))

    index = 0
    marker_index = 0
    plt.clf()

    for sensor,capacitance in capacitance_dict.items():

         plot_graph(cv_voltages, capacitance, '1/C$^2$ [F$^{-2}$]', batch, marker_style[marker_index],  color= colors[index], label=sensor)
         index += 1
         if index==len(colors):
             index = 0
             marker_index +=1

    plt.savefig("CV_{}.png".format(batch))
    
    plt.clf()
    
    for sensor, vfd in vfd_dict.items():

        plot_vfd(sensor, vfd, batch)

    plt.savefig("Vfd_{}.png".format(batch))



def plot_currents_per_batch(i_dict, batch, total_bad_strips, ratio_dict):

    i6 = [i[0] for i in i_dict.values()]
    i8 = [i[1] for i in i_dict.values()]
    i10 = [i[2] for i in i_dict.values()]
    sensors_reached_compliance = find_compliance(i_dict)
    sensors_with_large_ratio = find_sensors_with_large_ratio(ratio_dict)
    plot_scatter(i_dict.keys(), i6, 'red', 'I@600V', batch,total_bad_strips, sensors_reached_compliance, sensors_with_large_ratio)
    plot_scatter(i_dict.keys(), i8, 'blue', 'I@800V', batch,total_bad_strips, sensors_reached_compliance, sensors_with_large_ratio)
    plot_scatter(i_dict.keys(), i10, 'green', 'I@1000V', batch, total_bad_strips, sensors_reached_compliance, sensors_with_large_ratio)
    # plot_distribution(i600_list, 'Current@600V')
   
    plt.savefig(batch + '.png')


def find_sensors_with_large_ratio(ratio_dict):

  list_with_sensors = []
  for sensor, ratio in ratio_dict.items():
        
        if ratio>=2.5:
           list_with_sensors.append(sensor)
           
  return list_with_sensors
  


def find_compliance(i_dict):

    sensors_reached_compliance = []
    for key, value in i_dict.items():
        for i in value:
           if np.isnan(i):
            
               sensors_reached_compliance.append(key)

    return sensors_reached_compliance




def analyse_cv(v, c, cut_param=0.004, debug=False):

    # init
    v_dep2 = -1



    # get spline fit, requires strictlty increasing array
    y_norm = c / np.max(c)
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
    
    
def do_the_plots(files):

  plot_IVCV(files)
  plt.clf()

  i_dict, batch, total_bad_strips, ratio_dict, i600_list = make_dictionary_with_currents(files)
  plot_currents_per_batch(i_dict, batch, total_bad_strips, ratio_dict)
  i_dict.clear()
  plt.clf()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    return parser.parse_args()




def main():

    args = parse_args()
    


    for subdirs, dirs, files in os.walk(args.path):
     
      
      if len(dirs)>1:
         for dir in dirs:
            
            path = args.path  + os.sep + dir 
            
            txt_files = glob.glob(path + os.sep +  '**' + os.sep + '*.txt', recursive=True)
           
            right_files = []
            left_files =[]
            
            if len(txt_files)>1: # trick to skip the empty files that are generated in the PS-s/PS-p case
               if '2-S' in txt_files[0]:
                  do_the_plots(txt_files)
             
               else: 
                 for f in txt_files:
                  
                    if 'MAINL' in f:
                     left_files.append(f)
                    else:
                     right_files.append(f)
                  
                 do_the_plots(left_files)
                 do_the_plots(right_files)
                
           



      else:
          if len(subdirs)==len(args.path):
           # this condition verifies that there is no extra run for the lines below; this can happen due to subdirs itteration

            path = args.path
           
            txt_files = glob.glob(path + os.sep + dir + os.sep  + '**' + os.sep + '*.txt', recursive=True)
            do_the_plots(txt_files)





##### To run the script use the command: python HPK_data.py path/folder_name

if __name__ == "__main__":
   main()
