import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings
from statistics import mean, stdev, median
import glob
import yaml


headers_Stripscan = ['Pad', 'Istrip', 'Rpoly', 'Cac', 'Cac_Rp', 'Idiel', 'Cint', 'Cint_Rp', 'Idark', 'Rint', 'Temperature','Humidity']

headers_IVC = ['Voltage [V]',  'current [A]' , 'temperature [deg]', 'humidity [%]'] #'capacitance [F]',

headers_HPK = ['Voltage [V]',  'current [A]']

headers_HPK_cv = ['Voltage [V]',  'capacitance [F]']


def read_config():
    with open('SQC_parameters.yml', 'r') as f:
        conf = yaml.safe_load(f)#, Loader=yaml.FullLoader)

    return conf



def get_parameter(filename, parameter, clear):

    if '2-S' in filename:
        sensor_id = '2-S'
    else:
        sensor_id = 'PSS'

    dict = make_dictionary(parameter, filename, sensor_id)

    return dict

def make_dictionary(main_parameter, file, prefix):

    ### main_parameter: the parameter for which we will generate the xml file
    ### config: the configuration data
    ### file: the txt file to be analysed
    ### headers: the headers of the txt file data
    ### prefix: IVC or Str

    dict_with_values = {}
    config = read_config()
    prefix = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(file)))[0].split('_')[0:1])


    headers = config['headers']['IV' if prefix == 'IVC' else 'Str']


    if 'Str' in file:

         a= convert_txt_to_df(file, headers, 23 if '2-S' in file else 16)

         df = a[a[main_parameter].notnull()]

         for secondary_parameter in config['Strip_Parameters'][main_parameter]['variables']:
            values = df[secondary_parameter].values

            if secondary_parameter in config['Strip_Parameters']:

              unit = float(config['Strip_Parameters'][secondary_parameter]['units_conversion'])
              values = [i*unit for i in values] #convert the list with data in the required from DB units

            dict_with_values[secondary_parameter] = values



    elif 'IVC' in file:
         a = convert_txt_to_df(file, headers, 9)
         for secondary_parameter in config['IVCV_Parameters'][main_parameter]['variables']:

           if main_parameter =='capacitance':
             df = a[a[main_parameter].notnull()]
             values = df[secondary_parameter].values
           else:
             values = a[secondary_parameter].values

           if secondary_parameter in config['IVCV_Parameters']:

             unit = float(config['IVCV_Parameters'][secondary_parameter]['units_conversion'])
             values = [i * unit for i in values]  #convert the list with data in the required from DB units

           dict_with_values[secondary_parameter] = values

    return dict_with_values



def convert_txt_to_df(filename, headers, skip):

    if 'HPK_' in filename:
        dat = np.genfromtxt(filename, skip_header=skip, max_rows=51)

    else:

        dat = np.genfromtxt(filename, skip_header= skip)

    df = pd.DataFrame(dat, columns = headers)

    return df

def make_Dataframe_Stripscan(parameter, filename, sensor_id, clear=False):


    if clear:
       a= convert_txt_to_df(filename, headers_Stripscan, 23 if sensor_id=='2-S' else 16) #16
       a = a[a[parameter].notnull()]
       par = a[parameter]
       pad = a['Pad']

    else:
       df = convert_txt_to_df(filename, headers_Stripscan, 23 if sensor_id=='2-S' else 16) #16

    if parameter is '':
        return df
    else:
        return pad, par



def make_Dataframe_IVCV(parameter, filename, clear=False):

    if clear:
        a = convert_txt_to_df(filename, headers_IVC, 9) #9 23 77
        df = a.dropna()
    else:
        df = convert_txt_to_df(filename, headers_IVC, 9)#9

    if parameter is '':
        return df
    else:
        return df[parameter]



def make_Dataframe_HPK_IV(parameter, filename, clear=False):

    if clear:
        a = convert_txt_to_df(filename, headers_HPK, 23)
        df = a.dropna()
    else:
        df = convert_txt_to_df(filename, headers_HPK, 23)

    if parameter is '':
        return df
    else:
        return df[parameter]

def make_Dataframe_HPK_CV(parameter, filename, clear=False):

    if clear:
        a = convert_txt_to_df(filename, headers_HPK_cv, 76)
        df = a.dropna()
    else:
        df = convert_txt_to_df(filename, headers_HPK_cv, 76)

    if parameter is '':
        return df
    else:
        return df[parameter]



def MAD(parameter, parameter_median):

    parameter_mad = np.median(np.abs(((parameter - parameter_median))))
    return parameter_mad


def numpy_MAD(parameter, parameter_median):

    parameter_mad = np.median(np.abs(parameter - parameter_median))
    return parameter_mad



def assign_label(file):

    file = os.path.splitext(file)[0]

    lbl = '_'.join(file.split('_')[1:6])
    batch = '_'.join(file.split('_')[1:2])
    return lbl, batch

'''
def convert_txt_to_df(filename, parameter, skip, HPK_file):

    if HPK_file:
        dat = np.genfromtxt(filename, skip_header=skip, max_rows=51) #51
        print(dat)
    else:
         dat = np.genfromtxt(filename, skip_header= skip)

    if parameter in headers_Stripscan or parameter =='':
           print(dat)
           df = pd.DataFrame(dat, columns = headers_Stripscan)
    else:
           df = pd.DataFrame(dat, columns=headers_HPK if HPK_file else headers_IVC)

    return df
'''

def plot_graph(x, y, color, label, title, xlab, ylab):

     plt.plot(x, y, '-o', color=color, markersize=4, label =label)
     plt.title(title, fontname="Times New Roman", fontsize=16, fontweight='bold')
     plt.xlabel(xlab, fontsize=12)
     plt.ylabel(ylab, fontsize=12)
     #v_dep1, v_dep2, a_rise, b_rise, v_rise, a_const, b_const, v_const, spl_dev = analyse_cv(x, 1./np.sqrt(y))
     #plt.vlines(v_dep2, 0, 5.5e17, colors= 'black', linestyles = 'dashed')
     #print(v_dep2)
     #plt.annotate('Threshold of non-leaky strip', (0.5, 0.45), xycoords='figure fraction', color='black')
     #plt.yscale('log')
     plt.tick_params(axis = 'y', labelsize=10)
     plt.tick_params(axis='x', labelsize=10)
     plt.legend(loc='best', fontsize=8, ncol=2)



def plot_scatter(x, y, color, label, title, xlab, ylab):

     plt.scatter(x,y, s=10, color= color, marker='o', label=label)
     plt.title(title)
     plt.xlabel(xlab)
     plt.ylabel(ylab)
     #plt.yscale('log')
     plt.legend()


def plot_lines(x, y, color, label, title, xlab, ylab):

    plt.plot(x,y, '--', linewidth=1, color=color, label=label)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)



def plot_histogram(x, title, xlab, color, lbl):

    x_mean = np.median(x)
    x_std = stdev(x)
    x = np.clip(x, -2*x_mean, 2*x_mean)

    y1,x1, _ = plt.hist(x, bins=60, color=color, label=lbl)
    plt.xlabel(xlab, fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.title(title, fontname="Times New Roman", fontsize=25, fontweight='bold')
    plt.legend()



def find_right_quantile(x):

     q = np.quantile(x, 0.99)
     return q


def find_left_quantile(x):

    v = np.quantile(x, 0.01)
    return v


def find_left_outliers(x):

    l_quant = find_left_quantile(x)
    r_out = [i for i in x if i<l_quant]
    return r_out


def find_right_outliers(x, r_quant):

    #r_quant = find_right_quantile(x)
    r_out = [i for i in x if i>r_quant]
    return r_out


def write_outliers(x):

    a = []
    l_quant = find_left_quantile(x)
    r_quant = find_right_quantile(x)
    a.append(x[x< l_quant])
    a.append(x[x > r_quant])

    f = open("outliers.txt", "a")
    for i in a:
      print(i)
      f.write(str(i))
    f.close()



def reject_outliers(x):

    #Z_left = find_left_quantile(x)
    #Z_right = find_right_quantile(x)

    x_filtered= []
    x_mean = float(np.median(x))
    print(type(x_mean))
    for i in x:
        if i<x_mean :
            x_filtered.append(i)
            #print(i)
    print(x_filtered)

    #x_filtered = x[~(x < Z_right) & (x > Z_left)]

    return x_filtered



#@params('v_dep1, v_dep2, rho, conc, a_rise, b_rise, v_rise, a_const, b_const, v_const, spl_dev, status')
def analyse_cv(v, c, area=1.56e-6, carrier='electrons', cut_param=0.004, debug=False):
    """
    Diode CV: Extract depletion voltage and resistivity.
    Parameters:
    v ... voltage
    c ... capacitance
    area ... implant size in [m^2]
    carrier ... majority charge carriers ['holes', 'electrons']
    cut_param ... used to cut on 1st derivative to id voltage regions
    Returns:
    v_dep1 ... full depletion voltage via inflection
    v_dep2 ... full depletion voltage via intersection
    rho ... resistivity
    conc ... bulk doping concentration
    """

    # init
    v_dep1 = v_dep2 = rho = conc = -1
    a_rise = b_rise = v_rise = a_const = b_const = v_const = -1
    #status = STATUS_NONE

    # invert and square
    #c = 1./c**2

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


            # full depletion voltage via max. 1st derivative
            v_dep1 = v[np.argmax(spl_dev)]

            # full depletion via intersection
            v_dep2 = (b_const - b_rise) / (a_rise - a_const)

            # rest

            #status = STATUS_PASSED

        except np.RankWarning:
            #status = STATUS_FAILED
            print("The array has too few data points. Try changing the cut_param parameter.")

        except (ValueError, TypeError, IndexError):
            #status = STATUS_FAILED
            print("The array seems empty. Try changing the cut_param parameter.")

    return v_dep1, v_dep2, a_rise, b_rise, v_rise, a_const, b_const, v_const, spl_dev#, status




def units(data, unit):

    # This function converts the unit scale to the correct one

    x = np.median(abs(data))
    numer = 0
    unit_scale = ['P', 'T', 'G', 'M', 'k', '', 'm', '$\mu$', 'n', 'p', 'f', 'a', 'z', 'y']
    i = -1
    lower_limit = 1e-24

    max_scale = 1e+18
    while max_scale> lower_limit:
          previous_max = max_scale
          max_scale = max_scale/1000
          i +=1
          if x >= max_scale and x< previous_max:
             numerator = max_scale
             string = '{}{}'.format(unit_scale[i], unit)

    return numerator, string


def normalise_parameter(parameter, unit):

    # This function scales the data and return the latter and the units in the correct form

    denominator, unit = units(parameter, unit)
    #x = np.array([j / denominator for j in parameter])
    x = parameter.divide(denominator)
    return x, unit




def outlier_aware_hist(data, lower=None, upper=None):
    upp =[]
    low=[]
    if not lower or lower < min(data):# .min():
        lower = min(data)
        lower_outliers = False
    else:
        lower_outliers = True

    if not upper or upper > max(data):
        upper = max(data)
        upper_outliers = False
    else:
        upper_outliers = True

    n, bins, patches = plt.hist(data, range=(lower, upper), bins=50, alpha=0.5, histtype='bar', ec='black', color='green')

    if lower_outliers:

        n_lower_outliers = sum([i < lower for i in data])
        for i in data:
            if i < lower:
              low.append(float(i))
        patches[0].set_height(patches[0].get_height() + n_lower_outliers)
        patches[0].set_facecolor('blue')
        patches[0].set_label('Lower outliers: ({:.3g}, {:.3g})'.format(min(low), lower))

    if upper_outliers:
        n_upper_outliers = sum([i > upper for i in data])
        for i in data:
            if i > upper:
              upp.append(float(i))
        patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
        patches[-1].set_facecolor('red')
        patches[-1].set_label('Upper outliers: ({:.3f}, {:.3f})'.format(min(upp), max(data)))

    if lower_outliers or upper_outliers:
        plt.legend()

