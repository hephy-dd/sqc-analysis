import pandas as pd
import os
import sys
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt
import holoviews as hv
from bokeh.io import  output_file, show
from bokeh.models import LinearAxis, Range1d
from bokeh.models.renderers import GlyphRenderer


hv.extension('bokeh')



def make_dataframe_from_ascii(datafile, skip):

    data = np.genfromtxt(datafile, skip_header=skip, delimiter=",", max_rows = 20348)

    df = pd.DataFrame(data, columns= ['Timestamp', 'Voltage', 'Current', 'smu_current', 'pt100', 'cts_temp', 'cts_humi', 'cts_status', 'cts_program', 'hv_status'])

    return df



def plot_Ivstime(df, label, color, style):
    # plot the time evolution of current

    df['Current'] = (df['Current'].abs())*1e9


    df1 = pd.DataFrame({'Time [h]': df['Timestamp'].divide(3600), 'Current [nA]': df['Current'].values})
    plot = (hv.Curve(df1, label=label)).opts(width=800, height = 600, title = 'Current over time', fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12})

    return plot



def plot_IV(df, label, color):
    # plot the IV

    df1 = pd.DataFrame({'Voltage [V]': df['Voltage'].abs(), 'Current [nA]': ((df['Current'].abs())*1e9).values})
    plot = (hv.Curve(df1, label=label)).opts(width=800, height=600, title='IV curve',
                                             fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12})

    return plot




def plot_temp_humi(df):

    df['Timestamp'] = df['Timestamp'].divide(3600)
    df_temp = pd.DataFrame({'Time [h]': df.get('Timestamp'), 'Temperature [\u00B0C]': df.get('cts_temp').values})
    df_humi = pd.DataFrame({'Time [h]': df.get('Timestamp'), 'Rel. Humidity [%]': df.get('cts_humi').values})



    def apply_formatter(plot):
        # this function is necessary for the construction of the second y-axis --> humidity axis
        p = plot.state

        # create secondary range and axis
        p.extra_y_ranges = {"twiny": Range1d(start=0, end=60)}
        p.add_layout(LinearAxis(y_range_name="twiny", axis_label="Rel. Humidity [%] ", axis_label_text_color='blue'), 'right')

        # set glyph y_range_name to the one we've just created
        glyph = p.select(dict(type=GlyphRenderer))[0]
        glyph.y_range_name = "twiny"


    c_def = hv.Curve(df_temp , name='Temperature [\u00B0C]', label = 'Temperature').options(color='red', width=800, height = 600, title = 'Environmental conditions over time', fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12})
    c_sec = hv.Curve(df_humi, name = 'Rel. Humidity [%]', label = 'Humidity' ).options(color='blue', width=800, height = 600,  fontsize={'title': 15, 'labels': 14, 'xticks': 12, 'yticks': 12}) #, hooks=[apply_formatter])
    plot = c_def + c_sec #+ c_def*c_sec

    return plot



def find_time_temperature(df):

    time_list = []
    df['Timestamp'] = df['Timestamp'].divide(3600)
    for n in range(5):
      temp = df['cts_temp'][0] + n*7.0
      a = df.loc[(df['cts_temp']<= temp+0.1) & (df['cts_temp']>= temp-0.1)]
      time_list.append(a['Timestamp'].values[0])

    return time_list



def find_current_at_humi_level(df):
    # useless function, the idea is to create lists containing the currents which correspond to some particular humidity values

    current = df['Current']
    humi = df['cts_humi']
    i_30rh= []
    i_40rh = []
    i_50rh = []
    i_60rh = []
    i_30rh.append(df['Current'].loc[(29.0< df['cts_humi']) & (df['cts_humi']<31.0)].values[0])
    i_40rh.append(df['Current'].loc[(39.0< df['cts_humi']) &  (df['cts_humi']<41.0)].values[0])
    i_50rh.append(df['Current'].loc[(49.0 < df['cts_humi']) & (df['cts_humi'] <51.0)].values[0])
    i_60rh.append(df['Current'].loc[(59.0<df['cts_humi'])   & (df['cts_humi']<61.0)].values[0])

    print(i_30rh, i_40rh,  i_50rh,  i_60rh)


def return_current(df):
    df['Current'] = df['Current'] * 1e9

    return df['Current'], df



def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    return parser.parse_args()


def make_color_list():

    colors = []
    evenly_spaced_interval = np.linspace(0, 1, 10)
    for x in evenly_spaced_interval:
        colors.append(plt.cm.tab10(x))

    return colors


def create_list_with_plots(files):

    index_it = 0
    index_iv = 0

    colors = make_color_list()
    plot_iv_list = []
    plot_it_list = []

    df = pd.DataFrame({})
    df1 = pd.DataFrame({})

    for f in files:

        file = f.split(os.sep)[-1]
        sensor = '_'.join(os.path.splitext(file)[0].split('_')[0:4])
        sensor = '-'.join(os.path.splitext(sensor)[0].split('-')[1:3]) if '2-S' in sensor else '-'.join(
            os.path.splitext(sensor)[0].split('-')[1:2])

        if "it" in f:
            index_it += 1

            df = make_dataframe_from_ascii(f, 9)
            plot_it_list.append(plot_Ivstime(df, sensor, colors[index_it], '-'))



        elif "IV" in f:
            index_iv += 1

            df1 = make_dataframe_from_ascii(f, 9)
            plot_iv_list.append(plot_IV(df1, sensor, colors[index_iv]))

    return plot_it_list, plot_iv_list, df, df1



def main():

    args = parse_args()

    fig_index = 0
    for subdir, dirs, files in os.walk(args.path):

       if len(dirs)>=1:

           for dir in dirs:
               fig_index +=1
               files = glob.glob(args.path + os.sep + dir + os.sep + '*.txt')

               plot_it_list, plot_iv_list, df, df1 = create_list_with_plots(files)
               create_bookeh_plots(plot_it_list, plot_iv_list, df, fig_index)


       else:
            path = args.path

            files = glob.glob(path + os.sep + '*.txt')
            plot_it_list, plot_iv_list, df, df1 = create_list_with_plots(files)

            create_bookeh_plots(plot_it_list, plot_iv_list, df, fig_index)



def create_bookeh_plots(plot_it_list, plot_iv_list, df, fig_index):

       new_plot = hv.Overlay(plot_it_list)
       new_plot.opts(legend_position='bottom_right')
       new_plot.opts(norm={'axiswise': False})
       IV_plot = hv.Overlay(plot_iv_list)
       IV_plot.opts(legend_position='top_right')
       IV_plot.opts(norm={'axiswise': False})

       humi = plot_temp_humi(df)
       new_plot2 = hv.Layout( IV_plot+ humi + new_plot).cols(1)
       new_plot2.opts(shared_axes=False)
       renderer = hv.renderer('bokeh')
       renderer.save(new_plot2, 'fig1.html')

       final_plot = renderer.get_plot(new_plot2).state
       output_file("fig{}.html".format(fig_index))
       show(final_plot)#, gridplot(children = humi, ncols = 1, merge_tools = False))


       plot_iv_list.clear()
       index_it = 0
       index_iv = 0



if __name__ == "__main__":
    main()