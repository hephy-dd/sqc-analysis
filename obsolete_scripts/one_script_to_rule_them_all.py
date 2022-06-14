import argparse
import subprocess


'This script is simply a complementary one, which actually runs the main script according to the users input'
'The options are:'
'LT: longterm test / a: SQC analysis / xml: production and uploading of xml files on the DB'

'For example: python3 one_script_to_rule_them_all.py [path] -o LT'



def parse_args():

   parser = argparse.ArgumentParser()
   parser.add_argument('path')
   parser.add_argument('-o', '--option', type = str, help = 'The option to specify which script to run')
   return parser.parse_args()


def which_script_to_run():


    args = parse_args()
    path = args.path

    if args.option == 'LT':
        p1 = subprocess.run(['python3', 'Longterm_SQC.py', '{}'.format(path)])

    elif args.option== 'xml':
        p1 = subprocess.run(['python3', 'xml_generation.py', '{}'.format(path)])

    elif args.option == 'a':
        p1 = subprocess.run(['python3', 'SQC_data_analysis.py', '{}'.format(path)])

    else:
        print("Not valid input! \nPlease enter \n'LT' : if you want to run the longterm script \n'a': if you want to analyse the ascii files "
              "\n'xml': if you want to produce and upload the xml files on the DB")




if __name__ == "__main__":
   which_script_to_run()