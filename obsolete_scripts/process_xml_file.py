import xml.etree.ElementTree as ET
import glob
import os
import argparse
import datetime
import subprocess
import zipfile

parameters = {'Istrip': 'IS', 'Rpoly': 'RS', 'Cac': 'CS', 'Cint': 'CIS', 'Rint': 'RIS', 'Idiel': 'PHS', 'current': 'IV', 'capacitance': 'CV'}



def get_run_number():
    p1 = subprocess.run('python rhapi.py --login --url=https://cmsomsdet.cern.ch/tracker-resthub -f csv --clean "select r.run_number from trker_cmsr.trk_ot_test_nextrun_v r" ', capture_output=True)

    answer = p1.stdout.decode()
    answer = answer.split()

    #return only the number (run_number) which corresponds to the output message
    print(answer)
    return answer[1]


def modify_xml_file(file):

    file1 = file.split('.')[0]
    lbl = '_'.join(file1.split('_')[5:6])  # dummy solution to get the parameter string from the file name

    tree = ET.parse(file)
    root = tree.getroot()

    # First change "root" to "ROOT" --> stupid mismatch with what the DB requires
    for val in root.findall('.'):
        val.tag = 'ROOT'

    dat = root.findall('.//')

    tag_list = []
    temp = []
    delete = False
    temp_len = 5 if lbl in ['current', 'capacitance', 'Rint', 'Cint'] else 6

    for child in dat[17:]:
        temp.append(child)
        if child.text == 'nan':
            delete = True
        if len(temp) == temp_len:
            if delete == True:

                for elem in temp:
                    tag_list.append(elem)
                delete = False

            temp.clear()

    for el in root.iter():
        for child1 in list(el):
            if child1 in tag_list:
                to_remove = child1
                el.remove(to_remove)

    # necessary to update the xml format before upload it to the DB
    xml_modifications = {'VERSION': '1.0', 'LOCATION': 'HEPHY', 'RUN_TYPE': 'SQC',
                         'EXTENSION_TABLE_NAME': 'TEST_SENSOR_{}'.format(parameters[lbl]),
                         'NAME': 'Tracker Strip-Sensor {} Test'.format(parameters[lbl])}

    run_number = get_run_number()

    for elm in root.findall(".//"):
        if elm.tag in xml_modifications.keys():
            elm.text = xml_modifications[elm.tag]
        if elm.tag == 'RUN_BEGIN_TIMESTAMP':
            date = elm.text
            new_date = datetime.datetime.strptime(date, '%a %b %d %H:%M:%S %Y')
            elm.text = str(new_date)
        if elm.tag == 'RUN_NUMBER':
            elm.text = str(run_number)

    tree.write(file, xml_declaration=True, encoding='UTF-8')


def upload_to_db(folder):


    for file in folder:

       try:
            p1 = subprocess.run('python cmsdbldr_client.py --login --url=https://cmsdca.cern.ch/trk_loader/trker/cmsr {}',format(file),
                               capture_output=True)
             
             answer = p1.stdout.decode()
             answer = answer.split()
             print(answer)
       except Exception as error:
            print(error)


def make_zip_file(path):

    filePaths = []
    
    for dirname, subdirs, files in os.walk(path):
        for filename in files:
            # Create the full filepath by using os module.
            filePath = os.path.join(dirname, filename)
            filePaths.append(filePath)
    print(filePaths)

    with zipfile.ZipFile("zipfile.zip", "w", zipfile.ZIP_DEFLATED) as zf:
      for file in filePaths:
            file1 = file.split(os.sep).pop()
            print(file1)
            zf.write(file1)#os.path.join(dirname, filename))
      zf.close()


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    return parser.parse_args()


def main():


    args = parse_args()
    files = glob.glob(args.path + os.sep + '*.xml')

    #for file in files:
        #modify_xml_file(file)

    #upload_to_db(files)

if __name__ == "__main__":
    main()
