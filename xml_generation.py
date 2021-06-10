import pandas as pd
import numpy as np
import yaml
from copy import deepcopy
import re
import dicttoxml
import xml.etree.ElementTree as ET
import datetime
import subprocess
import argparse
import os
import glob



'''
This script fetches the data from the ascii files (IVC and/or Str) and produces
a xml file for each parameter individually, according to the specifications established 
from the DB and the OTSEEP group. 

Developed by Kostas Damanakis, based on Dominic's Blöch script

'''



########## Data handling functions ############


def read_yaml_configuration(yml_file):
    with open(yml_file, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf



def make_dictionary(main_parameter, config, file, headers,prefix):

    ### main_parameter: the parameter for which we will generate the xml file
    ### config: the configuration data
    ### file: the txt file to be analysed
    ### headers: the headers of the txt file data
    ### prefix: IVC or Str

    dict_with_values = {}

    if prefix =='Str':
         a= convert_txt_to_df(file, headers, 23 if '2-S' in file else 16)
         df = a[a[main_parameter].notnull()]
         for secondary_parameter in config['Parameters'][prefix][main_parameter]:
            values = df[secondary_parameter].values

            if secondary_parameter in config['Units']:

              unit = float(config['Units'][secondary_parameter])
              values = [i*unit for i in values] #convert the list with data in the required from DB units

            dict_with_values[secondary_parameter] = values



    elif prefix == 'IVC':
         a = convert_txt_to_df(file, headers, 9)
         for secondary_parameter in config['Parameters'][prefix][main_parameter]:

           if main_parameter =='capacitance':
             df = a[a[main_parameter].notnull()]
             values = df[secondary_parameter].values
           else:
             values = a[secondary_parameter].values

           if secondary_parameter in config['Units']:

             unit = float(config['Units'][main_parameter])
             values = [i * unit for i in values]  #convert the list with data in the required from DB units

           dict_with_values[secondary_parameter] = values

    return dict_with_values



def convert_txt_to_df(filename, headers, skip):

    dat = np.genfromtxt(filename, skip_header= skip)

    df = pd.DataFrame(dat, columns = headers)

    return df

##########################################################################################################


####### xml generation functions ##############################################

def dict_value_insert_iter(dictionary, header, keyword_re):

    for key, item in dictionary.items():
        if isinstance(item, dict):
            dict_value_insert_iter(item, header, keyword_re)
        else:
            keyword = keyword_re.match(str(item))

            if keyword:
                for line in header:

                    newvalue = re.search(r"{}\W\s?(.*)".format(keyword[1]), line)

                    if newvalue:
                        dictionary[key] = str(newvalue[1]).strip()
                        break
                    else:
                        # pass
                        dictionary[key] = ''
            else:
                dictionary[key] = str(None)

    return dictionary


def make_final_xml(dat, xml_string, xml_config_file):
        ###This function was developed by Dominic and can be found in the SQC COMET software
        """Inserts any template for data into the XML string and returns a XML string"""  #

        import xml.etree.ElementTree as ET

        template_re = re.compile(r"//(.*)//")  # Regex for the template
        root = ET.fromstring(xml_string)  # convert the xml string to a xmltree

        def validate_node(elem, path):
            """This just validates the node from a given path for easy access"""
            for child in elem.getchildren():
                if path[0] == child.tag:
                    if len(path[1:]):  # If len is left, the full path is not yet resolved
                        validate_node(child, path[1:])
                    else:
                        return child

        def generate_template_xml_elements(kdim, element_name, xml_node, template, data):
            """Genrerates a xml template entry"""
            xml_node.remove(
                xml_node.find(element_name)
            )  # So that the template entry is gone
            keyword_re = re.compile(r"<(.*)>")
            for i, value in enumerate(data[kdim]):

                root = ET.SubElement(xml_node, element_name)
                for key, entry in template.items():

                    data_key = keyword_re.findall(entry)

                    if data_key:
                        try:
                            element = ET.SubElement(root, key)
                            element.text = str(
                                data[entry.replace("<", "").replace(">", "")][i]
                            )

                        except IndexError:
                            print(
                                "The Index {} seems to be missing in the data".format(
                                    entry.replace("<", "").replace(">", "")
                                )
                            )
                            break
            pass

        def dict_template_insert_iter(diction, path):
            """Goes over all entries in the dict and inserts single values from the header"""
            final_tree = {}
            for key, item in diction.items():

                if isinstance(item, dict):
                    path.append(key)
                    final_tree.update(dict_template_insert_iter(item, path))
                    path.pop()
                else:
                    keyword = template_re.match(str(item))
                    subtrees = {}

                    if keyword:
                        path.append(key)
                        for kdim in xml_config_file[keyword.string.replace("/", "")]:

                            if (
                                    kdim in dat.keys()
                            ):

                                subtrees[kdim] = deepcopy(root)
                                node = validate_node(
                                    subtrees[kdim], path[:-1]
                                )  # Since we dont want the actual entry, just where to put it
                                generate_template_xml_elements(
                                    kdim,
                                    path[-1],
                                    node,
                                    xml_config_file[keyword.string.replace("/", "")][kdim],
                                    dat,
                                )
                        final_tree.update(subtrees)
                        path.pop()

            return final_tree

        xml_dicts = dict_template_insert_iter(xml_config_file["Template"], path=[])
        return xml_dicts

def change_file_specific_xml_header(final_xml_dict, template):
    """Changes the file specific header for each file"""
    import xml.etree.ElementTree as ET

    def validate_node(parent, temdict):
        try:
            for key, value in temdict.items():
                if isinstance(value, dict):
                    newvalue = validate_node(parent.find(key), value)
                else:
                    newvalue = value

                if newvalue:
                    child = parent.find(key)
                    child.text = value
        except:
            print("Child {} could not be found in xmltree. Skipping.".format(key))
            return None



    for file_header, new_header in template["File_specific_header"].items():
        if file_header in final_xml_dict:
            validate_node(final_xml_dict[file_header], new_header)
    return final_xml_dict



def save_as_xml(data_dict, filepath, name):
    from json import loads
    from dicttoxml import dicttoxml
    from xml.dom.minidom import parseString
    import xml.etree.ElementTree as ET

    """
    Writes out the data as xml file

    :param filepath: Filepath where to store the xml
    :param name: name of the file 
    :param data_dict: The data to store in this file. It has to be the dict representation of the xml file
    :return:
    """


    file = os.path.join(os.path.normpath(filepath), name.split(".")[0] + ".xml")
    file = os.path.join(os.getcwd(), file)


    if isinstance(data_dict, ET.Element):
        dom = parseString(ET.tostring(data_dict))
        with open(file, "w") as fp:
            fp.write(dom.toprettyxml())

    elif isinstance(data_dict, dict):
        xml = dicttoxml.dicttoxml(data_dict, attr_type=False)
        dom = parseString(xml)  # Pretty print style
        with open(file, "w+") as fp:
            fp.write(dom.toprettyxml())

    elif isinstance(data_dict, str):
        xml = dicttoxml.dicttoxml(loads(data_dict), attr_type=False)
        dom = parseString(xml)  # Pretty print style
        with open(file, "wb") as fp:
            fp.write(dom.toprettyxml())

    modify_xml_file(file)


def modify_xml_file(file):

    file1 = file.split('.')[0]
    lbl = '_'.join(file1.split('_')[6:7])  # dummy solution to get the parameter string from the file name
    xml_table_name =  read_yaml_configuration('config_xml_list.yml')

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
                         'EXTENSION_TABLE_NAME': 'TEST_SENSOR_{}'.format(xml_table_name['xml_table_name'][lbl]),
                         'NAME': 'Tracker Strip-Sensor {} Test'.format(xml_table_name['xml_table_name'][lbl])}

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


def generate_all_xml_files(filename, path):

    ### This is the function of functions, combines everything in order to produce eventually the xml files


    config = read_yaml_configuration('config_xml_list.yml')
    xml_config_file = read_yaml_configuration('CMSxmlTemplate.yml')

    keyword_re = re.compile(r"<(.*)>")

    for file in filename:

        prefix = '_'.join(os.path.splitext(os.path.basename(os.path.normpath(file)))[0].split('_')[0:1])
        template = deepcopy(xml_config_file['Template'])

        with open(file, 'r') as f:
            header = f.readlines()[:17 if prefix == 'Str' else 7]

        template2 = dict_value_insert_iter(template, header, keyword_re)

        for parameter in config['Parameters'][prefix]:

            if prefix == 'Str':
                headers = config['Headers']['Stripscan']
            else:
                headers = config['Headers']['IVCV']

            dict_with_values = make_dictionary(parameter, config, file, headers, prefix)

            xml_template = dicttoxml.dicttoxml(template2, attr_type=False)

            dict = make_final_xml(dict_with_values, xml_template, xml_config_file)

            dict = change_file_specific_xml_header(dict, xml_config_file)

            for subkey, value in dict.items():
                save_as_xml(
                    value,
                    path,
                    "{}_{}".format(os.path.splitext(file)[0], subkey),
                )


##################################################################################################################



########### Query and upload to DB --> get run number, upload to DB ##################################


def get_run_number():
    p1 = subprocess.run('python rhapi.py --login --url=https://cmsomsdet.cern.ch/tracker-resthub -f csv --clean "select r.run_number from trker_cmsr.trk_ot_test_nextrun_v r" ', capture_output=True)

    answer = p1.stdout.decode()
    answer = answer.split()

    #return only the number (run_number) which corresponds to the output message
    print(answer)
    return answer[1]


def upload_to_db(folder):
    for file in folder:

        try:
            p1 = subprocess.run(
                'python cmsdbldr_client.py --login --url=https://cmsdca.cern.ch/trk_loader/trker/cmsr {}', format(file),
                capture_output=True)

            answer = p1.stdout.decode()
            answer = answer.split()
            print(answer)
        except Exception as error:
            print(error)



##############################################################################################################################################


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    return parser.parse_args()


def run():

    args = parse_args()

    for subdirs, dirs, files in os.walk(args.path):

        path = args.path
        filename = glob.glob(path + os.sep + '*.txt')

        generate_all_xml_files(filename, path)

    xml_files_list = glob.glob(args.path + os.sep + '*.xml')

    #upload_to_db(xml_files_list)


if __name__ == "__main__":

    run()