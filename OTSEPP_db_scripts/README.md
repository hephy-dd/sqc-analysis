## How to run

To execute this script the user need to have all the rights to interact with CMS database. 
Also, the python file rhapi.py should be in the same directory with these files in order everything to run successfully.
The user should select and give as input the sensors they want to check if the data is uploaded correctly in the database.
The script queries the data and plots all SQC parameters which are stored on the SQL tables.
To execute the script, one has to type:

python3 get_SQC_data_from_DB.py -s sensor1 sensor2 sensor3 sensor4

