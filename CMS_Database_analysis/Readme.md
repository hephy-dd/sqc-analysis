
# CMS database analysis

## SQC analysis
For this part the following scripts are necessary:

* query_SQC_data_from_DB.py : executes queries and fetches data from the database
* rhapi.py: script developed by CMS to interract with DB
* analysis_tools.py: number of methods which are used for data analysis
* SQC_analysis_db.py: all routines which correspond to the measurements, the data of which is analysed.
* run_SQC.py: a structured way to execute the analysis. That's ultimately the script which one has to execute to produce the plots.

To analyse SQC data from database the following steps must be followed:

* Query data from DB first: The idea is that we query the data with a SQL commands in a pythonic way. The data are stored into json files
  in order to facilitate the analysis and make it independent of the database.
  To query data simply execute the following:

         python3 query_SQC_data_from_DB.py
         
  This gonna take a while. When its done verify that the json files contain data. Otherwise some error in the communication with DB must have occured.

* Analyse the data and produce the default plots:

        python3 run_SQC.py


