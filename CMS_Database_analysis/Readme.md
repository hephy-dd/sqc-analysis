
# CMS database analysis

## SQC analysis
To analyse SQC data from database the following steps must be followed:

* Query data from DB first: The idea is that we query the data with a SQL commands in a pythonic way. The data are stored into json files
  in order to facilitate the analysis and make it independent of the database.
  To query data simply execute the following:

         python3 query_SQC_data_from_DB.py
         
  This gonna take a while. When its done verify that the json files contain data. Otherwise some error in the communication with DB must have occured.

* Analyse the data and produce the default plots:

        python3 run_SQC.py


