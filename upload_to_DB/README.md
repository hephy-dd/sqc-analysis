# How to upload SQC data to the CMS database

python3 xml_generation.py [path]

The script generates the xml files based on the ASCII files where all measured parameters are stored. 
It gets a run number from the DB by using the rhapi.py script. All the xml files are stored in a new folder.
The script uploads all xml files to the database by using the cmsdbldr_client.py script.

Note: The line which calls the function to upload the data is by default commented out. 
      First check whether the scripts produce the correct format of xml files with a run number before you upload the data.
