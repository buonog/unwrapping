TIME SERIES SIMULATOR:

1 - In the working directory (in my case: /PS_DATA/Simulated/sTS_Collections) create a new parameter's file from templates (packets_info_single_sheet_template.xlsx) modifying generation parameters;

2 - In the python code directory, from a terminal window: activate environment -> then "streamlit run sts_streamlit.py"

3 - To generate a new collection: New -> select the file excel with parameters

4 - at the end of the generation, choose a new filename, a directory will be created and inside the new directory you will find a copy of the parameter's file and the timeseries collection generated (file with the extension .stsc). In this way, everytime a new collection is generated, it will be saved in  a directory with the relative parameters excel file.
