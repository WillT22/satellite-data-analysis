import spacepy.datamodel as dm
import datetime as dt
import numpy as np
import os

#%% Limit data to selected time period
def data_period(QD_folder, start_date, stop_date):
    print("Identifying Relevant Time Period...")
    start_date = dt.datetime.strptime(start_date, "%m/%d/%Y") 
    stop_date = dt.datetime.strptime(stop_date, "%m/%d/%Y")

    time_difference = stop_date - start_date
    number_of_days = time_difference.days
    QD_filenames = []
    current_date = start_date
    while current_date < stop_date:
        print(current_date)
        QD_year = os.path.join(QD_folder, str(current_date.year),'5min')
        QD_filenames.append(os.path.join(QD_year, f"QinDenton_{current_date.strftime("%Y%m%d")}_5min.txt"))
        current_date += dt.timedelta(days=1)
    QD_data = dm.readJSONheadedASCII(QD_filenames)
    return QD_data

#%% Main
QD_folder = "/home/will/QinDenton/"

start_date  = "01/19/2000"
stop_date   = "02/02/2000" # exclusive, end of the last day you want to see
QD_data = data_period(QD_folder, start_date, stop_date)

#QD_data.tree(verbose=True, attrs=True)