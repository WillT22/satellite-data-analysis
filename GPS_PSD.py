
#%% Importing all data files
import os
import sys
import glob
import spacepy.datamodel as dm
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_script_dir)

from GPS_PSD_func import import_GPS


#%% Limit data to selected time period
import datetime as dt
def data_period(data, start_date, stop_date):
    print("Identifying Relevant Time Period...")
    start_object = dt.datetime.strptime(start_date, "%m/%d/%Y")
    start_year = float(start_object.year)
    start_day = float(start_object.timetuple().tm_yday)
    
    stop_object = dt.datetime.strptime(stop_date, "%m/%d/%Y")
    stop_year = float(stop_object.year)
    stop_day = float(stop_object.timetuple().tm_yday)

    time_restricted_data = {}
    for satellite, sat_data in data.items():
        year_mask = (sat_data['year'] >= start_year) & (sat_data['year'] <= stop_year)
        day_mask = (sat_data['decimal_day'] >= start_day) & (sat_data['decimal_day'] <= stop_day)
        time_mask = year_mask & day_mask
        for item, item_data in data[satellite].items():
            if satellite not in time_restricted_data:
                time_restricted_data[satellite] = {}
            time_restricted_data[satellite][item] = item_data[time_mask]
    print("Relevant Time Period Identified \n")
    return time_restricted_data

#%% Steve's date conversion function
import datetime as dt
import spacepy.time as spt
def ticks_from_gps(data, use_astropy=False):
    '''Get a Ticktock from the year and decimal day in GPS time

    Notes
    -----
    1 - The decimal day is given as "GPS time" which is offset
    from UTC by the number of leapseconds since 1980.
    2 - The timestamps correspond to the midpoints of the integration
    intervals
    '''
    print('Converting Time for each Satellite...')
    for satellite, sat_data in data.items():
        year = sat_data['year']
        decday = sat_data['decimal_day']
        intyear = year.astype(int)
        datearray = spt.doy2date(intyear, decday, dtobj=True, flAns=True)
        # this is GPS time, so needs to be adjusted by leap seconds
        GPS0 = dt.datetime(1980, 1, 6)  # Zero epoch for GPS seconds system
        gpsoffset = datearray - GPS0
        gpsseconds = [tt.total_seconds() for tt in gpsoffset]
        if not use_astropy:
            data[satellite]['Time'] = spt.Ticktock(gpsseconds, dtype='GPS')
        else:
            import astropy.time
            data[satellite]['Time'] = astropy.time.Time(gpsseconds, format='gps')
    print('Satellite Times Converted \n')
    return data

# %% 
if __name__ == '__main__':
    # Load in data
    input_folder = "/home/will/GPS_data/april2017storm/"
    loaded_data = import_GPS(input_folder)

    # Restrict to time period
    start_date  = "04/21/2017"
    stop_date   = "04/26/2017" # exclusive, end of the last day you want to see
    storm_data = data_period(loaded_data, start_date, stop_date)

    # Convert satellite time to Ticktock object
    storm_data = ticks_from_gps(storm_data)