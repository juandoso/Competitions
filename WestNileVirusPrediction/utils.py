'''
Created on 19/05/2015

@author: Juandoso
'''
import pandas as pd
import os, csv
from pandas.core.frame import DataFrame

data_dir = 'F:/WestNileVirusPrediction/data/'
#Add up duplicated rows
def duplicates():
    rread = csv.reader(open(os.path.join(data_dir,'train.csv'), 'rb'))
    header = rread.next()
    print header
    row0 = rread.next()
    unique_rows = [row0]
    for row in rread:
        if row0[:5] == row[:5] :
            wn = int(row[-1]) + int(row0[-1])
            #row[-1] = 0 if wn == 0 else 1
            row[-1] = int(row[-1]) + int(row0[-1])
            row[-2] = int(row[-2]) + int(row0[-2])
        else:
            unique_rows.append(row0)
            if int(row0[-1]) > 0:
                for i in range(int(row0[-1])):
                    unique_rows.append(row0)
        row0 = row
    
    writer = csv.writer(open(os.path.join(data_dir,'train3.csv'), 'wb'))
    writer.writerow(header)
    writer.writerows(unique_rows)

from datetime import datetime, timedelta
t = "2007-05-30"
weather = pd.read_csv(os.path.join(data_dir,'weather.csv'), header=0)
weather = weather.interpolate() 

def lookup_last_week_weather(look_str, weatherDF, weather_station=1):
    now = datetime.strptime(look_str, "%Y-%m-%d")
    weathers = DataFrame()
    for i in range(35):
        one_day = timedelta(days=i)
        now1 = now - one_day
        row = weatherDF[(weatherDF.Date == now1.strftime("%Y-%m-%d")) & (weatherDF.Station == weather_station)]
        weathers = weathers.append(row)
    return weathers

def weather_data(look_str, weatherDF):
    features = ["Tmax","Tmin","Tavg","DewPoint", "WetBulb", "Heat","Cool","SnowFall", "PrecipTotal", "ResultSpeed"]
    weather_week0 = lookup_last_week_weather(look_str, weatherDF)
    weather_week = weather_week0[features]
    averagesS = weather_week.mean(0)
    maxs = weather_week.max(0)
    maxsS = pd.Series()
    mins = weather_week.min(0)
    minsS = pd.Series()
    for f in features:
        maxsS["%s_max" % f] = maxs[f]
        minsS["%s_min" % f] = mins[f]
    #datapoints = pd.concat([averagesS, maxsS, minsS])
    datapoints = averagesS
    weather_data = DataFrame(datapoints).T
    weather_data["Date"] = look_str
    return weather_data
        
weather_avg = DataFrame()
dates = weather["Date"]
for d in dates:
    row = weather_data(d, weather)
    weather_avg= weather_avg.append(row, ignore_index=True)
weather_avg.to_csv(os.path.join(data_dir,'weather_info_averages5.csv'), index=False)

# duplicates()