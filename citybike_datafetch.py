import urllib.request as urllib2
import pymysql
import json
import time

#TO DO: Make the script to make a schema in the rds database

#make the connection with the datbase
db = pymysql.connect("xxxx.abcd.com","user","password","db" )


#read data from the opennyc citibike feed
response_citibike = urllib2.urlopen('https://feeds.citibikenyc.com/stations/stations.json')
html = json.loads(response_citibike.read())

response_weather = urllib2.urlopen('https://api.darksky.net/forecast/ed1a586e14484a2dda7a4e8fa3b47fd8/40.7128,-74.0060')
new_weather = response_weather.read()
weather = str(eval(new_weather)['currently'])

#parse the data and split it according to the station
data_database=[]
for station in html['stationBeanList']:
    data_database.append((time.time(),station['stationName'],str(station),weather))

#feed into the database
query="""insert into gurlam_3456( timestamps, station, data, weather) values (%s, %s, %s, %s)"""
cursor  = db.cursor()
cursor.executemany(query, data_database)
db.commit()

