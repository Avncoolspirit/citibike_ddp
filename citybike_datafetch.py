import urllib.request as urllib2
import pymysql
import json
import time


db = pymysql.connect("xxxx.abcd.com","user","password","db" )


response = urllib2.urlopen('https://feeds.citibikenyc.com/stations/stations.json')
html = json.loads(response.read())

response1 = urllib2.urlopen('https://api.darksky.net/forecast/ed1a586e14484a2dda7a4e8fa3b47fd8/40.7128,-74.0060')
#html1 = json.loads(response.read())
new_weather = response1.read()
weather = str(eval(new_weather)['currently'])

data_database=[]
for station in html['stationBeanList']:
    data_database.append((time.time(),station['stationName'],str(station),weather))

query="""insert into gurlam_3456( timestamps, station, data, weather) values (%s, %s, %s, %s)"""
cursor  = db.cursor()
#print data_database
cursor.executemany(query, data_database)
db.commit()

