import pandas as pd
import ast
import datetime
from datetime import datetime
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
weekday = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0}


def load_data_by_stations():
    """
    Load data from csv
    :return:
    """
    weather_train_frame = pd.read_csv('datawithweather.csv')
    weather_train_latest = pd.read_csv('data_latest.csv')
    all_weather_data=pd.concat([weather_train_frame,weather_train_latest])
    return {k: x for k, x in all_weather_data.groupby(all_weather_data['station'])}


def availability_function_source(available_bikes, total_docks):
    """
    Given available bikes, return the probability of having a bike in source station
    :param available_bikes: amount of bikes in source station
    :type available_bikes: int
    :param total_docks: amount of total docks in station
    :type total_docks: int
    :return: probability of having a bike in source station
    :rtype: bool
    """
    return available_bikes > 0.2 * total_docks


def availability_function(available_docks, total_docks):
    """
    Given available docks, return the probability of having a bike in target station.
    :param available_docks: amount of bikes in dest station
    :type available_docks: int
    :param total_docks: amount of total docks in station
    :type total_docks: int
    :return: probability of having a bike in target station
    :rtype: bool
    """
    return available_docks > 0.2 * total_docks

def prepare_data_per_station_raw(weather_for_station):
    """
    Train logistic regression model for each of the stations
    :param source:
    :param weather_for_station:
    :type weather_for_station: pandas.DataFrame
    :return:
    """
    station_dataframe = weather_for_station.copy()
    data = [ast.literal_eval(t) for t in list(station_dataframe['data'])]
    weather = [ast.literal_eval(t) for t in list(weather_for_station['weather'])]
    station_dataframe['id'] = [d['id'] for d in data]
    station_dataframe['time_actual'] = [d['lastCommunicationTime'] for d in data]
    station_dataframe['weather'] = [w['precipProbability'] for w in weather]
    station_dataframe['temperature'] = [w['temperature'] for w in weather]
    time = [datetime.strptime(d['lastCommunicationTime'], "%Y-%m-%d %I:%M:%S %p") for d in data]
    day_in_week = [t.weekday() for t in time]
    station_dataframe['weekday'] = [weekday[d] for d in day_in_week]
    station_dataframe['time'] = [pd.Timestamp(t).round('H').hour for t in time]
    station_dataframe['availabile_bikes'] = [int(d['availableBikes']) for d in data]
    station_dataframe['availabile_docks'] = [int(d['availableDocks']) for d in data]
    station_dataframe['total_docks'] = [int(d['totalDocks']) for d in data]
    station_dataframe['windSpeed'] = [w['windSpeed'] for w in weather]
    station_dataframe['cloudCover'] = [w['cloudCover'] for w in weather]
    station_dataframe['precipIntensity'] = [w['precipIntensity'] for w in weather]
    del station_dataframe['data']
    del station_dataframe['timestamps']
    del station_dataframe['station']
    
    return station_dataframe



#def prepare_data_per_station(weather_for_station, source=True):
#    """
#    Train logistic regression model for each of the stations
#    :param source:
#    :param weather_for_station:
#    :type weather_for_station: pandas.DataFrame
#    :return:
#    """
#    station_dataframe = weather_for_station.copy()
#    data = [ast.literal_eval(t) for t in list(station_dataframe['data'])]
#    weather = [ast.literal_eval(t) for t in list(weather_for_station['weather'])]
#    station_dataframe['time'] = [d['lastCommunicationTime'] for d in data]
#    station_dataframe['weather'] = [int(w['precipProbability'] > 0.3) for w in weather]
#    time = [datetime.strptime(d['lastCommunicationTime'], "%Y-%m-%d %I:%M:%S %p") for d in data]
#    day_in_week = [t.weekday() for t in time]
#    station_dataframe['weekday'] = [weekday[d] for d in day_in_week]
#    station_dataframe['time'] = [pd.Timestamp(t).round('H').hour for t in time]
#    if source:
#        station_dataframe['availability_source'] = \
#            [int(availability_function(d['availableBikes'], d['totalDocks'])) for d in data]
#    else:
#        station_dataframe['availability_target'] = \
#            [int(availability_function(d['availableDocks'], d['totalDocks'])) for d in data]
#    del station_dataframe['data']
#    del station_dataframe['timestamps']
#    del station_dataframe['id']
#    del station_dataframe['station']
#    return station_dataframe



def prepare_data_per_station(weather_for_station, source=True):
    """
    Train logistic regression model for each of the stations
    :param source:
    :param weather_for_station:
    :type weather_for_station: pandas.DataFrame
    :return:
    """
    station_dataframe = weather_for_station.copy()
    data = [ast.literal_eval(t) for t in list(station_dataframe['data'])]
    weather = [ast.literal_eval(t) for t in list(weather_for_station['weather'])]
    station_dataframe['time'] = [d['lastCommunicationTime'] for d in data]
    station_dataframe['weather'] = [int(w['precipProbability'] > 0.3) for w in weather]
    time = [datetime.strptime(d['lastCommunicationTime'], "%Y-%m-%d %I:%M:%S %p") for d in data]
    day_in_week = [t.weekday() for t in time]
    station_dataframe['weekday'] = [weekday[d] for d in day_in_week]
    station_dataframe['time'] = [pd.Timestamp(t).round('H').hour for t in time]
    if source:
        station_dataframe['availability_source'] = \
            [int(availability_function(d['availableBikes'], d['totalDocks'])) for d in data]
    else:
        station_dataframe['availability_target'] = \
            [int(availability_function(d['availableDocks'], d['totalDocks'])) for d in data]
    del station_dataframe['data']
    del station_dataframe['timestamps']
    del station_dataframe['id']
    del station_dataframe['station']
    return station_dataframe


def create_logistic_regression_models(source_data, target_data, s=None, t=None, station_name=None,
                                      print_all_results=False):
    """
    Create logistic regression for a station given source and target data. split train and test no CV
    :param print_all_results:
    :param station_name:
    :param t:
    :param s:
    :param print_all_results:
    :param source_data: station source data
    :type source_data: pandas.DataFrame
    :param target_data: station source data
    :type target_data: pandas.DataFrame
    :return: source and data models
    :rtype tuple of sklearn.LogisticRegression
    """
    # Split train test
    train_no_label = source_data.drop(source_data.columns[[-1]], axis=1)
    source_train, source_test, source_train_labels, source_test_labels = \
        train_test_split(train_no_label, source_data['availability_source'], test_size=0.2)
    target_train, target_test, target_train_labels, target_test_labels = \
        train_test_split(train_no_label, target_data['availability_target'], test_size=0.2)

    # Source model
    source_logistic_regression = LogisticRegression()
    try:
        source_logistic_regression.fit(source_train, source_train_labels)
        source_predictions = source_logistic_regression.predict(source_test)
        source_cnf_matrix = metrics.confusion_matrix(source_test_labels, source_predictions)
        if s == station_name:
            print(source_cnf_matrix)
            print("Accuracy:", metrics.accuracy_score(source_test_labels, source_predictions))
            print("Precision:", metrics.precision_score(source_test_labels, source_predictions))
            print("Recall:", metrics.recall_score(source_test_labels, source_predictions))
        elif print_all_results:
            print(source_cnf_matrix)
            print("Accuracy:", metrics.accuracy_score(source_test_labels, source_predictions))
            print("Precision:", metrics.precision_score(source_test_labels, source_predictions))
            print("Recall:", metrics.recall_score(source_test_labels, source_predictions))
    except ValueError as e:
        print('Cannot fit logistic regression since all samples had the same label')
        source_logistic_regression = None

    # Target model
    target_logistic_regression = LogisticRegression()
    try:
        target_logistic_regression.fit(target_train, target_train_labels)
        target_predictions = target_logistic_regression.predict(target_test)
        target_cnf_matrix = metrics.confusion_matrix(target_test_labels, target_predictions)
        if t == station_name:
            print(target_cnf_matrix)
            print("Accuracy:", metrics.accuracy_score(target_test_labels, target_predictions))
            print("Precision:", metrics.precision_score(target_test_labels, target_predictions))
            print("Recall:", metrics.recall_score(target_test_labels, target_predictions))
        elif print_all_results:
            print(target_cnf_matrix)
            print("Accuracy:", metrics.accuracy_score(target_test_labels, target_predictions))
            print("Precision:", metrics.precision_score(target_test_labels, target_predictions))
            print("Recall:", metrics.recall_score(target_test_labels, target_predictions))
    except ValueError as e:
        print('Cannot fit logistic regression since all samples had the same label')
        target_logistic_regression = None

    return source_logistic_regression, target_logistic_regression


def train_stations_model(weather_data_frames, source_s, target_s, export_df=False):
    """
    Train two logistic regression models for each given station
    :param export_df:
    :param target_s:
    :param source_s:
    :param weather_data_frames: list of stations data frames
    :type weather_data_frames: dict of lists of pandas.DataFrame
    :return: dict of two trained models for each stations
    """
    stations_models_dict = {}
    for station_name, station_data_frame in weather_data_frames.items():
        source_data = prepare_data_per_station(station_data_frame, source=True)
        target_data = prepare_data_per_station(station_data_frame, source=False)
        if export_df:
            if station_name == source_s:
                source_data.to_csv(r'source.csv')
            if station_name == target_s:
                target_data.to_csv(r'target.csv')
        source_model, target_model = create_logistic_regression_models(source_data, target_data,
                                                                       source_s, target_s, station_name)
        stations_models_dict[station_name] = {'source': source_model, 'target': target_model}
    return stations_models_dict


def get_trip_prediction(s_sample, t_sample, source_station_name, target_station_name, stations_models_dict):
    """
    Given a data frame with one sample for source and target, and stations names, predict the probability of having a
    bike in source and dock in target
    :param s_sample: sample given by user
    :type s_sample: pandas.DataFrame
    :param t_sample: sample given by user
    :type t_sample: pandas.DataFrame
    :param source_station_name:
    :type source_station_name: str
    :param target_station_name:
    :type target_station_name: str
    :param stations_models_dict: dict of all trained models
    :return: probability of having a bike in source and dock in target (0/1)
    :rtype: tuple of ints
    """
    source_model, target_model = \
        stations_models_dict[source_station_name]['source'], stations_models_dict[target_station_name]['target']
    if source_model is None or target_model is None:
        print('Cannot predict since train data contained sampled from only 1 class of labels')
    return source_model.predict(s_sample)[0], target_model.predict(t_sample)[0]


if __name__ == "__main__":
    source_station = '1 Ave & E 110 St'
    target_station = '1 Ave & E 16 St'
    source_hour = 8
    target_hour = 9
    weekday_sample = 1
    # Create sample vector
    source_sample = pd.DataFrame({'time': [source_hour], 'weekday': [weekday_sample]})
    target_sample = pd.DataFrame({'time': [target_hour], 'weekday': [weekday_sample]})
    
    weather_data = load_data_by_stations()
    for station_name, station_data_frame in weather_data.items():
        raw_data = prepare_data_per_station_raw(station_data_frame)
        raw_data.to_csv(str(station_name)+'.csv')
            
#    stations_models = train_stations_model(weather_data, source_station, source_station)
#    source_probability, target_probability = \
#        get_trip_prediction(source_sample, target_sample, source_station, target_station, stations_models)
#    print(source_probability, target_probability)




