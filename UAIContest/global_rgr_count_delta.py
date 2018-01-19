#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
	import cPickle as pickle
except:
	import pickle
import datetime
from collections import defaultdict
import data_provider
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans

poi_csv_name = 'data/poi.csv'
weather_csv_name = 'data/weather1.csv'
July_csv_name = 'data/train_July.csv'
Aug_csv_name = 'data/train_Aug.csv'
test_csv_name = 'data/test_id_Aug_agg_private5k.csv'
result_csv_name = 'global_result_01_15_private5k_count_delta_prev.csv'

poi_avg = []
d0 = datetime.datetime(2017, 7, 1)
n_clusters = 6
hour_delta = -1

'''
0,1:晴, 4:多云, 9:阴,
10:阵雨, 11:雷阵雨, 13:小雨, 14:中雨, 15:大雨
'''
weather_code = {
	'0': '0', '1': '0', 
	'4': '1',
	'9': '2',
	'10': '3',
	'11': '4',
	'13': '5',
	'14': '6',
	'15': '7',
}

def preDate(date):
	year, month, day = date.split('-')
	year, month, day = int(year), int(month), int(day)
	if day == 1:
		month -= 1
		day = 31
	else:
		day -= 1
	res = '%04d-%02d-%02d' % (year, month, day)
	return res

def nextDate(date):
	year, month, day = date.split('-')
	year, month, day = int(year), int(month), int(day)
	if day == 31:
		month += 1
		day = 1
	else:
		day += 1
	res = '%04d-%02d-%02d' % (year, month, day)
	return res

def createHours(create_date, create_hour, diffs=[0]):
	hour = int(create_hour)
	hours = [(_d + hour) for _d in diffs]
	create_hours = []
	for h in hours:
		c_date = create_date
		if h < 0:
			h += 24
			_h = int(h)
			_m = 30 if (h - int(h)) == 0.5 else 0
			c_date = preDate(create_date)
		elif h >= 24:
			h -= 24
			_h = int(h)
			_m = 30 if (h - int(h)) == 0.5 else 0
			c_date = nextDate(create_date)
		else:
			_h = int(h)
			_m = 30 if (h - int(h)) == 0.5 else 0
			c_date = create_date
		c_hour = '%02d:%02d' % (_h, _m)
		_hour = '%s,%s' % (c_date, c_hour)
		create_hours.append(_hour)
	return create_hours

def calWeekday(date):
	year, month, day = date.split('-')
	year, month, day = int(year), int(month), int(day)
	day = datetime.datetime(year, month, day)
	return day.weekday()

def cal_date_id(date):
	year, month, day = date.split('-')
	year, month, day = int(year), int(month), int(day)
	d1 = datetime.datetime(year, month, day)
	return (d1 - d0).days

def create_poi_fea(A_geo_id, B_geo_id, poi_dict_data, poi_kmeans_labels):
	A_poi_nums = 0
	A_kmeans_label = 0
	if A_geo_id in poi_dict_data:
		_poi_nums = poi_dict_data[A_geo_id].split(',')
		A_poi_nums = [int(_t) for _t in _poi_nums]
		A_poi_nums = sum(A_poi_nums)
		A_kmeans_label = poi_kmeans_labels[A_geo_id]
	B_poi_nums = 0
	B_kmeans_label = 0
	if B_geo_id in poi_dict_data:
		_poi_nums = poi_dict_data[B_geo_id].split(',')
		B_poi_nums = [int(_t) for _t in _poi_nums]
		B_poi_nums = sum(B_poi_nums)
		B_kmeans_label = poi_kmeans_labels[B_geo_id]
	# poi_fea = [B_poi_nums[i]-A_poi_nums[i] for i in range(len(A_poi_nums))]
	# poi_fea = [A_poi_nums, B_poi_nums]
	poi_fea = [A_kmeans_label, B_kmeans_label]
	return poi_fea

def count_poi_data(data):
	poi_dict_data = defaultdict(str)
	poi_avg = [0] * 10
	cc = 0
	for each_data in data:
		geo_id = each_data[0]
		nums = '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' % (
			each_data[2], each_data[4], each_data[6], each_data[8], each_data[10], 
			each_data[12], each_data[14], each_data[16], each_data[18], each_data[20])
		poi_dict_data[geo_id] = nums
		_nums = nums.split(',')
		poi_avg = [poi_avg[i]+int(_nums[i]) for i in range(10)]
		cc += 1
	poi_avg = [n * 1.0 / cc for n in poi_avg]
	# print(poi_avg)
	return poi_dict_data, poi_avg

def create_poi_data(poi_dict_data):
	X = []
	for geo_id in poi_dict_data:
		nums = poi_dict_data[geo_id].split(',')
		nums = [int(num) for num in nums]
		X.append(nums)
	X = np.array(X)
	return X

def kmeans_cluster(poi_X):
	kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(poi_X)
	# print kmeans.cluster_centers_
	return kmeans

def get_kmeans_labels(kmeans, poi_dict_data):
	poi_kmeans_labels = {}
	for geo_id in poi_dict_data:
		nums = poi_dict_data[geo_id].split(',')
		nums = [int(num) for num in nums]
		label = kmeans.predict([nums])
		poi_kmeans_labels[geo_id] = label[0]
	return poi_kmeans_labels

def count_weather_data(data):
	# date,text,code,temperature,feels_like,pressure,humidity,visibility,wind_direction,wind_direction_degree,wind_speed,wind_scale
	res = {}
	for each_data in data:
		date = each_data[0]
		create_date, create_hour = date.split(' ')
		year, month, day = create_date.split('-')
		hour, minute = create_hour.split(':')
		create_date = '%04d-%02d-%02d' % (int(year), int(month), int(day))
		create_hour = '%02d:%02d' % (int(hour), int(minute))
		key = '%s,%s' % (create_date, create_hour) # 2017-07-14 00:30
		code, visibility, wind_speed, wind_scale = each_data[2], each_data[7], each_data[10], each_data[11]
		# code = weather_code[str(int(code))]
		res[key] = '%s,%s,%s,%s' % (code, visibility, wind_speed, wind_scale)
	return res

def query_weather(weather_dict_data, weather_key):
	# weather_key: 2017-07-14,00:30
	weather = None
	if weather_key in weather_dict_data:
		weather = weather_dict_data[weather_key]
	else:
		_date, _hour = weather_key.split(',')
		_hour = _hour.split(':')[0]
		prev_weather = None
		prev_hour_deltas = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
		for prev_hour_delta in prev_hour_deltas:
			pre_hour = createHours(_date, _hour, [prev_hour_delta])[0]
			if pre_hour in weather_dict_data:
				prev_weather = weather_dict_data[pre_hour]
				break
		next_weather = None
		next_hour_deltas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
		for next_hour_delta in next_hour_deltas:
			next_hour = createHours(_date, _hour, [next_hour_delta])[0]
			if next_hour in weather_dict_data:
				next_weather = weather_dict_data[next_hour]
				break
		if prev_weather != None and next_weather != None:
			prev_weather = prev_weather.split(',')
			next_weather = next_weather.split(',')
			assert len(prev_weather) == len(next_weather)
			weather = [str((float(prev_weather[i])+float(next_weather[i]))/2.0) for i in range(len(prev_weather))]
			if int(prev_weather[0]) != int(next_weather[0]):
				weather[0] = next_weather[0]
			weather = ','.join(weather)
		elif prev_weather != None:
			weather = prev_weather
		else:
			weather = next_weather
	return weather

def create_fea(create_date, create_hour, weather_dict_data, key1=None, count_dict_data=None):
	# 2017-08-03 06
	rain_code = [10, 11, 13, 14, 15]
	hour = int(create_hour)
	create_hours = createHours(create_date, create_hour, [-3, -2, -1, 0, 1, 2, 3])
	rains = []
	for weather_key in create_hours:
		# code, visibility, wind_speed, wind_scale
		weather = query_weather(weather_dict_data, weather_key)
		if weather != None:
			code = int(float(weather.split(',')[0]))
			if code in rain_code: rains.append('1')
			else: rains.append('0')
		else:
			rains.append('0')
	rains_fea = ','.join(rains)
	date_id = cal_date_id(create_date)
	weekday = calWeekday(create_date)
	week_fea = str(weekday)
	hour_fea = str(hour)
	if count_dict_data != None:
		delta_hour_counts = []
		create_hours = createHours(create_date, create_hour, [hour_delta])
		for c_hour in create_hours:
			_date, _hour = c_hour.split(',')
			_hour = _hour.split(':')[0]
			delta_hour_key = '%s,%s' % (_date, _hour)
			if key1 in count_dict_data:
				if delta_hour_key in count_dict_data[key1]:
					delta_hour_counts.append(str(count_dict_data[key1][delta_hour_key]))
				else:
					delta_hour_counts.append('0')
			else:
				delta_hour_counts.append('0')
		delta_hour_counts_fea = ','.join(delta_hour_counts)
		fea = '%s,%s,%s,%s,%s' % (date_id, week_fea, hour_fea, delta_hour_counts_fea, rains_fea)
	else:
		fea = '%s,%s,%s' % (hour_fea, week_fea, rains_fea)
	return fea

def convert_fea(fea):
	fea = fea.split(',')
	fea = [float(_t) if _t != '' else 0.0 for _t in fea]
	return fea

def gbr_train(train_X, train_Y, model_path=""):
	if model_path == "":
		print('training gbr model.')
		# 1.7188
		# clf = GradientBoostingRegressor(loss='lad', n_estimators=400, max_depth=300, learning_rate=0.1,
		# 	min_samples_leaf=256, min_samples_split=256, random_state=1024)
		# 1.7114
		# clf = GradientBoostingRegressor(loss='lad', n_estimators=400, max_depth=350, learning_rate=0.1,
		# 	min_samples_leaf=128, min_samples_split=128, random_state=1024)
		# 1.7136
		# clf = GradientBoostingRegressor(loss='lad', n_estimators=400, max_depth=400, learning_rate=0.1,
		# 	min_samples_leaf=100, min_samples_split=100, random_state=1024)
		# 1.7184 on DELL PC
		# clf = GradientBoostingRegressor(loss='lad', n_estimators=500, max_depth=350, learning_rate=0.05,
		# 	min_samples_leaf=128, min_samples_split=128, random_state=1024)
		clf = GradientBoostingRegressor(loss='lad', n_estimators=400, max_depth=350, learning_rate=0.1,
			min_samples_leaf=160, min_samples_split=160, random_state=1024)
		clf.fit(train_X, train_Y)
		saved_path = "delta_hour_gbr.pkl"
		with open(saved_path, 'wb') as fid:
			pickle.dump(clf, fid)
	else:
		print('reading gbr model.')
		with open(model_path, 'rb') as fid:
			clf = pickle.load(fid)
		print('feature importances: ', clf.feature_importances_)
	return clf

def test_predict(test_data, weather_dict_data, poi_dict_data, poi_kmeans_labels, Aug_count_dict_data, train_model):
	# test_id,start_geo_id,end_geo_id,create_date,create_hour
	global_pred_Ys = []
	print('predciting test data.')
	for each_data in test_data:
		start_geo_id = each_data[1]
		end_geo_id = each_data[2]
		key1 = '%s,%s' % (start_geo_id, end_geo_id)
		create_date, create_hour = each_data[3], each_data[4]
		create_hours = createHours(create_date, create_hour, [0])
		Ys = []
		for c_hour in create_hours:
			_date, _hour = c_hour.split(',')
			_hour = _hour.split(':')[0]
			fea = create_fea(_date, _hour, weather_dict_data, key1, Aug_count_dict_data)
			fea = convert_fea(fea)
			poi_fea = create_poi_fea(start_geo_id, end_geo_id, poi_dict_data, poi_kmeans_labels)
			fea = fea + poi_fea
			_x = np.array([fea], dtype=np.float64)
			_y = train_model.predict(_x)
			if Aug_count_dict_data != None:
				delta_hour_count = 0.0
				delta_hour = createHours(create_date, create_hour, [hour_delta])[0]
				_date, _hour = delta_hour.split(',')
				_hour = _hour.split(':')[0]
				delta_hour_key = '%s,%s' % (_date, _hour)
				if key1 in Aug_count_dict_data:
					if delta_hour_key in Aug_count_dict_data[key1]:
						delta_hour_count = Aug_count_dict_data[key1][delta_hour_key]
				_y = (_y[0] + delta_hour_count) if (_y[0] + delta_hour_count) >= 0.0 else 0.0
			else:
				_y = _y[0] if _y[0] >= 0.0 else 0.0
			Ys.append(_y)
		global_pred_Ys.append(Ys[0])
	return global_pred_Ys

def create_test_fea(test_data, weather_dict_data, poi_dict_data, poi_kmeans_labels, Aug_count_dict_data):
	# test_id,start_geo_id,end_geo_id,create_date,create_hour
	test_feas = []
	print('predciting test data.')
	for each_data in test_data:
		start_geo_id = each_data[1]
		end_geo_id = each_data[2]
		key1 = '%s,%s' % (start_geo_id, end_geo_id)
		create_date, create_hour = each_data[3], each_data[4]
		create_hours = createHours(create_date, create_hour, [0])
		Ys = []
		for c_hour in create_hours:
			_date, _hour = c_hour.split(',')
			_hour = _hour.split(':')[0]
			fea = create_fea(_date, _hour, weather_dict_data, key1, Aug_count_dict_data)
			fea = convert_fea(fea)
			poi_fea = create_poi_fea(start_geo_id, end_geo_id, poi_dict_data, poi_kmeans_labels)
			fea = fea + poi_fea
			test_feas.append(fea)
	import pdb
	pdb.set_trace()
	return test_feas

def count_data(data):
	count_dict_data = {}
	# id,driver_id,member_id,create_date,create_hour,status,estimate_money,estimate_distance,estimate_term,start_geo_id,end_geo_id
	for each_data in data:
		start_geo_id = each_data[9]
		end_geo_id = each_data[10]
		create_date, create_hour = each_data[3], each_data[4] # 2017-08-03 06
		key1 = '%s,%s' % (start_geo_id, end_geo_id)
		key2 = '%s,%s' % (create_date, create_hour)
		if key1 in count_dict_data:
			count_dict_data[key1][key2] += 1
		else:
			count_dict_data[key1] = defaultdict(float)
			count_dict_data[key1][key2] = 1
	return count_dict_data

def create_data(July_count_dict_data, weather_dict_data, poi_dict_data, poi_kmeans_labels):
	train_X = []
	train_Y = []
	for key1 in July_count_dict_data:
		A_geo_id, B_geo_id = key1.split(',')
		for key2 in July_count_dict_data[key1]:
			create_date, create_hour = key2.split(',')
			fea = create_fea(create_date, create_hour, weather_dict_data, key1, July_count_dict_data)
			fea = convert_fea(fea)
			poi_fea = create_poi_fea(A_geo_id, B_geo_id, poi_dict_data, poi_kmeans_labels)
			fea = fea + poi_fea
			train_X.append(fea)
			delta_hour = createHours(create_date, create_hour, [hour_delta])[0]
			_date, _hour = delta_hour.split(',')
			_hour = _hour.split(':')[0]
			delta_hour_key = '%s,%s' % (_date, _hour)
			delta_hour_count = 0.0
			if delta_hour_key in July_count_dict_data[key1]:
				delta_hour_count = July_count_dict_data[key1][delta_hour_key]
			label = July_count_dict_data[key1][key2] * 1.0 - delta_hour_count
			train_Y.append(label)
	return train_X, train_Y

def cal_avgs(train_count_dict_data, weighted=True):
	train_avgs = {}
	for key1 in train_count_dict_data:
		train_avgs[key1] = defaultdict(float)
		date_dict = defaultdict(set)
		for key2 in train_count_dict_data[key1]:
			create_date, create_hour = key2.split(',')
			train_avgs[key1][create_hour] += train_count_dict_data[key1][key2]
			date_dict[create_hour].add(create_date)
		for hour_key2 in train_avgs[key1]:
			# train_avgs[key1][hour_key2] = train_avgs[key1][hour_key2] * 1.0 / len(date_dict[hour_key2])
			if int(hour_key2) % 2 == 0:
				train_avgs[key1][hour_key2] = train_avgs[key1][hour_key2] * 1.0 / 35
			else:
				train_avgs[key1][hour_key2] = train_avgs[key1][hour_key2] * 1.0 / 34
	if weighted:
		weighted_train_avgs = {}
		for key1 in train_avgs:
			weighted_train_avgs[key1] = defaultdict(float)
			for hour_key2 in train_avgs[key1]:
				current_avg = train_avgs[key1][hour_key2]
				current_hour = int(hour_key2)
				pre_hour = (current_hour - 1) if (current_hour - 1) >= 0 else 23
				pre_hour_key2 = '%02d' % (pre_hour)
				next_hour = (current_hour + 1) if (current_hour + 1) <= 23 else 0
				next_hour_key2 = '%02d' % (next_hour)
				pre_avg = 0.0
				next_avg = 0.0
				if pre_hour_key2 in train_avgs[key1]:
					pre_avg = train_avgs[key1][pre_hour_key2]
				if next_hour_key2 in train_avgs[key1]:
					next_avg = train_avgs[key1][next_hour_key2]
				weighted_train_avgs[key1][hour_key2] = np.ceil(current_avg * 0.6 + pre_avg * 0.2 + next_avg * 0.2)
	if weighted:
		return train_avgs, weighted_train_avgs
	else:
		return train_avgs

def preprocess_data(train_X, train_Y, return_scaler=False, do_scaler=False, _train_scaler=None):
	if return_scaler:
		train_scaler = None
	X = np.array(train_X, dtype=np.float64)
	if return_scaler:
		scaler = StandardScaler()
		scaler.fit(X)
		X = scaler.transform(X)
		train_scaler = scaler
	if do_scaler == True and _train_scaler != None:
		X = _train_scaler.transform(X)
	train_X = X
	train_Y = np.array(train_Y, dtype=np.float64)
	if return_scaler:
		return train_X, train_Y, train_scaler
	else:
		return train_X, train_Y

def maer(pred_Ys, gt_Ys):
	assert len(pred_Ys) == len(gt_Ys)
	e = 0.0
	for i in range(len(pred_Ys)):
		e += abs(pred_Ys[i] - gt_Ys[i])
	return e * 1.0 / len(pred_Ys)

def main():
	# create_hours = createHours('2017-07-31', '23', [-1, 1])
	# print create_hours
	poi_data = data_provider.read_poi_csv(poi_csv_name)
	poi_dict_data, poi_avg = count_poi_data(poi_data)
	poi_X = create_poi_data(poi_dict_data)
	kmeans = kmeans_cluster(poi_X)
	poi_kmeans_labels = get_kmeans_labels(kmeans, poi_dict_data)

	weather_data = data_provider.read_csv(weather_csv_name)
	weather_dict_data = count_weather_data(weather_data)
	# print weather_dict_data['2017-07-15,18:30']

	July_data = data_provider.read_csv(July_csv_name)
	Aug_data = data_provider.read_csv(Aug_csv_name)
	print('July data:', len(July_data))
	print('Aug_data:', len(Aug_data))
	test_data = data_provider.read_csv(test_csv_name)
	print('test data:', len(test_data))

	July_count_dict_data = count_data(July_data)
	Aug_count_dict_data = count_data(Aug_data)
	
	July_X, July_Y = create_data(July_count_dict_data, weather_dict_data, poi_dict_data, poi_kmeans_labels)
	July_X, July_Y = preprocess_data(July_X, July_Y)
	print('July_Y:', July_X.shape, 'July_Y:', July_Y.shape)

	train_model = gbr_train(July_X, July_Y, model_path="delta_hour_gbr.pkl")

	global_pred_Ys = test_predict(test_data, weather_dict_data, poi_dict_data, poi_kmeans_labels, Aug_count_dict_data, train_model)
	data_provider.write_csv(global_pred_Ys, result_csv_name)

if __name__ == '__main__':
	main()
