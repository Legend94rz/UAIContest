#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
from collections import defaultdict
import data_provider
import numpy as np

poi_csv_name = 'data/poi.csv'
weather_csv_name = 'data/weather.csv'
July_csv_name = 'data/train_July.csv'
Aug_csv_name = 'data/train_Aug.csv'
test_csv_name = 'data/test_id_Aug_agg_private5k.csv'
result_csv_name = 'Aug_mean_private5k.csv'
# result_csv_name = 'July_Aug_mean.csv'

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

def count_Aug_mean_data(data, Aug_count_dict_data):
	mean_res = []
	for each_data in data:
		start_geo_id = each_data[1]
		end_geo_id = each_data[2]
		key1 = '%s,%s' % (start_geo_id, end_geo_id)
		create_date, create_hour = each_data[3], each_data[4]
		_Ys = 0.0
		cc = 2.0
		if key1 in Aug_count_dict_data:
			if int(create_hour) == 23:
				_hour = '%02d' % (int(create_hour)-1)
				key2 = '%s,%s' % (create_date, _hour)
				_Ys += Aug_count_dict_data[key1][key2]
				mean_res.append(_Ys)
				continue
			if int(create_hour) == 0:
				_hour = '%02d' % (int(create_hour)+1)
				key2 = '%s,%s' % (create_date, _hour)
				_Ys += Aug_count_dict_data[key1][key2]
				mean_res.append(_Ys)
				continue
			create_hours = createHours(create_date, create_hour, [-1, 1])
			for c_hour in create_hours:
				_date, _hour = c_hour.split(',')
				_hour = _hour.split(':')[0]
				key2 = '%s,%s' % (_date, _hour)
				if key2 in Aug_count_dict_data[key1]:
					_Ys += Aug_count_dict_data[key1][key2]
			_Ys = _Ys * 1.0 / cc
			mean_res.append(_Ys)
		else:
			mean_res.append(0)
	return mean_res

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

def cal_avgs(train_count_dict_data, weighted=True):
	train_avgs = {}
	for key1 in train_count_dict_data:
		train_avgs[key1] = defaultdict(float)
		date_dict = defaultdict(set)
		for key2 in train_count_dict_data[key1]:
			# if train_count_dict_data[key1][key2] >= 50:
			# 	continue
			create_date, create_hour = key2.split(',')
			train_avgs[key1][create_hour] += train_count_dict_data[key1][key2]
			date_dict[create_hour].add(create_date)
		for hour_key2 in train_avgs[key1]:
			train_avgs[key1][hour_key2] = train_avgs[key1][hour_key2] * 1.0 / len(date_dict[hour_key2])
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
				weighted_train_avgs[key1][hour_key2] = current_avg * 0.6 + pre_avg * 0.2 + next_avg * 0.2
		train_avgs = weighted_train_avgs
	return train_avgs

def count_July_mean_data(test_data, July_avgs):
	mean_res = []
	for each_data in test_data:
		start_geo_id = each_data[1]
		end_geo_id = each_data[2]
		key1 = '%s,%s' % (start_geo_id, end_geo_id)
		create_date, create_hour = each_data[3], each_data[4]
		_Ys = 0.0
		if key1 in July_avgs:
			hour_key2 = '%02d' % (int(create_hour))
			if hour_key2 in July_avgs[key1]:
				_Ys = July_avgs[key1][hour_key2]
		mean_res.append(_Ys)
	return mean_res

def main():
	July_data = data_provider.read_csv(July_csv_name)
	Aug_data = data_provider.read_csv(Aug_csv_name)
	print('July data:', len(July_data))
	print('Aug_data:', len(Aug_data))
	test_data = data_provider.read_csv(test_csv_name)
	print('test data:', len(test_data))

	July_data.extend(Aug_data)
	July_count_dict_data = count_data(July_data)
	July_avgs = cal_avgs(July_count_dict_data, weighted=False)
	July_mean_res = count_July_mean_data(test_data, July_avgs)

	# Aug_count_dict_data = count_data(Aug_data)
	Aug_count_dict_data = July_count_dict_data
	Aug_mean_res = count_Aug_mean_data(test_data, Aug_count_dict_data)

	# assert len(July_mean_res) == len(Aug_mean_res)
	# result = [np.round(July_mean_res[i] * 0.6 + Aug_mean_res[i] * 0.4) for i in range(len(July_mean_res))]
	# result = [Aug_mean_res[i] for i in range(len(Aug_mean_res))]
	result = [np.round(Aug_mean_res[i]) for i in range(len(Aug_mean_res))]

	data_provider.write_csv(result, result_csv_name)

if __name__ == '__main__':
	main()
