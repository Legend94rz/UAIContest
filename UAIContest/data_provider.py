#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv

def read_csv(csv_name):
	res = []
	with open(csv_name, 'rb') as csvfile:
		csv_reader = csv.reader(csvfile)
		cc = 0
		for row in csv_reader:
			if cc == 0:
				cc += 1
				continue
			res.append(row)
	return res

def write_csv(res, csv_name):
	with open(csv_name, 'wb') as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow(['test_id', 'count'])
		for i, row in enumerate(res):
			csv_writer.writerow([i, row])

def read_poi_csv(csv_name):
	res = []
	with open(csv_name, 'rb') as csvfile:
		csv_reader = csv.reader(csvfile)
		for row in csv_reader:
			res.append(row)
	return res

def write_list2csv(res, csv_name):
	with open(csv_name, 'wb') as csvfile:
		csv_writer = csv.writer(csvfile)
		csv_writer.writerow(['start_geo_id', 'end_geo_id', 'train_data_count', 'val_data_count'])
		for item in res:
			csv_writer.writerow([item[0], item[1], item[2], item[3]])
