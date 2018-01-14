#!/usr/bin/env python
# -*- coding: utf-8 -*-

import data_provider
import numpy as np
import math

result_csv_name_1 = 'Results/result_01_12.csv'
result_csv_name_2 = 'Results/archives/1.7354.csv'

def main():
	result1 = data_provider.read_csv(result_csv_name_1)
	result1 = [float(r[1]) for r in result1]
	print(sum(result1))
	print(sum(result1) * 1.0 / len(result1))

	result2 = data_provider.read_csv(result_csv_name_2)
	result2 = [float(r[1]) for r in result2]
	print(sum(result2))
	print(sum(result2) * 1.0 / len(result2))

	assert len(result1) == len(result2)
	cc = 0
	for i in xrange(len(result1)):
		if result1[i] < result2[i]:
		# if result1[i] != result2[i]:
			print(i, result1[i], result2[i])
			cc += 1
	print(cc)

if __name__ == '__main__':
	main()
