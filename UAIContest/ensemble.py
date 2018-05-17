#!/usr/bin/env python
# -*- coding: utf-8 -*-

import data_provider
import numpy as np
import math

model_result_csv_name = 'global_result_01_15_private5k.csv'
July_Aug_mean_csv_name = 'Aug_mean_private5k.csv'
# July_Aug_mean_csv_name = 'archives/w.csv'
result_csv_name = 'result_01_15_private5k.csv'

def main():
	model_result = data_provider.read_csv(model_result_csv_name)
	model_result = [float(r[1]) for r in model_result]

	# model_result2 = data_provider.read_csv(model_result_csv_name2)
	# model_result2 = [float(r[1]) for r in model_result2]
	
	July_Aug_mean_res = data_provider.read_csv(July_Aug_mean_csv_name)
	July_Aug_mean_res = [float(r[1]) for r in July_Aug_mean_res]
	
	assert len(model_result) == len(July_Aug_mean_res)
	result = [np.ceil(model_result[i] * 0.6 + July_Aug_mean_res[i] * 0.4) for i in range(len(model_result))]

	# assert len(model_result) == len(July_Aug_mean_res) and len(model_result2) == len(model_result)
	# result = [model_result[i] * 0.5 + model_result2[i] * 0.2 + July_Aug_mean_res[i] * 0.3 for i in range(len(model_result))]
	
	print(sum(result) * 1.0)
	print(sum(result) * 1.0 / len(result))
	data_provider.write_csv(result, result_csv_name)

if __name__ == '__main__':
	main()
