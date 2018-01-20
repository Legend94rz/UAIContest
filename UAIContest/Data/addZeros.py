import pandas as pd
import numpy as np
x = pd.DataFrame()
x['test_id'] = range(5000)

t = pd.read_csv('gbr_300est_300dep_256min_useEstimate_nomerge.csv')
x['mygbr'] = t['count']
t = pd.read_csv('global_result_01_14_private5k_1_664.csv')
x['cur'] = t['count']
t = pd.read_csv('global_result_01_15_private5k_count_delta_next.csv')
x['next'] = t['count']
t = pd.read_csv('global_result_01_15_private5k_count_delta_prev.csv')
x['prev'] = t['count']
t = pd.read_csv('private5k_1_664.csv')
x['std1664'] =t['count']
t = pd.read_csv('Aug_mean_private5k.csv')
x['augmean']=t['count']
x.to_csv('whole.csv')

#x = pd.read_csv('whole.csv')
result = pd.DataFrame()
result['test_id'] = range(5000)

count = []

for i in range(len(x)):
	a = np.ceil( x.loc[i,['prev','next','cur','augmean']].mean() )
	b = x.loc[i,'std1664']
	if x.loc[i,'mygbr']==0 and (x.loc[i,['prev','next','cur','augmean']].values<=1).all():
		count.append(0)
	#elif  a - b >= 4 or a-b <= -7 :
	elif a - b > 0:
		count.append( a )
	else:
		count.append( x.loc[i,'std1664'])

result['count']=count

result.to_csv('private_has_zero.csv',index = False)