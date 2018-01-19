import pandas as pd
import numpy as np

x = pd.read_csv('whole.csv')
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