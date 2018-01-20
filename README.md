# UAI数据大赛 PALM队 模型说明
## 概述
我们集成了5个模型，为方便起见，5个模型的输出放在Data文件夹中，包括：

1. gbr_300est_300dep_256min_useEstimate_nomerge.csv
1. global_result_01_15_private5k_count_delta_prev.csv
1. global_result_01_14_private5k_1_664.csv
1. global_result_01_15_private5k_count_delta_next.csv
1. Aug_mean_private5k.csv

下面分别说明。

## 模型1
对应输出文件是gbr_300est_300dep_256min_useEstimate_nomerge，训练集使用7、8月份所有已知数据构建，测试集使用给定的public或private集构建。

特征：

1. 起讫点poi总数
1. 最近距离
1. 周几
1. 距离2017-6-30的天数
1. 时刻
1. 当前时刻的体感温度
1. 当前时刻的湿度
1. 未来半小时的湿度
1. 在当前时刻的历史均值（排除8月份未知的数据）
1. 在当前时刻、星期几的历史均值（排除8月份未知的数据）

回归目标：

ceil(0.6\*当前时刻历史均值+0.2\*(前一小时历史均值+后一小时历史均值)) - 当前时刻的订单总数


模型及参数设置：

```
sklearn.ensemble.GradientBoostingRegressor(loss='lad',n_estimators = 300,max_depth = 300, learning_rate = 0.1, verbose = 2, min_samples_leaf = 256, min_samples_split = 256)
```

## 模型2
对应输出文件是global_result_01_15_private5k_count_delta_prev.csv。训练集仅采用7月份数据，测试集采用public或private集构建。

特征：

1. 距2017-7-1的天数
2. 周几
3. 时刻
4. 前一个小时的订单总数
5. 前3小时，当前小时，后三个小时下不下雨的7维one hot编码特征
6. 地点按照poi特征使用KMeans算法聚为6类，取起讫点的簇编号

模型2回归目标：
当前时刻订单总数 - 前一小时订单总数

模型及参数设置：
```{python}
sklearn.ensemble.GradientBoostingRegressor(loss='lad', n_estimators=400, max_depth=350, learning_rate=0.1, min_samples_leaf=160, min_samples_split=160, random_state=1024)
```

## 模型3
对应输出文件是global_result_01_14_private5k_1_664.csv。训练集采用7、8月份的全部数据。测试集采用public或private集构建。

特征：

1. 距2017-7-1的天数
2. 周几
3. 时刻
4. 当前时刻的历史均值
5. 前3小时，当前小时，后三个小时下不下雨的7维one hot编码特征
6. 地点按照poi特征使用KMeans算法聚为6类，取起讫点的簇编号

模型3回归目标：
当前时刻订单总数 - 当前时刻历史均值

模型及参数设置：
```{python}
sklearn.ensemble.GradientBoostingRegressor(loss='lad', n_estimators=400, max_depth=350, learning_rate=0.1, min_samples_leaf=128, min_samples_split=128, random_state=1024)
```

## 模型4
对应输出文件是global_result_01_15_private5k_count_delta_next.csv。训练集仅采用7月份数据，测试集采用public或private集构建。

特征：

1. 距2017-7-1的天数
2. 周几
3. 时刻
4. 后一小时的订单总数
5. 前3小时，当前小时，后三个小时下不下雨的7维one hot编码特征
6. 地点按照poi特征使用KMeans算法聚为6类，取起讫点的簇编号

模型4回归目标：
当前时刻订单总数 - 后一小时订单总数

模型及参数设置：
```{python}
sklearn.ensemble.GradientBoostingRegressor(loss='lad', n_estimators=400, max_depth=350, learning_rate=0.1, min_samples_leaf=160, min_samples_split=160, random_state=1024)
```

## 模型5
对应输出文件是Aug_mean_private5k.csv。它保存了public或private集中，每个样本对应时刻的前一小时、后一小时的平均值。

## 集成方式
单独采用模型2和模型5可得到结果为1.664左右的结果。

我们在1.664的结果的基础上改进。为后文叙述方便，定义变量：
* a=mean(模型2、3、4、5输出)
* b=1.664的结果
* c=ceil(a)

集成的结果按照如下步骤生成：
1. 将模型1输出为0，且a<=1的样本，预测结果置为0
2. 如果c-b>0，则该样本的预测结果取c；否则取b

该集成方法的脚本为addZeros.py，需要的whole.csv文件由模型1~5的输出、1.664的结果手工构建。

最终，可得到1.655的结果。

备注：因为sklearn中聚类以及GBDT算法训练有一定的随机性，复现结果不一定会完全相同，但是基本不会差太多。

谢谢