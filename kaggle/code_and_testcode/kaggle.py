import pandas as pd
import numpy as np
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer,VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#1、读取用户行为数据并进行处理
user_all = pd.read_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\tianchi_fresh_comp_train_user.csv",\
                       usecols = ['user_id','item_id','behavior_type','time'])

#2、读取商品子集数据并进行处理
item_all = pd.read_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\tianchi_fresh_comp_train_item.csv",\
                       usecols = ['item_id'])
item_all = item_all.drop_duplicates('item_id')

#3、预测在商品子集上进行，所以可以只考虑用户在商品子集上的交互数据
user_sub = pd.merge(user_all,item_all,on = ['item_id'],how = 'inner')
user_sub.to_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\user_sub.csv",index=False)

#5、特征处理
#(1)对交互行为进行编码
user_sub = pd.read_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\user_sub.csv",parse_dates=True,\
                       usecols=['user_id','item_id','behavior_type','time'])
typeDummies1 = pd.get_dummies(user_sub['behavior_type'],prefix = 'type')    #onehot哑变量编码
usersub_OneHot = pd.concat([user_sub[['time','user_id','item_id']],typeDummies1],axis = 1)
usersub_OneHot['time_day'] = pd.to_datetime(usersub_OneHot.time.values).date
dataDay = usersub_OneHot.groupby(['time_day','user_id','item_id'],as_index = False).sum()
dataDay.to_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\dataDay.csv",index=False)

#构造训练集、
dataDay = pd.read_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\dataDay.csv",\
                      usecols=['time_day','user_id','item_id','type_1','type_2','type_3','type_4'],index_col='time_day')

train_x = dataDay.loc[['2014-12-14','2014-12-15','2014-12-16'],:].groupby(['user_id','item_id'],as_index=False).sum()
train_y = dataDay.loc['2014-12-17',['user_id','item_id','type_4']].groupby(['user_id','item_id'],as_index=False).sum()
train_table = pd.merge(train_x,train_y,on=['user_id','item_id'],suffixes=('_x','_y'),how='left').fillna(0.0)
train_table['labels'] = train_table.type_4_y.map(lambda x:1.0 if x>0.0 else 0.0)
train_table.to_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\traindata.csv",index=False)

test_x = dataDay.loc[['2014-12-15','2014-12-16','2014-12-17'],:].groupby(['user_id','item_id'],as_index=False).sum()
test_y = dataDay.loc['2014-12-18',['user_id','item_id','type_4']].groupby(['user_id','item_id'],as_index=False).sum()
test_table = pd.merge(train_x,train_y,on=['user_id','item_id'],suffixes=('_x','_y'),how='left').fillna(0.0)
test_table['labels'] = train_table.type_4_y.map(lambda x:1.0 if x>0.0 else 0.0)
test_table.to_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\testdata.csv",index=False)

predict_data = dataDay.loc[['2014-12-16','2014-12-17','2014-12-18'],:].groupby(['user_id','item_id'],as_index=False).sum()
predict_data.to_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\predictdata.csv")

sc = SparkContext()
hiveCtx = HiveContext(sc)
train_table = hiveCtx.read.csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\train1_10.csv",\
                               header=True)
train_table.createOrReplaceTempView('train_view')
test_table = hiveCtx.read.csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\test1_10.csv",\
                              header=True)
test_table.createOrReplaceTempView('test_view')

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


sql = 'select user_id,item_id,type_1,type_2,type_3,type_4_x,labels as label\
,type_1*1 as t1,type_2*2 as t2,type_3*2 as t3,type_4_x*1 as t4\
,(type_1*1+type_2*2+type_3*3-type_4_x*1) as t5\
,(type_1+1)/(type_1+type_2+type_3+type_4_x+1) as t6\
,(type_2+1)/(type_1+type_2+type_3+type_4_x+1) as t7\
,(type_3+1)/(type_1+type_2+type_3+type_4_x+1) as t8\
,(type_4_x+1)/(type_1+type_2+type_3+type_4_x+1) as t9 from test_view where type_4_x <20 '
train_data = hiveCtx.sql(sql)
train_data.registerTempTable('train_data1')

sql2 = 'select user_id,item_id,type_1,type_2,type_3,type_4_x,labels as label\
,type_1*1 as t1,type_2*2 as t2,type_3*2 as t3,type_4_x*1 as t4\
,(type_1*1+type_2*2+type_3*3-type_4_x*1) as t5\
,(type_1+1)/(type_1+type_2+type_3+type_4_x+1) as t6\
,(type_2+1)/(type_1+type_2+type_3+type_4_x+1) as t7\
,(type_3+1)/(type_1+type_2+type_3+type_4_x+1) as t8\
,(type_4_x+1)/(type_1+type_2+type_3+type_4_x+1) as t9 from test_view where type_4_x <20 '
test_data = hiveCtx.sql(sql2)
test_data.registerTempTable('test_data1')

data = hiveCtx.sql('select * from train_data1 union all (select * from test_data1)')
#训练模型
assem = VectorAssembler(inputCols=['t1','t2','t3','t4','t5','t6','t7','t8','t9'],outputCol='features')
output_data = assem.transform(data.na.drop())
data1 = output_data.select('label','features')

labelIndexer = StringIndexer(inputCol='label',outputCol='indexedLabel').fit(data1)
featureIndexer = VectorIndexer(inputCol='features',outputCol='indexedFeature',handleInvalid='skip').fit(data1)
(trainingmat,testingmat) = data1.randomSplit([0.7,0.3])
gbdt = GBTClassifier(maxIter=30,maxDepth=3,labelCol='indexedLabel',featuresCol='indexedFeature',seed=42)
pipeline = Pipeline(stages=[labelIndexer,featureIndexer,gbdt])
model = pipeline.fit(trainingmat)
predictions = model.transform(testingmat)

evaluator = MulticlassClassificationEvaluator(labelCol='indexedLabel',predictionCol='prediction',metricName='f1')
f1 = evaluator.evaluate(predictions)
print('test f1:%f'%f1)
evaluator = MulticlassClassificationEvaluator(labelCol='indexedLabel',predictionCol='prediction',metricName='weightedPrecision')
weightedPrecision = evaluator.evaluate(predictions)
print('test weightedPrecision:%f'%weightedPrecision)
evaluator = MulticlassClassificationEvaluator(labelCol='indexedLabel',predictionCol='prediction',metricName='weightedRecall')
weightedRecall = evaluator.evaluate(predictions)
print('test weightedRecall:%f'%weightedRecall)

predict_data = hiveCtx.read.csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\predictdata.csv",\
                                header=True)
predict_data.registerTempTable('predict_view')
sql3 = 'select user_id,item_id,type_4\
,type_1*1 as t1,type_2*2 as t2,type_3*2 as t3,type_4*1 as t4\
,(type_1*1+type_2*2+type_3*3-type_4*1) as t5\
,(type_1+1)/(type_1+type_2+type_3+type_4+1) as t6\
,(type_2+1)/(type_1+type_2+type_3+type_4+1) as t7\
,(type_3+1)/(type_1+type_2+type_3+type_4+1) as t8\
,(type_4+1)/(type_1+type_2+type_3+type_4+1) as t9 from predict_view where type_4 <5  '
predict_data = hiveCtx.sql(sql3)

output_predict = assem.transform(predict_data.na.drop())
predict_data = output_predict.select('user_id','item_id','type_4','features')
prediction = model.transform(predict_data)
result = prediction.select('user_id','item_id','type_4','prediction')
result.registerTempTable('result1')
out_data = hiveCtx.sql('select user_id,item_id from result1 where prediction>0.0')
out_data.write.csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\outdata1_10.csv")














