import pandas as pd
import numpy as np
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer,VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#5、特征处理
#(1)对交互行为进行编
#构造训练集、
dataDay = pd.read_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\dataDay.csv",\
                      usecols=['time_day','user_id','item_id','type_1','type_2','type_3','type_4'],index_col='time_day')

datax = dataDay.loc[['2014-12-13','2014-12-14','2014-12-15','2014-12-16','2014-12-17'],:].groupby(['user_id','item_id'],as_index=False).sum()
datay = dataDay.loc['2014-12-18',['user_id','item_id','type_4']].groupby(['user_id','item_id'],as_index=False).sum()
data_table = pd.merge(datax,datay,on=['user_id','item_id'],suffixes=('_x','_y'),how='left').fillna(0.0)
data_table['labels'] = data_table.type_4_y.map(lambda x:1.0 if x>0.0 else 0.0)
data_table.to_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\data.csv")

predict_data = dataDay.loc[['2014-12-15','2014-12-16','2014-12-17','2014-12-18'],:].groupby(['user_id','item_id'],as_index=False).sum()
predict_data.to_csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\predictdata1111.csv")

sc = SparkContext()
hiveCtx = HiveContext(sc)
data_table = hiveCtx.read.csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\data1_20.csv",\
                               header=True)
data_table.createOrReplaceTempView('data_view')

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


sql = 'select user_id,item_id,type_1,type_2,type_3,type_4_x,labels as label\
,type_1*1 as t1,type_2*2 as t2,type_3*2 as t3,type_4_x*1 as t4\
,(type_1*1+type_2*2+type_3*3-type_4_x*1) as t5\
,(type_1+1)/(type_1+type_2+type_3+type_4_x+1) as t6\
,(type_2+1)/(type_1+type_2+type_3+type_4_x+1) as t7\
,(type_3+1)/(type_1+type_2+type_3+type_4_x+1) as t8\
,(type_4_x+1)/(type_1+type_2+type_3+type_4_x+1) as t9 from data_view where type_4_x <20 '
data = hiveCtx.sql(sql)
data.registerTempTable('data1')

#训练模型
assem = VectorAssembler(inputCols=['t1','t2','t3','t4','t5','t6','t7','t8','t9'],outputCol='features')
output_data = assem.transform(data.na.drop())
data1 = output_data.select('label','features')

labelIndexer = StringIndexer(inputCol='label',outputCol='indexedLabel').fit(data1)
featureIndexer = VectorIndexer(inputCol='features',outputCol='indexedFeature',handleInvalid='skip').fit(data1)
(trainingmat,testingmat) = data1.randomSplit([0.7,0.3])
gbdt = GBTClassifier(maxIter=100,maxDepth=6,labelCol='indexedLabel',featuresCol='indexedFeature',seed=42)
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


predict_data = hiveCtx.read.csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\predictdata1111.csv",\
                                header=True)
predict_data.registerTempTable('predict_view')
sql3 = 'select user_id,item_id,type_4\
,type_1*1 as t1,type_2*2 as t2,type_3*2 as t3,type_4*1 as t4\
,(type_1*1+type_2*2+type_3*3-type_4*1) as t5\
,(type_1+1)/(type_1+type_2+type_3+type_4+1) as t6\
,(type_2+1)/(type_1+type_2+type_3+type_4+1) as t7\
,(type_3+1)/(type_1+type_2+type_3+type_4+1) as t8\
,(type_4+1)/(type_1+type_2+type_3+type_4+1) as t9 from predict_view where type_4 <5 '
predict_data = hiveCtx.sql(sql3)

output_predict = assem.transform(predict_data.na.drop())
predict_data = output_predict.select('user_id','item_id','type_4','features')
prediction = model.transform(predict_data)
result = prediction.select('user_id','item_id','type_4','prediction')
result.registerTempTable('result1')
out_data = hiveCtx.sql('select user_id,item_id from result1 where prediction>0.0')
out_data.write.csv(r"E:\pycharm\pycharm_script\kaggle\fresh_comp_offline\fresh_comp_offline\test_test1_20.csv")
