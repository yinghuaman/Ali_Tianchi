import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
#1、读取用户行为数据并进行处理
user_all = pd.read_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\tianchi_fresh_comp_train_user.csv",\
                       usecols = ['user_id','item_id','behavior_type','time'])
user_all = user_all.drop_duplicates()  #去除重复行
#print(user_all.info())

#2、读取商品子集数据并进行处理
item_all = pd.read_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\tianchi_fresh_comp_train_item.csv",\
                       usecols = ['item_id'])
item_all = item_all.drop_duplicates('item_id')

#print(item_all.info())

#3、预测在商品子集上进行，所以可以只考虑用户在商品子集上的交互数据
user_sub = pd.merge(user_all,item_all,on = ['item_id'],how = 'inner')
#print(user_sub.head())
user_sub.to_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\user_sub.csv")
#4、处理时间
user_sub = pd.read_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\user_sub.csv",parse_dates = True)
#user_sub = user_sub.sort_index().copy()
user_sub = user_sub.sort_index()

#print(user_sub.head())

#5、特征处理
#(1)对交互行为进行编码
typeDummies = pd.get_dummies(user_sub['behavior_type'],prefix = 'type')    #onehot哑变量编码
usersub_OneHot = pd.concat([user_sub[['user_id','item_id','time']],\
                           typeDummies],axis = 1)
usersub_OneHotgroup = usersub_OneHot.groupby(['user_id','item_id','time'],\
                                 as_index = False).sum()

usersub_OneHotgroup['time_day'] = pd.to_datetime(usersub_OneHotgroup['time'].values).date
#print(usersub_OneHotgroup.head())
dataDay = usersub_OneHotgroup.groupby(['time_day','user_id','item_id']).sum()
dataDay.to_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\dataDay.csv")

#6、构造训练集和测试集
data = pd.read_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\dataDay.csv",index_col = 'time_day')
train_x2 = data.loc[['2014-12-13','2014-12-14','2014-12-15'],:]#13、14、15号数据选作训练数据集
train_y2 = data.loc['2014-12-16',['user_id','item_id','type_4']]  #16号购买数据作为类别标签
train_x1 = data.loc[['2014-12-14','2014-12-15','2014-12-16'],:]#14、15、16号数据选作特征数据集
train_y1 = data.loc['2014-12-17',['user_id','item_id','type_4']]  #17号购买数据作为类别标签
trainx_mat = pd.concat([train_x1,train_x2])
trainy_mat = pd.concat([train_y1,train_y2])
traindata = pd.merge(trainx_mat,trainy_mat,on = ['user_id','item_id'],suffixes = ('_x','_y'),how = 'left')
traindata['labels'] = traindata.type_4_y.map(lambda x:1.0 if x>0.0 else 0.0)
trainset = traindata.copy()
trainset.to_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\traindata.csv")

test_x = data.loc[['2014-12-15','2014-12-16','2014-12-17'],:]#15、16、17号数据选作测试数据集
test_y = data.loc['2014-12-18',['user_id','item_id','type_4']]  #18号购买数据作为测试类别标签
testdata = pd.merge(test_x,test_y,on = ['user_id','item_id'],suffixes = ('_x','_y'),how = 'left')
testdata['labels'] = testdata.type_4_y.map(lambda x:1.0 if x>0.0 else 0.0)
print(testdata['labels'].sum())
testset = testdata.copy()
testset.to_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\testdata.csv")

#7、训练模型
traindata = pd.read_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\traindata.csv")
gbdt = GradientBoostingClassifier(random_state = 10)
gbdt.fit(traindata.iloc[:,2:6],traindata.iloc[:,-1])
precision_train = cross_val_score(gbdt,traindata.iloc[:,2:6],traindata.iloc[:,-1],cv = 5,scoring = 'precision')
recall_train = cross_val_score(gbdt,traindata.iloc[:,2:6],traindata.iloc[:,-1],cv = 5,scoring = 'recall')
f1_train = cross_val_score(gbdt,traindata.iloc[:,2:6],traindata.iloc[:,-1],cv = 5,scoring = 'f1')
print(np.mean(precision_train),np.mean(recall_train),np.mean(f1_train))
print(gbdt.score(traindata.iloc[:,2:6],traindata.iloc[:,-1]))

testdata = pd.read_csv(r"E:\kaggle\fresh_comp_offline\fresh_comp_offline\testdata.csv")
precision_test = cross_val_score(gbdt,testdata.iloc[:,2:6],testdata.iloc[:,-1],cv = 5,scoring = 'precision')
print(np.mean(precision_test))
recall_test = cross_val_score(gbdt,testdata.iloc[:,2:6],testdata.iloc[:,-1],cv = 5,scoring = 'recall')
print(np.mean(recall_test))
f1_test = cross_val_score(gbdt,testdata.iloc[:,2:6],testdata.iloc[:,-1],cv = 5,scoring = 'f1')

print("f1得分:\n",np.mean(f1_test))


#8、对19号购买情况进行预测
predict_x = data.loc[['2014-12-16','2014-12-17','2014-12-18'],:]
predict_y = gbdt.predict(predict_x.iloc[:,2:])

user_item_19 = predict_y.loc[predict_y > 0.0,['user_id','item_id']]
print(user_item_19.info())












