import pandas as pd 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


filename='dataset/milestone1.csv'
d=pd.read_csv(filename)
d.head()


le=preprocessing.LabelEncoder()

d['sex']=le.fit_transform(d['sex'])
d['province']=le.fit_transform(d['province'])
d['country']=le.fit_transform(d['country'])
d['outcome']=le.fit_transform(d['outcome'])

cols=[col for col in d.columns if col not in ['additional_information','source','date_confirmation','outcome']]
data=d[cols]
target=d['outcome']



data_train, data_test, target_train, target_test=train_test_split(data,target,test_size=0.2,random_state=10)




gnb=GaussianNB()


pred=gnb.fit(data_train,target_train)

val=pred.predict(data_train)
val2=pred.predict(data_test)


pickle.dump(pred,open('models/naivebayer_classifier.pkl','wb'))
lmodel=pickle.load(open('models/naivebayer_classifier.pkl','rb'))




print("Training Data results on Naive Bayes Classifier")
print("Accuracy %:",lmodel.score(data_train,target_train)*100)
t=pd.crosstab(pd.Series(le.inverse_transform(target_train)), pd.Series(le.inverse_transform(val)), rownames=['True'], colnames=['Predicted'], margins=True)
print(t)

rec=t['All'][3]
de=t['All'][0]
hp=t['All'][1]
nhp=t['All'][2]

print("deceased cases accuracy %:",round((t['deceased'][0]/de)*100,2))
print("hospitalized cases accuracy %:",round((t['hospitalized'][1]/hp)*100,2))
print("nonhospitalized cases accuracy %:",round((t['nonhospitalized'][2]/nhp)*100,2))
print("recovered cases accuracy %:",round((t['recovered'][3]/rec)*100,2))
print("r2 score:",r2_score(target_train,val))

print("Testing Data results on Naive Bayes Classifier")
print("Accuracy %:",lmodel.score(data_test,target_test)*100)
t1=pd.crosstab(pd.Series(le.inverse_transform(target_test)), pd.Series(le.inverse_transform(val2)), rownames=['True'], colnames=['Predicted'], margins=True)
print(t1)
rec=t1['All'][3]
de=t1['All'][0]
hp=t1['All'][1]
nhp=t1['All'][2]

print("deceased cases accuracy %:",(t1['deceased'][0]/de)*100)
print("hospitaized cases accuracy %:",(t1['hospitalized'][1]/hp)*100)
print("nonhospitalized cases accuracy %:",(t1['nonhospitalized'][2]/nhp)*100)
print("recovered cases accuracy %:",(t1['recovered'][3]/rec)*100)
print("r2 score:",r2_score(target_test,val2))



#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(data_train, target_train)
# predict the response

val=neigh.predict(data_train)
val2=neigh.predict(data_test)


pickle.dump(neigh,open('models/Kneighbors_classifier.pkl','wb'))
lmodel=pickle.load(open('models/Kneighbors_classifier.pkl','rb'))




print("Training Data results on K neighbors Classifier")
print("Accuracy %:",lmodel.score(data_train,target_train)*100)
t=pd.crosstab(pd.Series(le.inverse_transform(target_train)), pd.Series(le.inverse_transform(val)), rownames=['True'], colnames=['Predicted'], margins=True)
print(t)

rec=t['All'][3]
de=t['All'][0]
hp=t['All'][1]
nhp=t['All'][2]

print("deceased cases accuracy %:",round((t['deceased'][0]/de)*100,2))
print("hospitalized cases accuracy %:",round((t['hospitalized'][1]/hp)*100,2))
print("nonhospitalized cases accuracy %:",round((t['nonhospitalized'][2]/nhp)*100,2))
print("recovered cases accuracy %:",round((t['recovered'][3]/rec)*100,2))

print("r2 score:",r2_score(target_train,val))

print("Testing Data results on K neighbors Classifier")
print("Accuracy %:",lmodel.score(data_test,target_test)*100)
t1=pd.crosstab(pd.Series(le.inverse_transform(target_test)), pd.Series(le.inverse_transform(val2)), rownames=['True'], colnames=['Predicted'], margins=True)
print(t1)
rec=t1['All'][3]
de=t1['All'][0]
hp=t1['All'][1]
nhp=t1['All'][2]

print("deceased cases accuracy %:",(t1['deceased'][0]/de)*100)
print("hospitaized cases accuracy %:",(t1['hospitalized'][1]/hp)*100)
print("nonhospitalized cases accuracy %:",(t1['nonhospitalized'][2]/nhp)*100)
print("recovered cases accuracy %:",(t1['recovered'][3]/rec)*100)

print("r2 score:",r2_score(target_test,val2))


#Ada Boost
abc_clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)
pred_abc = abc_clf.fit(data_train, target_train)

val_abc = pred_abc.predict(data_train)
val2_abc = pred_abc.predict(data_test)


pickle.dump(pred_abc,open('models/abc_classifier.pkl','wb'))
lmodel_abc = pickle.load(open('models/abc_classifier.pkl','rb'))

print("Training Data results on AdaBoost")
print("Accuracy %:",lmodel_abc.score(data_train,target_train)*100)
t=pd.crosstab(pd.Series(le.inverse_transform(target_train)), pd.Series(le.inverse_transform(val_abc)), rownames=['True'], colnames=['Predicted'], margins=True)
print(t)

rec=t['All'][3]
de=t['All'][0]
hp=t['All'][1]
nhp=t['All'][2]

print("deceased cases accuracy %:",round((t['deceased'][0]/de)*100,2))
print("hospitalized cases accuracy %:",round((t['hospitalized'][1]/hp)*100,2))
print("nonhospitalized cases accuracy %:",round((t['nonhospitalized'][2]/nhp)*100,2))
print("recovered cases accuracy %:",round((t['recovered'][3]/rec)*100,2))
print("r2 score:",r2_score(target_train,val_abc))

print("Testing Data results on AdaBoost")
print("Accuracy %:",lmodel_abc.score(data_test,target_test)*100)
t1=pd.crosstab(pd.Series(le.inverse_transform(target_test)), pd.Series(le.inverse_transform(val2_abc)), rownames=['True'], colnames=['Predicted'], margins=True)
print(t1)
rec=t1['All'][3]
de=t1['All'][0]
hp=t1['All'][1]
nhp=t1['All'][2]

print("deceased cases accuracy %:",(t1['deceased'][0]/de)*100)
print("hospitaized cases accuracy %:",(t1['hospitalized'][1]/hp)*100)
print("nonhospitalized cases accuracy %:",(t1['nonhospitalized'][2]/nhp)*100)
print("recovered cases accuracy %:",(t1['recovered'][3]/rec)*100)
print("r2 score:",r2_score(target_test,val2_abc))


