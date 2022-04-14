#objectif mesurable #=============================#===============================
- classification 'y' : 0/1
- Precision Score : (Specificity) (TP / float(TP + FP)) ==1 if FP=0 
- Recall Score : (Sensitivity) (TP / float(TP + FN)) ==1 if FN=0 
- F1= Precision/Recall
#----------------------------------------------------
'Cleaning data :
    - Creat subset data :
index = Data.isna().sum(axis=0)/Data.shape[0]          #get array[sum(True, false] in each column of datafram[True, false]
index_na=(index >0.2 & index <0.9)                     #get 0.2<array[True, False]<0.9 in order to detect column nan
group_columns = Data.columns[index_na]                 #list of column of empeting 
key_columns = ['column_A','column_B']
Data = Data[key_columns + group_columns]
print(Data.head())
'Train/Test :
Data_train , Data_test = train_test_split(Data, test_size=0.2, random_state=0)      #Data == X , y
plt.pie((Data_train.shape[0],Data_test.shape[0]), labels=('Data_train', 'Data_test'), autopct='%1.1f%%')    #chek if train_size==1-0.2 of all rows
Data_train['column_y'].value_counts().plot.pie(autopct='%1.1f%%')     #chek train['column_y'] class equilibre or not
Data_test['column_y'].value_counts().plot.pie(autopct='%1.1f%%')      #chek test['column_y'] class equilibre or not
or :
# KFold(n_splits=4).get_n_splits(X)
# for train_index, test_index in KFold(n_splits=4, random_state=44, shuffle=False).split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     print("%s %s" % (train_index, test_index)
#     print(Data_train, Data_test)
#     Data_train, Data_test = Data[train_index], Data[test_index]
'Encodage :
def imput(data):
    data = data.dropna(axis=1)                         #Eliminate columns nana or null
or  index_0 = Data.isnull().sum(axis=0) > 1            #get 0<array[True, False] of [sum(True, false] in order to detect columns nan or 0 of datafram[True, false]_isna or isnull
    group_columns = Data.columns[index_0]              #list of column of empeting
    data = data.drop(group_columns,1)
    or
    data = data.dropna(axis=0)                         #Eliminate rows nana or null
or  index_0 = (data==0).sum(axis=1) > 1                #get 0<array[True, False] of [sum(True, false] in order to detect rows nan or 0 of datafram[True, false]_isna or isnull
    group_rows = data.index[index_0]                    #list of column of empeting
    data = data.drop(group_rows,0)
    or  
    data = data.fillna(-999)                    #replace all val_nan by -999 for Giving more data to eliminate overfitting
or  data = data.fillna(data.mean())            #replace float val_nan by data.mean()
    data = data.fillna('m')                    #replace object val_nan by 'm' for Giving more data to eliminate overfitting
    or
    data['is na'] = (data['column_A'].isna())||(data['column_B'].isna())         #if col A or B==nan add new column[True,False] in data['is na'] in order to detect missing nan for new categorie(membr of equipage) 
    data = data.fillna(-999)      

    data.isnull().sum().max()               #just checking that there's no missing data of nan or null()
    return data
def encodage(data):
    print(data['column_A'].unique())                                                #for analyse how many object
    for col in data.select_dtypes('object'):
        data[col]=data[col].map({'object_A' : 1,'object_B' : 0, })                  #creat encodage
    or 
    data = pd.get_dummies(data)                                                 #for small data encodage with new columns and without geting the sorted columns
    or 
    from sklearn.Preprocessing import LabelEncoder                                  #for big data encodage label
    cod_dict = {}
    for col in data.select_dtypes('object'):
        cod_dict[col] = LabelEncoder()
        print(cod_dict.keys())
        print('classed found : ' , list(cod_dict[col].fit(data[col]).classes_))
        data[col] = cod_dict[col].fit_transform(data[col])                          #Create a label (category) encoder object
        print('Updates dataframe is : \n' ,data )
    # data['score0'] = cod_dict['score0'].inverse_transform(data['score0'])         #for inverse transform not important
    # print('Inverse Transform  : \n' ,data)
    or
    from sklearn.Preprocessing import OneHotEncoder                              #for big data encodage label with new columns
    for col in data.select_dtypes('object'): 
        print(sorted(data[col].unique()))                                         #sorted alphabet list ['normal' 'strong' 'weak'] in order to get the good names after
        array = np.array(data[col]).reshape(len(np.array(data[col])), 1)        #must creat array&reshape for OneHotEncoder ['strong' 'weak' 'normal' 'weak' 'strong']
        newmatrix = OneHotEncoder().fit_transform(array).toarray()             #creat with toarry() new array with len(column)==len(data[col].unique())
        for i in range(len(data[col].unique())):
            data[sorted(data[col].unique())[i]] = newmatrix[:,i]            #sorted(data[col].unique())[i]==list.sort()[i]
        data.drop([col],axis=1,inplace=True)                                #drop column transfered
    print('Updates dataframe is : \n' ,data )
    data.dtypes.value_counts().plot.pie(autopct='%1.1f%%')      #to check if elimanate object
    return data
def outlier(data):
    data_out = StandardScaler().fit_transform(data['column_A'][:,np.newaxis])   #standardize the data in order to have mean of 0 and a standard deviation of 1
    index_out = data_out['column_A'].argsort().index             #get index of dataframe of sorted values column_A
    array_out = np.array(data_out.loc[index_out,'column_A'])     #get array of value column_A sorted
    print('low range :\n',array_out[:10])                    #show if 10 Low range values are far from 0.
    print('high range :\n',array_out[-10:])                  #show if 10 High range values are far from 0.
    data = data.drop(index_out[-1:],0)                  #Eliminate last row outlier
    or
    model = IsolationForest(contamination = 0.1).fit(data)    #0.02 ==% of datafilter / we can use random_state=0
    indx = model.predict(data)==1                             #get array_indx(True&False) of(-1anomalie & +1normale)
    sns.heatmap(pd.DataFrame(indx.T,index=data.index))        #get visualisat of data_array[True, False] in order to detect outliers in %
    plt.figure()
    pd.DataFrame(indx.T,index=indx).value_counts().plot.pie(autopct='%1.1f%%',startangle = 90)      #chek % of outlier
    data = data.loc[indx,:]
    return data
'Preprocessing :
def prep(data):
    data = imput(data)
    data = encodage(data)
    data = outlier(data)
    X = data.drop('column_y',axis=1)
    y = data['column_y']
    return X,y
X_train, y_train = prep(Data_train)
X_test, y_test = prep(Data_test)
y_train.value_counts().plot.pie(
    autopct=lambda p:f'{p*sum(y_train.value_counts())/100 :.0f}')      #check nbr of row y_train['column_y'] after elimin nan
y_test.value_counts().plot.pie(
    autopct=lambda p:f'{p*sum(y_test.value_counts())/100 :.0f}')      #check nbr of row y_test['column_y'] after elimin nan
#----------------------------------------------------
#Evaluation - Diagnostiq :
'Modellisation :
(+100000 rows) => Descente Gradient (RL, SGDReg, SGDClassifier, RN'Tensorflow,Torch,)
(-100000 rows) => Normal law of data : 
                yes => parametriq model :
                        - Regression:'sklearn.linear_model: (LineareRegression(), Lasso(), Ridge(),DecisionTreeRegressor()) 
                        - classification: 'sklearn.Tree: (DecisionTreeClassifier(),LinearSVC, logistReg, NaiveBayes, RN)
                                          'sklearn.ensemble: (RandomForestClassifier())
                Non => Visualisation data : or Pre-processing
                        - Linear graph : 'sklearn.neighbors: (KNeighborsClassifier())
                        - categorys : 'sklearn.svm: (SVC(), SVR(),LogisticRegression())
'Evaluation :
from sklearn.tree import DecisionTreeClassifier()
model = DecisionTreeClassifier(random_state=0)
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, classification_report, recall_score, precision_recall_curve
from sklearn.model_selection import learning_curve
def scoring(y_pred):
    # Accuracy : acc=-((y-y_pred)**2/(y-y_mean)**2) describe in % test score
    acc=accuracy_score(y_test, y_pred)
    print(acc)                  
    # Confusion_matrix : best score is with Diagonal matrix               [[TP FP]
    cm = confusion_matrix(y_test, y_pred)                #                 [FN TN]]
    print(cm)                                          #get array_carré[y_test / y_predict]
    sns.heatmap(cm, centre = True)                     #visualisation of result
    # f1 score : f1 = 2*(precision*recall)/(precision +recall) (1/precision+1/recall)==1 if F=0
    f1 = f1_score(y_test, y_pred, average='micro')     #it can be : binary,micro,weighted,samples
    print(f1)
    # Classification report :
    cr = classification_report(y_test,y_pred)
    print(cr)                                          #get (accuracy acc, score f1, cm)
    #Recall Score : (Sensitivity) (TP / float(TP + FN)) ==1 if FN=0 (danger malade)
    rs = recall_score(y_test, y_pred, average=None)
    print(cr)
    #Precision Score : (Specificity) (TP / float(TP + FP)) ==1 if FP=0 
    ps = precision_score(y_test, y_pred, average=None)
    print(cr)
def Learning_Curve(mod):
    N, train_score, val_score = Learning_curve(mod, X_train,y_train, cv = 5, scoring='f1'     #cv == ndr of split(kFold(),LeaveOneOut(),ShuffleSplit(4,size0.2),StratifiedKFold(),GroupKFold(4).get_n_splits(X,y,groups=X[:,0]))
                            train_sizes = np.linspace(0.2, 1,5),random_state=0)     #(0.2, 1 ,5) 5 == nbr of lot(20%,40%,60%,80%,100%) in data train
    plt.figure(figsize=(10,8))
    plt.plot(N, train_score.mean(axis=1), label='tr_score')            #N == nbr of row in data_train [20%,40%,60%,80%,100%]=>[19,38,57,76,95]
    plt.plot(N, val_score.mean(axis=1), label='val_score')             #val_score will stagnate from % of row of data_train => is not in need of data
    plt.legend()
'Fit :
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
'Classification_report:  #overfinting==ecart[(%tr_score=good),(%val_score=bad)] / underfiting==[(%tr_score=bad),(%val_score=good)]
scoring(y_pred)                                     #compare (accuracy acc, score f1, cm)
'Learning_curve:
Learning_Curve(model)                       #tr_score : Decreasess ,increases / val_score : stagnate
#----------------------------------------------------
#Idée(Dropna) - code - Fit - Evaluation
'Idée : 
=> changing in def imput(data): in order to give more data to eliminate overfitting
'Fit : => Classification_report: => Learning_curve:
#----------------------------------------------------
#Idée(feature_important) - code - Fit - Evaluation
'Idée : 
keep only the important features in order to eliminate overfitting
=> choise feature important
'Code :
An : index_imp = model.feature_importances_                            #get array[col_important]
An : data_feat_impor = pd.DataFrame(index_imp,index=X_train.columns)   #creat datfram with columns of X_train
An : data_feat_impor.plot.bar()                                    #visualisat feature_importances_
An : column_imp = data.columns[index_imp >0.01]                      #get list of feature_importances_ that >0.01
    In : 'Encodage :def imput(data)
Add : data = data[resultof(column_imp)]                  #get data of column_important by copy(Result of list index > 0.01[True, False]>0.01)
'Fit : => Classification_report: => Learning_curve:
#----------------------------------------------------
#Idée(Feature Engineering) - code - Fit - Evaluation
'Idée : 
Creat new 'column' in following Relation variables/variables
=> grouped by sum in new variabale
'Code :
    In : 'Encodage :
Add : def Feat_Engin(data):
        group_columns = data.columns[1:3]                     #list of column concerned
        index_newcol = data[group_columns]=='object'          #get data array[True, False]=='object'
        data['sum'] = index_newcol.sum(axis=1)                #put new column'sum' that have array[sum index]
        or data['sum'] = index_newcol.sum(axis=1)>=1          #put new column'sum' that have array[True, False]>=1
        data = data.drop(group_columns,axis=1)                #eliminat column concerned
        return data
    In : 'Preprocessing : def prep(data):
Add : data = Feat_Engin(data)
'Fit : => Classification_report: => Learning_curve:
#----------------------------------------------------
#Idée(complex model) - code - Fit - Evaluation
'Idée : 
Use a model that struggles against overfitting
=> use RandomForestClassifier()
'Code :
    In : 'Evaluation :
Add : from sklearn.ensemble import RandomForestClassifier
Replace : model = DecisionTreeClassifier(random_state=0)
    By : model = RandomForestClassifier(random_state=0)
'Fit : => Classification_report: => Learning_curve:
#----------------------------------------------------
#Idée(Feature Select) - code - Fit - Evaluation
'Idée : 
Use Feature Select test_classificat_ANOVA in order to eliminate overfitting
=> SelectKBest(chi2, k=1) and RandomForestClassifier() by pipline
'Code :
    In : 'Evaluation :
Add : from sklearn.ensemble import RandomForestClassifier
Add : from sklearn.pipeline import make_pipeline
Add : from sklearn.feature_selection import SelectKBest,SelectPercentile,GenericUnivariateSelect,SelectFromModel,chi2, f_classif
An : array_col = SelectKBest(score_func= fclassif ,k=10).fit_transform(X_train,y_train)                                #fclassif , chi2(X,y)/k=10==nbr of features
or An :array_col = SelectPercentile(score_func = chi2, percentile=20).fit_transform(X_train,y_train)                    #select 20% of features
or An :array_col = GenericUnivariateSelect(score_func= chi2, mode= 'k_best', param=3).fit_transform(X_train,y_train)    #mode can = percentile,fpr,fdr,fwe / param=3==nbr of features
or An :array_col = SelectFromModel(estimator ='R(model)', max_features = None).fit_transform(X_train,y_train)           #estimator == LinearRegression() or LogisticRegression() and import estimator (well-defined)
An : print(array_col.get_support())                #get list of [True , False] indexng columns
Replace : model = DecisionTreeClassifier(random_state=0)
    By : model = make_pipeline(SelectKBest(score_func=f_classif, k=10),         #fclassif , chi2(X,y)
                            RandomForestClassifier(random_state=0))           #k=10 nbr of best indexing feature that have lien le plus fort 'column_y'
'Fit : => Classification_report: => Learning_curve:
#----------------------------------------------------
#Idée(Polynomial Feature in underfit :) - code - Fit - Evaluation
'Idée : 
Use Polynomial Feature in order to eliminate overfitting
=> SelectKBest(chi2, k=1) and RandomForestClassifier() and PolynomialFeatures by pipline
'Code :
    In : 'Evaluation :
Add : from sklearn.ensemble import RandomForestClassifier
Add : from sklearn.pipeline import make_pipeline
Add : from sklearn.feature_selection import SelectKBest, f_classif
Add : from sklearn.Preprocessing import PolynomialFeatures
Replace : model = DecisionTreeClassifier(random_state=0)
    By : model = make_pipeline(PolynomialFeatures(2),          #add columns  of features for non Linear prob in order to increase the degree, 3==degree
                            SelectKBest(score_func=f_classif, k=10),         #fclassif , chi2(X,y)
                            RandomForestClassifier(random_state=0))           #k=10 nbr of best indexing feature that have lien le plus fort 'column_y'
'Fit : => Classification_report: => Learning_curve:
#----------------------------------------------------
#Idée(compare List of models & pipline) - code - Fit - Evaluation
'Idée : 
Use of all models in order to eliminate overfitting
=> RadForst, AdBoost, SVM, KNN 
    In : 'Evaluation :
Add : from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
Add : from sklearn.svm import SVC
Add : from sklearn.neighbors import KNeighborsClassifier
Add : from sklearn.pipeline import make_pipeline
Add : from sklearn.feature_selection import SelectKBest, f_classif
Add : from sklearn.Preprocessing import PolynomialFeatures, StandardScaler, #MinMaxScaler,[Normalizer(copy=True, norm=''l1'or'max'')modif suivant row,MaxAbsScaler(copy=True)modif suivant columns,FunctionTransformer(func=fun1 or lambda x: x**0.1,validate = True),Binarizer(threshold = 1.0),
Add : from sklearn.impute import SimpleImputer,KNNImputer
Remove : model = DecisionTreeClassifier(random_state=0)
Add : proce_1 = model_union = make_union(SimpleImputer(missing_values=np.nan, strategy='mean',fill_value=-9)or KNNImputer(n_neighbors=1),   #composed of two transformation in parallel
                        MissingIndicator())                                                                                                 #stategy= mean , median , most_frequent , constant
Add : proce_2 = make_pipeline(PolynomialFeatures(2, include_bias=False),SelectKBest(f_classif, k=10))       
Add : mod_RadForst = make_pipeline(proce_1,proce_2,RandomForestClassifier(random_state=0))    #pas besoin de normalisation
Add : mod_AdBoost = make_pipeline(proce_1,proce_2,AdaBoostClassifier(random_state=0))         #pas besoin de normalisation
Add : mod_SVM = make_pipeline(proce_1,proce_2, StandardScaler(), SVC(random_state=0))         #must use Scaling standard => -1< X_train <1 & mean=0 & std=1
Add : mod_KNN = make_pipeline(proce_1,proce_2, StandardScaler(), KNeighborsClassifier())      #Scaling MinMax => 1< X_train <0   
Add : models = {'RadForst':mod_RadForst,'AdBoost':mod_AdBoost,'SVM':mod_SVM,'KNN':mod_KNN}
for nam, model in models.items:
    print('result of model nam : ' , nam)
    'Fit :
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    'Classification_report:
    scoring(y_pred)                             #compare (accuracy acc, score f1, cm)
    'Learning_curve:
    Learning_Curve(model)                       #tr_score : Decreasess ,increases / val_score : stagnate
or : use crosse_validation_score cv
for nam, model in models.items:
    print(nam)
    for n in range(2,11):
        print('result of model nam : ' , nam ,' for cv value ',n,' is ' , cross_val_score(model, X, y, cv=n))  
        print('-----------------------------------')
#----------------------------------------------------
#Optimisation(hyperparametres) - code - Fit - Evaluation
'GridSearch :
from sklearn.model_selection import GridSearchCV or RandomizedSearchCV
hyper_params = {'svc__gamma': [1e-3,1e-4],          #get params of model_pipe if want
                'svc__C' : [1,10,100,1000],
                'pipeline__polynomialfeatures__degree':[2,3,4],
                'pipeline__selectkbest__k':range(4, 100)}          
grid = GridSearchCV(mod_SVM, hyper_params, scoring='recall',cv=4).fit(X_train,y_train)    #cv=3ou4ou5 == ndr of split
grid = RandomizedSearchCV(mod_SVM, hyper_params, scoring='recall',cv=4, n_iter=40).fit(X_train,y_train)   #n_iter == 40 iteration by chance
print(grid.best_score_)                      #get best score in GridSearchCV
print(grid.best_params_)                     #get best parametres
model_grid = grid.best_estimator_            #get best model
'Fit :
model_grid.fit(X_train,y_train)
y_pred = model_grid.predict(X_test)
'Classification_report:
scoring(y_pred)                             #compare (accuracy acc, score f1, cm)
'Learning_curve:
Learning_Curve(model_grid)                       #tr_score : Decreasess ,increases / val_score : stagnate
#----------------------------------------------------
'Precision Recall Curve
from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_test, model_grid.decision_function(X_test)) #tabl np of each(precision, recall, threshold)
plt.plot(threshold, precision[:-1], label='precision')          #:-1 is to eliminat last col n because threshold[:n-1]
plt.plot(threshold, recall[:-1], label='recall')
legent()
def model_final(model, X_test, threshold=-1):
    return model.decision_function(X_test) > threshold           # modify the result y_pred according to threshold 
y_pred = model_final(model_grid, X_test, threshold=-1)
'Classification_report:
scoring(y_pred)                             #compare (accuracy acc, score f1, cm)
#-----------------------------------------------------
#Saving theta :
import sklearn.externals.joblib as jb
model_final.predict([[2,3,6,5,9]])
jb.dump(model_final , 'saved file.sav')
##############################################
savedmodel = jb.load('saved file.sav')
savedmodel.predict([[2,3,6,5,9]])
savedmodel.predict([[2,3,6,5,9]])











