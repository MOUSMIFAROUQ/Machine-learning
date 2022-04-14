iris = load_iris()
data = pd.DataFrame({'sepal length (cm)': iris.data[:,0],
        'sepal width (cm)': iris.data[:,1],
        'petal length (cm)': iris.data[:,2],
        'petal width (cm)': iris.data[:,3],
        'target': iris.target
        })
Data = data.copy()
#========================= Call module ==============================================================================================<
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#=============== we are doing in this kernel something like:EDA (Exploratory Data Analysis)==========================================<
#Analysis the shape : We'll just focus on the form of data.
'Ndr of rows & columns : (150, 5)
print(Data.head())
print(Data.shape)                                                               
'Analyse target : (reg or classif) / (variables type:'numerical' or 'categorical') / (equilibre or not) classif non equilibre 10% of var1
Data['column_y'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')
'% of each types of values data : 
print(Data.dtypes)                                                                         #display all features with dtype
plt.figure()
Data.dtypes.value_counts(normalize=True).plot.pie(autopct='%1.1f%%')                       #column1, column2 , .. = object , float, int,..
#=============== Missing data ==========================================================================================================<
#Analysis missing data : We'll see the percentage of values missing data and try to choice of the right features (feature selection) and not complex relationships (feature engineering)
'visualisat val missing : (visualisat of array[True, False] in order to detect nan(% of emptying))
sns.heatmap(Data.isna()) 
sns.heatmap(Data.isnull())
'Analyse val nan : ( % of nan(emptying) for each column with sum in axis 0 )                                                 
total = Data.isna().sum().sort_values()
percent = (Data.isna().sum()/Data.isna().count()).sort_values()                 
Data_nan = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 
Data_nan.head(50)
'Analyse val 0 : ( % of null(emptying) for each column with sum in axis 0 )
total = Data.isnull().sum().sort_values()
percent = (Data.isnull().sum()/Data.isnull().count()).sort_values()              
Data_null = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
Data_null.head(50)
# If 15% of the data is missing, we should delete the corresponding variable. According to this, we should delete...
# If we have one missing observation in variable. we'll delete this observation and keep the variable.
'Filtration data :
Data.drop(['column_ID','column_NAME', 'column_REF', ...], axis=1, inplace=True)           #axis=1 for column / inplace=modifiat in data
# We can see that these variables have the same number of missing data, and considering that we are just talking about 5% of missing data. According to this, we should delete...
'Save column of val_nan < 0.9 : 
index_na=(Data.isna().sum()/Data.shape[0]) > 0.9                                          #get array[True, False]<0.9 in order to detect nan
group_columns = Data.columns[index_na]                                                    #list of column that 90% is empeting 
Data.drop([group_columns], axis=1, inplace=True)
'Impacte of drop nan in rows: (calculate nbr of rows remaining after drop) / (classif non equilibre or not)
Data.dropna(axis = 0).count()                                     
Data.dropna()['column_y'].value_counts(normalize=True).plot.pie(autopct='%1.1f%%')
#============= Grouped data ===========================================================================================================<
#To fill Segment & Expectation column, we should ask ourselves:
'Variables columns : (Display Variable type & name)
for col in data.columns:
    print(f'{col :-<50} {data[col].unique()}') 
#Do we think about this variable when we talk about target? Identification of the variables segment: building, space or location.
'grouped by segment in new variabale : (put new column'satut' that have array[fun])
def fun(data): 
    if data['column_A']==1:     return 'ok'
    elif data['column_B']==1:     return 'Nok'
    else:   return 'inconnu'
Data['statut'] = Data.apply(fun,axis=1)
#How important would this variable be? the variable influence in target :'High','Medium' and 'Low'
'grouped by number of features correlation
corr_frame = Data.corr().nlargest(20, 'score4')                                           #sorted data corelation features according to 'score4' with only 20 features correlation
group_column = corr_frame.index                                                           #list of column
Data_corr = Data[group_columns]
#Is this information already described in any other variable?
'grouped by sum in new variabale :
index_sum = Data[Data.columns[1:3]]=='object'                                             #get data array[True, False]=='object'
Data['sum'] = index_sum.sum(axis=1)                                                       #put new column'sum' that have array[sum index]
Data['sum'] = index_sum.sum(axis=1)>=1                                                    #put new column'sum' that have array[True, False]>=1
#I ended up with'building' variables ('OverallQual' and 'YearBuilt') and 'space' variables ('TotalBsmtSF' and 'GrLivArea').
#============= Analysis of Target =====================================================================================================<
#Univariable target Analyse : We ll analysis the target variable and looking in its characteristics
'descriptive the target variable : (min & max value is larger than zero or not)
Data['column_y'].describe()                            
'describe skewness and kurtosis of target : (Mode=max freq/Moyenne/Mediane)/(degree of symmetry(desymetriq a gauche ou a droit) & flatness)
sns.distplot(Data['column_y'])
print("Skewness: %f" % Data['column_y'].skew())                                           #degree of symmetry (-sk_left<Normalsk_sym<+sk_right)
print("Kurtosis: %f" % Data['column_y'].kurt())                                           #degree of flatness (-ku_platy<Normalku_meso<+ku_Lepto)
#============= Analysis of variables =====================================================================================================<
#Univariable variable Analyse : We ll analysis each variable dependently and looking in their meaning and importance for this problem
'Histo des variables columns number : (distibution normale ou gausien symetrique or not) / uniforme / multi modale
for col in Data.select_dtypes(include=['int','float']):
    sns.displot(Data, x=col,kde=True)                                                     #hist visualisation figure for each numerical col
'Histo des variables columns object : (binaire(0,1) or not) / (equilibre or not)
for col in Data.select_dtypes('object'):
    print(f'{col :-<50} {Data[col].unique()}')                                            #get list[object] of each col
    Data[col].value_counts(normalize=True).plot.pie()
#============= Analysis of relations Traget =====================================================================================================<
#Multivariate target study : We ll try to understand how the target and other types variables relate.
'Relation Target/Variables : (Taux variables/target >0.9)
for data/data :
sns.heatmap(Data.corr())                                                                     #get correlat of all_columns/all_columns == print(Data.corr())
#'column_A', 'column_F' and 'column_S' are strongly correlated with 'Target'.
#we can see a significant correlation between 'column_A' and 'column_F'.
#'column_A' is a consequence of 'column_F'. we can only keep 'column_A' since its correlation with 'Target' is higher.
for col in Data.select_dtypes(exclude=['object','int']):                                     #Data_A=Data[Data['column_y']==A] 
    plt.figure()                                 
    for cat in Data['column_y'].unique():
        sns.distplot(Data[Data['column_y']==cat][col],kde=True,label=cat)                    #display histo target/col_'float64'
    plt.legend()
#The variables 'column_A', 'column_F' and 'column_S', can play an important role in this problem According to the histo.
for col in Data.select_dtypes(include=['float64']):  
    sns.scatter(x=col, y='column_Y', data=Data)                                              #display target/col scatter data 
#'column_A' is linearly related with 'Target'. the relationships is positive & the slope of the linear relationship is particularly high.
for col in Data.select_dtypes(include=['int']):                                           
    plt.figure()
    sns.countplot(data=Data, x=col,hue=Data['column_y'])                                     #display counthisto target/col_'int'
for col in Data.select_dtypes(include=['object']):                                        
    plt.figure(figsize=(15,5))
    sns.heatmap(pd.crosstab(Data['column_y'],Data[col]), annot=True, fmt='d')               #get matrice of nbr apparition and not correlation matrix
#These variables have not impact in target. Furthermore, they are already considered by correlation with 'OverallQual'.According to this, we should delete...
#============= Analysis of relations Variables =====================================================================================================<
#Multivariate target study : We ll try to understand how the dependent and independent variables relate.
'Relation variables/variables : (Taux de corre variables/variables>0.9)
#Scatter plots between the most correlated variables
sns.pairplot(Data)                                                                           #get scatter_correlat for Regression
sns.pairplot(Data, hue='column_y')                                                           #display all scatter data GROUP BY color==Y=='column_y' for classification
#If the dots drawing are linear line / the upper limit is a linear function / the bottom limit is an exponential function)
sns.clustermap(Data.corr())                                                                  #get heatmap_correlat with cluster useful for modelisation

    - for group_columns/group_columns :
sns.heatmap(Data[group_columns].corr())                                                      #get correlat of group_columns/group_columns
sns.heatmap(corr_frame[group_column], annot=True, fmt='.2f', annot_kws={'size': 10})         #get matrice of correlation

    - for column/group_columns :
for col in group_columns: '(detecte outliers)/(if regression lines are super) / (if relation is lineare)
    plt.figure()    
    for cat in Data['column'].unique():                                                      #unique()==get list[object] of 'column'
        sns.distplot(Data[Data['column']==cat][col],kde=True,label=cat)                      #Data_A=Data[Data['column']==A]
    plt.legend() 
    sns.lmplot(Data, x='column', y=col,hue='column_y')                                       #get scatter_correlat with 'regression line' of 'column_y'
print(Data.corr()['column'].sort_values())                                                   #get correlat of 'columns_int'/all variables

    - for column/column :
sns.heatmap(pd.crosstab(Data['column_A'],Data['column_B']), annot=True, annot_kws={'size': 5}, fmt="d")   #get matrice of nbr apparition
sns.jointplot('column_A','column_B',data=Data)                                               #display distibution of column_A & column_B(kind='kde', size='')
sns.scatter(x='column_A', y='column_B', data=Data, hue='column_y')                           #display col1/col2 scatter data GROUP BY 'column_y' to regression
sns.catplot(x='column_A', y='column_B', data=Data, hue='column_y')                           #display col1/col2 box data GROUP BY 'column_y'  to detect outliers
#scatter : values that seem strange and they are not following the crowd. these two points are not representative of the typical case. We will define them as outliers and delete them.
    data[(data['column1'] > 18])&(data['column1'] < 20])]['column2']                         #SELECT 'column2' FROME data WHERE 18 < column1 < 20
    data.groupby(['column1','column2']).mean()                                               #get (mean en %) GROUP BY 'column1' and 'column2'

#----------------------------------------------------
#Analyse Hypothese:
    - rate of 'columns_y' depending to each col => prove if the rate of 'column_y' are not equal
from scipy.stats import ttest_ind                                                            #calculate p for each column of test => compare p with alpha
def t_test(col):
    alpha=0.02
    #=>array[True, False]           #=>Data_A=Data[rows['column_y']==A]
    index_A=Data['column_y']==A     Data_A = Data[index_A]
    index_B=Data['column_y']==B     Data_B = Data[index_B]
    stat, p = ttest_ind(Data_A[col].dropna(),Data_B[col].dropna())                           #dropna for eliminat nan values for test
    if p < alpha:
        return 'H0 Rejetée'                                                                  #col have rate not equal for 'column_y'
    else:
        return 0                                                                             #col have not impact to 'column_y'
for col in Data.columns:
    print(f'{col :-<50} {t_test(col)}')


