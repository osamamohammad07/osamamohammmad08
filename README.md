# osamamohammmad08
Data Modelling
• Built a classification model to classify whether the income exceeds $50K/yr
• Performed data cleaning to handle missing values, and ensure datatype consistency
• Conducted exploratory data analysis
• Built a data pipeline with the following components
• Used scalers (min max, standardize, normalize) for numerical variables
• Reduced the number of factors for categorical predictors
• Models used (logistic, naive bayes, decision trees, kNN, and neural network)
• Performed hyperparameter tuning and reported all the different model combinations
• Compared the performance of different models and selected the best model along with the best model parameters
• Tried hold out method and cross validation method for training
• Perfomed variable selection procedure (forward, backward) to select the best predictors 
#Import the dataset
import pandas as pd
from google.colab import files
file = files.upload()
train_df = pd.read_csv("Train_data.csv")
test_df = pd.read_csv("Test_data.csv")
from sklearn import neighbors
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from tabulate import tabulate
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, plot_roc_curve
from sklearn.naive_bayes import MultinomialNB
train_df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
test_df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
train_df_us = train_df
test_df_us = test_df
train_df
train_df_us.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
test_df_us.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
import numpy as np
from sklearn.preprocessing import StandardScaler
from numpy import nan
train_df = train_df[(train_df != ' ?').all(axis=1)]
train_df
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
Ord = OrdinalEncoder()
lab = LabelEncoder()
scale = StandardScaler()


categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
target = ['income']
scaler = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

train_df[categorical] = Ord.fit_transform(train_df[categorical])
train_df[target] = lab.fit_transform(train_df[target])
train_df_us[categorical] = Ord.fit_transform(train_df_us[categorical])
train_df_us[target] = lab.fit_transform(train_df_us[target])
train_df[scaler] = scale.fit_transform(train_df[scaler])
test_df[categorical] = Ord.fit_transform(test_df[categorical])
test_df[target] = lab.fit_transform(test_df[target])
test_df[scaler] = scale.fit_transform(test_df[scaler])
test_df_us[categorical] = Ord.fit_transform(test_df_us[categorical])
test_df_us[target] = lab.fit_transform(test_df_us[target])
tdfn = train_df
tedfn = test_df
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.subplots(figsize=(20,10))
sns.heatmap(tdfn.corr(), annot=True, cmap="Blues")
plt.subplots(figsize=(20,10))
sns.heatmap(tedfn.corr(), annot=True, cmap="Reds")
plt.subplots(figsize=(20,10))
sns.boxplot(x = 'income', y='age', data=tdfn , width = 0.25, linewidth = 1)
plt.subplots(figsize=(20,10))
sns.scatterplot(data = tdfn , x="age", y="fnlwgt", hue = 'income')
X_train = tdfn.iloc[:,0:14]
y_train = tdfn.iloc[:,14]
X_test = tedfn.iloc[:,0:14]
y_test = tedfn.iloc[:,14]
X_us_train = train_df_us.iloc[:,0:14]
y_us_train = train_df_us.iloc[:,14]
X_us_test = test_df_us.iloc[:,0:14]
y_us_test = test_df_us.iloc[:,14]
#Logistic Regression Backward predictor
logreg = LogisticRegression(max_iter= 200)
for i in range(14, 0, -1):
  X_train = tdfn.iloc[:,0:i]
  y_train = tdfn.iloc[:,14]
  X_test = tedfn.iloc[:,0:i]
  y_test = tedfn.iloc[:,14]
  logreg.fit(X_train,y_train)
  y_pred=logreg.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(accuracy)
#Logistic Regression Forward predictor
logreg = LogisticRegression(max_iter= 200)
for i in range(1, 14):
  X_train = tdfn.iloc[:,0:i]
  y_train = tdfn.iloc[:,14]
  X_test = tedfn.iloc[:,0:i]
  y_test = tedfn.iloc[:,14]
  logreg.fit(X_train,y_train)
  y_pred=logreg.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(accuracy)
#KNN Classifier Backward predictor
Accuracy_score = 0
logreg = LogisticRegression()
for i in range(14, 0, -1):
  X_train = tdfn.iloc[:,0:i]
  y_train = tdfn.iloc[:,14]
  X_test = tedfn.iloc[:,0:i]
  y_test = tedfn.iloc[:,14]
  kNN_classifier = neighbors.KNeighborsClassifier(n_neighbors = 16, metric = 'euclidean')
  kNN_classifier.fit(X_train, y_train)
  pred = kNN_classifier.predict(X_test)
  accuracy = accuracy_score(y_test, pred)
  print(accuracy)
  
#KNN Classifier Forward predictor
Accuracy_score = 0
logreg = LogisticRegression()
for i in range(1,14):
  X_train = tdfn.iloc[:,0:i]
  y_train = tdfn.iloc[:,14]
  X_test = tedfn.iloc[:,0:i]
  y_test = tedfn.iloc[:,14]
  kNN_classifier = neighbors.KNeighborsClassifier(n_neighbors = 16, metric = 'euclidean')
  kNN_classifier.fit(X_train, y_train)
  pred = kNN_classifier.predict(X_test)
  accuracy = accuracy_score(y_test, pred)
  print(accuracy)
  
#Classification Metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('True Positive:',tp)
print('True Negative:',tn)
print('False Positive:',fp)
print('False Negative:',fn)

# Classification accuracy
print('Accuracy:',(tp+tn)/(tn+fp+fn+tp))

# Overall classification error
print('Error:',(fp+fn)/(tn+fp+fn+tp))

# Sensitivity or recall
print('Sensitivity:', (tp)/(fn+tp))

# Specificity
print('Specificty:', (tn)/(fp+tn))

# Positive prediction value or precision
print('Positive Prediction value:',(tp)/(fp+tp))

# Negative prediction value 
print('Negative Prediction value:',(tn)/(fn+tn))

# False positive rate
print('False Positive Rate:',(fp)/(fp+tp))

# False negative rate
print('False Negative Rate:',(fn)/(fn+tn))

# F1 Score
print('F1 Score:', f1_score(y_test, y_pred))

#ROC and AUC
y_p_roc = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_p_roc)
auc = metrics.roc_auc_score(y_test, y_p_roc)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

validation_error = np.empty(20)
auc = np.empty(20)
k = 0
df_results = pd.DataFrame(columns = ['k', 'Validation Error'])
for i in range(1, 21):
  kNN_classifier = neighbors.KNeighborsClassifier(n_neighbors = i, metric = 'euclidean')
  kNN_classifier.fit(X_train, y_train)
  pred = kNN_classifier.predict(X_test)
  k = i
  validation_error = 1-accuracy_score(y_test, pred)
  df_results.loc[i] = [k, validation_error]

print(df_results)
sns.lineplot(x = 'k', y = 'Validation Error', data = df_results )
#Classification Metrics
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print('True Positive:',tp)
print('True Negative:',tn)
print('False Positive:',fp)
print('False Negative:',fn)
# Classification accuracy
print('Accuracy:',(tp+tn)/(tn+fp+fn+tp))

result = pd.DataFrame(columns=['Learning Rate', 'Transfer Function',  'Accuracy'])
lr = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 1, 3, 10]
transfer_function = ['identity', 'logistic', 'tanh', 'relu']
k=0
for i in lr:
  for j in transfer_function:
    model = MLPClassifier(learning_rate_init= i, activation=j, max_iter = 300)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    result.loc[k] = [i, j,  accuracy_score(y_test, pred)]
    k+=1
print(tabulate(result, headers=result.columns, tablefmt='grid'))

sns.barplot(x='Learning Rate', y='Accuracy', hue='Transfer Function', data=result)

#Hold Out Method
pipe_kNN = Pipeline (steps = [ ('model',neighbors.KNeighborsClassifier(n_neighbors=16))])
pipe_kNN_scaled = Pipeline (steps = [('Scaler', StandardScaler()), ('model',neighbors.KNeighborsClassifier(n_neighbors=16))])
pipe_lr = Pipeline (steps = [ ('model',linear_model.LogisticRegression())])
pipe_NN = Pipeline (steps = [ ('model',MLPClassifier(learning_rate_init= 0.001, activation= 'relu'))])
pipe_DT = Pipeline (steps = [ ('model',DecisionTreeClassifier())])

rmse = 0
mae = 0
Accuracy_score = 0
df_read = pd.DataFrame(columns = ['rmse','mae'])
df_read_as = pd.DataFrame(columns = ['Accuracy Score'])
pipe_list = [pipe_kNN,pipe_kNN_scaled,pipe_lr,pipe_NN,pipe_DT]
for pipe in pipe_list:
  model = pipe.fit(X_train, y_train)
  pred = model.predict(X_test)
  rmse = mean_squared_error(y_test, pred, squared=False)
  mae = mean_absolute_error(y_test, pred)
  Accuracy_score= accuracy_score(y_test, pred)
  df_read.loc[pipe] = [rmse, mae]
  df_read_as.loc[pipe] = [Accuracy_score]
print(df_read)
print(" ")
print(df_read_as)

from sklearn.naive_bayes import MultinomialNB
NB_model = MultinomialNB()
NB_model.fit(X_us_train, y_us_train)
pred_us= NB_model.predict(X_us_test)
rmse = mean_squared_error(y_us_test, pred_us, squared=False)
mae = mean_absolute_error(y_us_test, pred_us)
Accuracy_score= accuracy_score(y_us_test, pred_us)
print('RMSE=', rmse ,'MAE=', mae, 'Accuracy Scores=', Accuracy_score)

from sklearn.model_selection import GridSearchCV
model = MLPClassifier()
parameters = {'activation':('identity', 'logistic', 'tanh', 'relu'), 'learning_rate_init':[0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 1, 3, 10]}
optimized_model = GridSearchCV(model, parameters, cv=5)
optimized_model.fit(X_train, y_train)
print(tabulate(optimized_model.cv_results_, headers=optimized_model.cv_results_.keys(), tablefmt='grid'))

pipe_kNN = Pipeline (steps = [ ('model',neighbors.KNeighborsClassifier())])
pipe_kNN_scaled = Pipeline (steps = [('Scaler', StandardScaler()), ('model',neighbors.KNeighborsClassifier())])
pipe_kNN_scaled_pca = Pipeline (steps = [('Scaler', StandardScaler()), ('pca',PCA(n_components=2)), ('model',neighbors.KNeighborsClassifier())])


pipe_list_1 = [pipe_kNN, pipe_kNN_scaled, pipe_kNN_scaled_pca]

for pipe in pipe_list_1:
  parameters = {'model__n_neighbors':np.arange(1,21), 'model__metric': ['euclidean', 'manhattan'], 'model__weights': ['uniform', 'distance']}
  optimized_model = GridSearchCV(pipe, parameters, cv=5 )
  optimized_model.fit(X_train, y_train)
  print('Best score: ',optimized_model.best_score_)
  print('Best parameters: ',optimized_model.best_params_)
  
  
