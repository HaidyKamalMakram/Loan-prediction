### Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

### Data Collection & Preprocessing
loan_dataset = pd.read_csv('Train data.csv')
loan_dataset.head()

# Show NO of rows, col in loan_dataset
loan_dataset.shape

# Show statistical Measure
loan_dataset.describe()

# Show No of Missing Values in Each col
loan_dataset.isnull().sum()

# Dropping the missing values
loan_dataset = loan_dataset.dropna()

# No of Missing Values in Each col
loan_dataset.isnull().sum()

# label Encoding
loan_dataset.replace({"Loan_Status" :{'N': 0, 'Y': 1}}, inplace = True)

# the frist 5 rows of DataFrame
loan_dataset.head()

# Dependent col Values
loan_dataset['Dependents'].value_counts()

# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# Dependent col Values
loan_dataset['Dependents'].value_counts()

# Data Visualization

# Education & loan Status
sns.countplot(x = 'Education', hue = 'Loan_Status', data = loan_dataset)

# marital Status & loan Status
sns.countplot(x = 'Married', hue = 'Loan_Status', data = loan_dataset)

# convert categorical col to numerical Val
loan_dataset.replace({'Married' : {'No': 0, 'Yes': 1}, 'Gender' : {'Male': 1, 'Female': 0}, 'Self_Employed': {'No': 0, 'Yes': 1},
'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}, 'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace = True)

loan_dataset.head()

loan_dataset['LoanAmount'].plot(kind="hist", bins=10)

loan_dataset['Loan_Status'].value_counts().plot(kind='bar')

loan_dataset['Education'].plot(kind="hist", bins=10)

loan_dataset['Loan_Amount_Term'].plot(kind="hist", bins=10)

loan_dataset['Self_Employed'].plot(kind="hist", bins=10)

loan_dataset['Gender'].plot(kind="hist", bins=10)

loan_dataset['Married'].plot(kind="hist", bins=10)

loan_dataset['ApplicantIncome'].plot(kind="hist", bins=10)

loan_dataset['CoapplicantIncome'].plot(kind="hist", bins=10)

loan_dataset['Credit_History'].plot(kind="hist", bins=10)

loan_dataset['Property_Area'].plot(kind="hist", bins=10)

# Separating into Data, Label
X = loan_dataset.drop(columns= ['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

print(X)
print(Y)

# Split Data into Training, and Test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify= Y, random_state = 2)

print(X.shape, X_train.shape, X_test.shape)

# 1. Sepal scatter visualization
from termcolor import colored as cl # elegant printing of text
import seaborn as sb # visualizations
import matplotlib.pyplot as plt # editing visualizations
from matplotlib import style # setting styles for plots
from sklearn.preprocessing import StandardScaler # normalizing data
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.metrics import accuracy_score # algorithm accuracy
from sklearn.model_selection import train_test_split # splitting the data

sb.scatterplot('Loan_ID', 'Loan_Amount_Term', data = loan_dataset, hue = 'Loan_Status', palette = 'Set2', edgecolor = 'b', s = 150, alpha = 0.7)
plt.title('Loan_ID / Loan_Amount_Term')
plt.xlabel('Loan_ID')
plt.ylabel('Loan_Amount_Term')
plt.legend(loc = 'upper left', fontsize = 12)
plt.savefig('sepal.png')

# 2. Petal scatter visualization

sb.scatterplot('Loan_ID', 'Loan_Amount_Term', data = loan_dataset, hue = 'Loan_Status', palette = 'magma', edgecolor = 'b', s = 150,
               alpha = 0.7)
plt.title('Loan_ID / Loan_Amount_Term')
plt.xlabel('Loan_ID')
plt.ylabel('Loan_Amount_Term')
plt.legend(loc = 'upper left', fontsize = 12)
plt.savefig('petal.png')

# 3. Data Heatmap

df_corr = loan_dataset.corr()

sb.heatmap(df_corr, cmap = 'Blues', annot = True, xticklabels = df_corr.columns.values, yticklabels = df_corr.columns.values)
plt.title('Iris Data Heatmap', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.savefig('heatmap.png')

# 5. Distribution plot

plt.subplot(211)
sb.kdeplot(loan_dataset['Loan_Amount_Term'], color = 'b', shade = True, label = 'Sepal Width')

plt.subplot(212)
sb.kdeplot(loan_dataset['Loan_Status'], color = 'coral', shade = True, label = 'Petal Length')
sb.kdeplot(loan_dataset['LoanAmount'], color = 'green', shade = True, label = 'Petal Width')

plt.savefig('dist.png')

##########################################################################
# Training Model
# Vector Machine
# Import Libraries
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics, model_selection, svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

VM_classifier = svm.SVC(kernel='linear')

# training the Support Vector Machine Model
VM_classifier.fit(X_train, Y_train)

### Model Evaluation ###

# Predict the Train set results
VM_X_TrainPrediction = VM_classifier.predict(X_train)

# Calculate VM_Training Data Accuracy
VM_TrainingDataAccuracy = accuracy_score(VM_X_TrainPrediction, Y_train)
print('VM_Training Data Accuracy = ', VM_TrainingDataAccuracy)

# Predict the Test set results
VM_X_TestPrediction = VM_classifier.predict(X_test)

# Calculate VM_Testing Data Accuracy
VM_TestingDataAccuracy = accuracy_score(VM_X_TestPrediction, Y_test)
print('VM_Testing Data Accuracy = ', VM_TestingDataAccuracy)
print('\n')

### Visualization Model ###

# Visualize support vectors
metrics.plot_roc_curve(VM_classifier, X_test, Y_test)
plt.show()

VM_Training_CM = confusion_matrix(Y_train, VM_X_TrainPrediction)
print('\nMatrix Of Training Support Vector Machine = \n', VM_Training_CM)
VM_Testing_CM = confusion_matrix(Y_test, VM_X_TestPrediction)
print('\nMatrix Of Testing Support Vector Machine = \n', VM_Testing_CM)

sns.heatmap(VM_Testing_CM, annot=True)
plt.savefig('VM_Testing_CM.png')

# printing the report
print('\nclassification_report = \n ',classification_report(Y_test, VM_X_TestPrediction))

############################################
# Decision Tree Model

# Import Libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree # tree diagram
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt # visualization
from matplotlib import rcParams # figure size
from sklearn.metrics import confusion_matrix
from termcolor import colored as cl # text customization
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as dtc # tree algorithm

labelEncoder = LabelEncoder()
labelEncoder.fit(loan_dataset['Loan_Status'])

loan_dataset['Loan_Status'] = labelEncoder.transform(loan_dataset['Loan_Status'])
print(loan_dataset)

DecisionTree_train = loan_dataset.iloc[0: 384, :]
print(DecisionTree_train)

DecisionTree_test = loan_dataset.iloc[384: , :]
print(DecisionTree_test)

DecisionTree_classifier = DecisionTreeClassifier()
DecisionTree_classifier = DecisionTree_classifier.fit(DecisionTree_train.iloc[: ,1 : 12], DecisionTree_train['Loan_Status'])
print(plot_tree(DecisionTree_classifier))

### Model Evaluation ###

# Predict the Train set results
DT_X_TrainPrediction  = DecisionTree_classifier.predict(DecisionTree_train.iloc[:, 1:12])

# Calculate Decision Tree training Accuracy
DT_TrainingDataAccuracy  = accuracy_score(DecisionTree_train['Loan_Status'], DT_X_TrainPrediction)
print("\nDM_TrainingDataAccuracy = ", DT_TrainingDataAccuracy)

# Predict the Test set results
DT_X_TestPrediction = DecisionTree_classifier.predict(DecisionTree_test.iloc[:, 1:12])

# Calculate training_data_accuracy
DT_TestingDataAccuracy = accuracy_score(DecisionTree_test['Loan_Status'], DT_X_TestPrediction)
print("\nDM_TestingDataAccuracy = ", DT_TestingDataAccuracy)
print('\n')

### Visualization Model ###
tree.plot_tree(DecisionTree_classifier)
plt.savefig('DecisionTree_classifier1.png')

metrics.plot_roc_curve(DecisionTree_classifier, X_test, Y_test)
plt.show()
####################################
# Decision Tree Classifier with criterion entropy

# instantiate the DecisionTreeClassifier model with criterion entropy
DecisionTree_ClassifierEntropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

# fit the model
DecisionTree_ClassifierEntropy.fit(X_train, Y_train)

### Model Evaluation ###

# Predict the Train set results with criterion entropy
DT_Entropy_X_TrainPrediction = DecisionTree_ClassifierEntropy.predict(X_train)

# Calculate DM_Entropy_Training Data Accuracy
DT_Entropy_TrainingDataAccuracy = accuracy_score(DT_Entropy_X_TrainPrediction, Y_train)
print('DT_Entropy_Training Data Accuracy = ', DT_Entropy_TrainingDataAccuracy)

# Predict the Test set results with criterion entropy
DT_Entropy_X_TestPrediction  = DecisionTree_ClassifierEntropy.predict(X_test)

# Calculate VM_TestiDM_Entropy_Training  Data Accuracy
DT_Entropy_TestDataAccuracy = accuracy_score(DT_Entropy_X_TestPrediction, Y_test)
print('DT_Entropy_TestDataAccuracy = ', DT_Entropy_TestDataAccuracy)

#print('\nTraining set score: ',(DecisionTree_ClassifierEntropy.score(X_train, Y_train)))

#print('Test set score: ',(DecisionTree_ClassifierEntropy.score(X_test, Y_test)))

### Visualization Model ###

tree.plot_tree(DecisionTree_ClassifierEntropy)

print('\nMatrix Of Training Decision Tree = \n', confusion_matrix(Y_train, DT_Entropy_X_TrainPrediction))

print('\nMatrix Of Testing Decision Tree = \n', confusion_matrix(Y_test, DT_Entropy_X_TestPrediction))

print('\n')

tree.plot_tree(DecisionTree_ClassifierEntropy)
plt.savefig('DecisionTree_ClassifierEntropy.png')

metrics.plot_roc_curve(DecisionTree_ClassifierEntropy, X_test, Y_test)
plt.show()

DT_Entropy_Training_CM = confusion_matrix(Y_train, DT_Entropy_X_TrainPrediction)
print('\nMatrix Of Training Support Vector Machine = \n', DT_Entropy_Training_CM)
DT_Entropy_Testing_CM = confusion_matrix(Y_test, DT_Entropy_X_TestPrediction)
print('\nMatrix Of Testing Support Vector Machine = \n', DT_Entropy_Testing_CM)

sns.heatmap(DT_Entropy_Testing_CM, annot=True)
plt.savefig('DT_Entropy_Testing_CM.png')

# printing the report
print(classification_report(Y_test, DT_Entropy_X_TestPrediction))

##############################################################################
# Knn
# DONE
#import Libraries
from sklearn.neighbors import KNeighborsClassifier

data_train = loan_dataset.iloc[0:  384,:]
data_test = loan_dataset.iloc[384:  ,:]

KNeighbors_Classifier1 = KNeighborsClassifier(n_neighbors= 3 )

# fit the Model
KNeighbors_Classifier1.fit(data_train.iloc[:, 1:12], data_train['Loan_Status'])

### Model Evaluation ###

# Predict the Train set results
knn_X_TrainPredict = KNeighbors_Classifier1.predict(data_train.iloc[:, 1:12])

# Calculate knn_Train Data Accuracy
knn_TrainDataAccuracy = accuracy_score(data_train['Loan_Status'], knn_X_TrainPredict)
print('\nknn_TrainDataAccuracy = ', knn_TrainDataAccuracy)

# Predict the Train set results
knn_X_TestPredict = KNeighbors_Classifier1.predict(data_test.iloc[:, 1:12])

# Calculate DM_Entropy_Training Data Accuracy
knn_TestDataAccuracy = accuracy_score(data_test['Loan_Status'], knn_X_TestPredict)
print('\nknn_TestDataAccuracy = ', knn_TestDataAccuracy)

### Visualization Model ###
metrics.plot_roc_curve(KNeighbors_Classifier1, X_test, Y_test)
plt.show()

##########################
########## Another Knn Model (81.25%) ###########
# DONE
# Import Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Fit the model
KNeighbors_Classifier2 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

KNeighbors_Classifier2.fit(X_train_std, Y_train)

### Model Evaluation ###

# Predict the Train set results
y_predformodel = KNeighbors_Classifier2.predict(X_test_std)

# Calculate DM_Entropy_Training Data Accuracy
print("accuracy_score = ",accuracy_score(Y_test, y_predformodel))

# Evaluate the training and test score
print('\nTraining accuracy score: ', KNeighbors_Classifier2.score(X_train_std, Y_train))
print('\nTest accuracy score: ' , KNeighbors_Classifier2.score(X_test_std, Y_test))

### Visualization Model ###
metrics.plot_roc_curve(KNeighbors_Classifier2, X_test, Y_test)
plt.show()

KNeighbors2_Testing_CM = confusion_matrix(Y_test, y_predformodel)
print('\nMatrix Of Testing Support Vector Machine = \n', KNeighbors2_Testing_CM)

sns.heatmap(KNeighbors2_Testing_CM, annot=True)
plt.savefig('KNeighbors2_Testing_CM.png')

print('\nReport \n', classification_report(Y_test,y_predformodel))
##################################################

########## Another Knn Model ###########

# Import Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from matplotlib import cm
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred))

error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,Y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != Y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)

knntrain_results = knn.predict(X_train)

knntrain_accuracy = accuracy_score(Y_train, knntrain_results)
print('\ntrain_accuracy = ', knntrain_accuracy)

knntest_results = knn.predict(X_test)

knntest_accuracy = accuracy_score(Y_test, knntest_results)
print('\ntest_accuracy = ', knntest_accuracy)

### Visualization Model ###
metrics.plot_roc_curve(knn, X_test, Y_test)
plt.show()

KNeighbors2_Testing_CM = confusion_matrix(Y_test, y_predformodel)
print('\nMatrix Of Testing Support Vector Machine = \n', KNeighbors2_Testing_CM)

sns.heatmap(KNeighbors2_Testing_CM, annot=True)
plt.savefig('KNeighbors2_Testing_CM.png')

print('\nReport \n', classification_report(Y_test,y_predformodel))
###########################################
# Naive
# DONE
# Import Packages
from sklearn.naive_bayes import GaussianNB

Naive_X_train, Naive_X_test, Naive_Y_train, Naive_Y_test = train_test_split(loan_dataset.iloc[:, 1:12], loan_dataset['Loan_Status'], test_size=0.2, random_state=0)

GNB_classifier = GaussianNB()

### Model Evaluation ###

# Predict the Train set results
GNB_predictTrain = GNB_classifier.fit(Naive_X_train, Naive_Y_train).predict(Naive_X_train)

# Calculate Decision Tree training Accuracy
GNB_accuracyTrain = accuracy_score(Naive_Y_train, GNB_predictTrain)
print('\nGNB_accuracyTrain = ', GNB_accuracyTrain)

# Predict the Test set results
GNB_predictTest = GNB_classifier.fit(Naive_X_train, Naive_Y_train).predict(Naive_X_test)

# Calculate GNB training Accuracy
GNB_accuracyTest = accuracy_score(Naive_Y_test, GNB_predictTest)
print('\GNB_accuracy Test = ', GNB_accuracyTest)

# Visualize
metrics.plot_roc_curve(GNB_classifier, Naive_X_test, Naive_Y_test)
plt.show()

GNB_Training_CM = confusion_matrix(Naive_Y_train, GNB_predictTrain)
print('\nMatrix Of GNB_Training_CM = \n', GNB_Training_CM)
GNB_Testing_CM = confusion_matrix(Naive_Y_test, GNB_predictTest)
print('\nMatrix Of GNB_Testing_CM = \n', GNB_Testing_CM)

sns.heatmap(GNB_Testing_CM, annot=True)
plt.savefig('GNB_Testing_CM.png')

# printing the report
print(classification_report(Naive_Y_test, GNB_predictTest))
#################################################
# DONE
########## Another Naive Bayes Model ###########
GNB2_classifier = GaussianNB()

### Model Evaluation ###

# Predict the Train set results
GNB2_predictTrain = GNB2_classifier.fit(X_train, Y_train).predict(X_train)

# Calculate Decision Tree training Accuracy
GNB2_accuracyTrain = accuracy_score(Y_train, GNB2_predictTrain)
print('\nGNB2_accuracy Train = ', GNB2_accuracyTrain)

# Predict the Test set results
GNB2_predictTest = GNB2_classifier.fit(X_train, Y_train).predict(X_test)

# Calculate GNB training Accuracy
GNB2_accuracyTest = accuracy_score(Y_test, GNB2_predictTest)
print('\nGNB2_accuracy Test = ', GNB2_accuracyTest)

# Visualize
metrics.plot_roc_curve(GNB2_classifier, X_test, Y_test)
plt.show()

GNB2_Training_CM = confusion_matrix(Y_train, GNB2_predictTrain)
print('\nMatrix Of GNB2_Training_CM = \n', GNB2_Training_CM)
GNB2_Testing_CM = confusion_matrix(Y_test, GNB2_predictTest)
print('\nMatrix Of GNB_Testing_CM = \n', GNB2_Testing_CM)

sns.heatmap(GNB2_Testing_CM, annot=True)
plt.savefig('GNB2_Testing_CM.png')

# printing the report
print(classification_report(Y_test, GNB2_predictTest))
#########################################
########## Another Naive Bayes Model ###########
# DONE
# Import Packages
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import GaussianNB

sc = StandardScaler()
XXX_train = sc.fit_transform(X_train)
XXX_test = sc.transform(X_test)

#Training the Naive Bayes model on the training set
GNB3_Classifier = GaussianNB()
GNB3_Classifier.fit(XXX_train, Y_train)

### Model Evaluation ###

# Predict the Train set results
GNB3_predictTrain  =  GNB3_Classifier.predict(XXX_train)

# Calculate Decision Tree training Accuracy
GNB3_accuracyTrain = accuracy_score(Y_train, GNB3_predictTrain)
print('\nGNB_accuracy train = ', GNB3_accuracyTrain)

# Predict the Train set results
GNB3_predictTest  =  GNB3_Classifier.predict(XXX_test)

# Calculate Decision Tree training Accuracy
GNB3_accuracyTest = accuracy_score(Y_test, GNB3_predictTest)
print('\nGNB_accuracy = ', GNB3_accuracyTest)

# Visualize
metrics.plot_roc_curve(GNB3_Classifier, XXX_test, Y_test)
plt.show()

GNB3_Training_CM = confusion_matrix(Y_train, GNB3_predictTrain)
print('\nMatrix Of GNB3_Training_CM = \n', GNB3_Training_CM)
GNB3_Testing_CM = confusion_matrix(Y_test, GNB3_predictTest)
print('\nMatrix Of GNB3_Testing_CM = \n', GNB3_Testing_CM)

sns.heatmap(GNB3_Testing_CM, annot=True)
plt.savefig('GNB3_Testing_CM.png')

# printing the report
print(classification_report(Y_test, GNB3_predictTest))
#####################################
########## Another Naive Bayes Model ###########

# Import Packages
from sklearn.naive_bayes import GaussianNB
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score

GNB4_Classifier = GaussianNB()
GNB4_Classifier.fit(X_train, Y_train)

### Model Evaluation ###

# Predict the Train set results
GNB4_predictTrain  =  GNB4_Classifier.predict(X_train)

# Calculate Decision Tree training Accuracy
GNB4_accuracyTrain = accuracy_score(Y_train, GNB4_predictTrain)
print('\nGNB_accuracy train = ', GNB4_accuracyTrain)

# Predict the Train set results
GNB4_predictTest  =  GNB4_Classifier.predict(X_test)

# Calculate Decision Tree training Accuracy
GNB4_accuracyTest = accuracy_score(Y_test, GNB4_predictTest)
print('\nGNB_accuracy = ', GNB4_accuracyTest)

# Visualize
metrics.plot_roc_curve(GNB4_Classifier, X_test, Y_test)
plt.show()

GNB4_Training_CM = confusion_matrix(Y_train, GNB4_predictTrain)
print('\nMatrix Of GNB3_Training_CM = \n', GNB4_Training_CM)
GNB4_Testing_CM = confusion_matrix(Y_test, GNB4_predictTest)
print('\nMatrix Of GNB4_Testing_CM = \n', GNB4_Testing_CM)

sns.heatmap(GNB4_Testing_CM, annot=True)
plt.savefig('GNB4_Testing_CM.png')

# printing the report
print(classification_report(Y_test, GNB4_predictTest))
##################################################
########## Another Naive Bayes Model ###########

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import BernoulliNB

# initializaing the NB
NV_Classifier = BernoulliNB()

# training the model
NV_Classifier.fit(X_train, Y_train)

### Model Evaluation ###

# Predict the Train set results
NV_predictTrain  =  NV_Classifier.predict(X_train)

# Calculate Decision Tree training Accuracy
NV_accuracyTrain = accuracy_score(Y_train, NV_predictTrain)
print('\nGNB_accuracy train = ', NV_accuracyTrain)

# Predict the Train set results
NV_predictTest  =  NV_Classifier.predict(X_test)

# Calculate Decision Tree training Accuracy
NV_accuracyTest = accuracy_score(Y_test, NV_predictTest)
print('\nGNB_accuracy = ', NV_accuracyTest)

# Visualize
metrics.plot_roc_curve(NV_Classifier, X_test, Y_test)
plt.show()

NV_Training_CM = confusion_matrix(Y_train, NV_predictTrain)
print('\nMatrix Of NV_Training_CM = \n', NV_Training_CM)
NV_Testing_CM = confusion_matrix(Y_test, NV_predictTest)
print('\nMatrix Of NV_Testing_CM = \n', NV_Testing_CM)

sns.heatmap(NV_Testing_CM, annot=True)
plt.savefig('GNB4_Testing_CM.png')

# printing the report
print(classification_report(Y_test, NV_predictTest))
###############################################
# Random Forest
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm

# Building Random Forest model
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, Y_train)

# Evaluate Model

# Predict the Train set results with criterion entropy
RF_Entropy_X_TrainPrediction = rf.predict(X_train)

# Calculate DM_Entropy_Training Data Accuracy
RF_Entropy_TrainingDataAccuracy = accuracy_score(RF_Entropy_X_TrainPrediction, Y_train)
print('DT_Entropy_Training Data Accuracy = ', RF_Entropy_TrainingDataAccuracy)

# Predict the Test set results with criterion entropy
RF_Entropy_X_TestPrediction  = rf.predict(X_test)

# Calculate VM_TestiDM_Entropy_Training  Data Accuracy
RF_Entropy_TestDataAccuracy = accuracy_score(RF_Entropy_X_TestPrediction, Y_test)
print('DT_Entropy_TestDataAccuracy = ', RF_Entropy_TestDataAccuracy)

### Visualization Model ###

fn = str(loan_dataset['Loan_Amount_Term'])
cn = str(loan_dataset['Loan_Status'])
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=700)
tree.plot_tree(rf.estimators_[0],
               feature_names = fn,
               class_names=cn,
               filled = True);

fig.savefig('rf_individualtree.png')

print('\n')

metrics.plot_roc_curve(rf, X_test, Y_test)
plt.show()

RF_Training_CM = confusion_matrix(Y_train, RF_Entropy_X_TrainPrediction)
print('\nMatrix Of Training Support Vector Machine = \n', RF_Training_CM)
RF_Testing_CM = confusion_matrix(Y_test, RF_Entropy_X_TestPrediction)
print('\nMatrix Of Testing Support Vector Machine = \n', RF_Testing_CM)

sns.heatmap(RF_Testing_CM, annot=True)
plt.savefig('RF_Testing_CM.png')

# printing the report
print(classification_report(Y_test, RF_Entropy_X_TestPrediction))

##################################################
# Testing Data

### Import Packages
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import accuracy_score

### Data Collection & Preprocessing
NewCustomerDataset = pd.read_csv('New Customer.csv')
NewCustomerDataset.head()

# Show NO of rows, col in New Customer Dataset
NewCustomerDataset.shape

# Show statistical Measure
NewCustomerDataset.describe()

# Show No of Missing Values in Each col
NewCustomerDataset.isnull().sum()

# Dropping the missing values
NewCustomerDataset = NewCustomerDataset.dropna()

# No of Missing Values in Each col
NewCustomerDataset.isnull().sum()

# the frist 5 rows of DataFrame
NewCustomerDataset.head()

# Dependent col Values
NewCustomerDataset['Dependents'].value_counts()

# replacing the value of 3+ to 4
NewCustomerDataset = NewCustomerDataset.replace(to_replace='3+', value=4)


# convert categorical col to numerical Val
NewCustomerDataset.replace({'Married' : {'No': 0, 'Yes': 1}, 'Gender' : {'Male': 1, 'Female': 0}, 'Self_Employed': {'No': 0, 'Yes': 1},
'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}, 'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace = True)


NewCustomerDataset = NewCustomerDataset.drop(columns= ['Loan_ID'], axis=1)
print(NewCustomerDataset)


#Predict values using test data (Naive Bayes)
pred_test = GNB_classifier.predict(NewCustomerDataset)

#Write test results in csv file
predictions = pd.DataFrame(pred_test, columns=['predictions']).to_csv('m3.csv')
mm = pd.read_csv('m3.csv')

mm.replace({"predictions" :{0: 'N', 1:'Y'}}, inplace = True)
m = mm.copy().to_csv('m.csv')
mm['predictions'].value_counts()
