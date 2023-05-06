import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder

def showGraphics(stu):
    # visualizing the grades
    plt.figure(figsize=(8, 6))
    sns.countplot(stu["grades"], order=["low", "average", "high"], palette='Set1')
    plt.title('Final Grade - Number of Students', fontsize=20)
    plt.xlabel('Final Grade', fontsize=16)
    plt.ylabel('Number of Student', fontsize=16)
    plt.show()

    # describing correlation
    corr = stu.corr()

    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, cmap="Reds")
    plt.title('Correlation Heatmap', fontsize=20)
    plt.show()

    # comparing school with grades
    sns.boxplot(x="school", y="total_grades", data=stu)

    school_counts = stu["school"].value_counts().to_frame()
    school_counts.rename(columns={"school": "school_counts"}, inplace=True)
    school_counts.index.name = 'school'

    school_sns = sns.countplot(hue=stu["school"], x=stu["grades"], data=stu)

    # crosstab is expanded form of value counts the the factors inside any variables
    perc = (lambda col: col / col.sum())
    index = ["average", "high", "low"]
    schooltab1 = pd.crosstab(columns=stu.school, index=stu.grades)

    school_perc = schooltab1.apply(perc).reindex(index)

    school_perc.plot.bar(colormap="PiYG_r", fontsize=15, figsize=(7, 7))
    plt.title('Final Grade By school', fontsize=20)
    plt.ylabel('Percentage of Student Counts ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.show()

    # comparing address with grades
    sns.boxplot(x="address", y="total_grades", data=stu)
    index = ["average", "high", "low"]
    addresstab1 = pd.crosstab(columns=stu.address, index=stu.grades)

    address_perc = addresstab1.apply(perc).reindex(index)

    address_perc.plot.bar(colormap="PiYG_r", fontsize=15, figsize=(7, 7))
    plt.title('Final Grade By address', fontsize=20)
    plt.ylabel('Percentage of Student Counts ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.show()
    # address is factor for the grades

    # comparing famsize with grades
    sns.boxplot(x="famsize", y="total_grades", data=stu)
    famsizetab1 = pd.crosstab(columns=stu.famsize, index=stu.grades)

    famsize_perc = famsizetab1.apply(perc).reindex(index)

    famsize_perc.plot.bar(colormap="PiYG_r", fontsize=15, figsize=(7, 7))
    plt.title('Final Grade By famsize', fontsize=20)
    plt.ylabel('Percentage of Student Counts ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.show()
    # famsize has great impact on grades

    # comparing sex with grades
    sns.boxplot(x="sex", y="total_grades", data=stu)
    school_counts = stu["sex"].value_counts()
    # as the graph of sex nearly overlaps so it will not have impact on grades

    # comparing pstatus with grades
    sns.boxplot(x="Pstatus", y="total_grades", data=stu)
    Pstatustab1 = pd.crosstab(columns=stu.Pstatus, index=stu.grades)

    Pstatus_perc = Pstatustab1.apply(perc).reindex(index)

    Pstatus_perc.plot.bar(colormap="PiYG_r", fontsize=15, figsize=(7, 7))
    plt.title('Final Grade By Pstatus', fontsize=20)
    plt.ylabel('Percentage of Student Counts ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.show()
    # it is not a good factor

    # comparing jobs
    sns.boxplot(x="Mjob", y="total_grades", data=stu)
    sns.boxplot(x="Fjob", y="total_grades", data=stu)
    stu1 = stu[["Fjob", "Mjob", "total_grades"]]
    job_grp = stu1.groupby(['Mjob', 'Fjob'], as_index=False).mean()
    job_pivot = job_grp.pivot(index='Mjob', columns='Fjob', values='total_grades')
    plt.show()

    # so father and mother jobs has great impact on grades

    # comparing reasons
    sns.boxplot(x="reason", y="total_grades", data=stu)
    plt.show()
    # it has impact on the grades

    # comparing guardians
    sns.boxplot(x="guardian", y="total_grades", data=stu)

    guardiantab1 = pd.crosstab(columns=stu.guardian, index=stu.grades)
    guardian_perc = guardiantab1.apply(perc).reindex(index)
    guardian_perc.plot.bar(colormap="BrBG", fontsize=15, figsize=(7, 7))
    plt.title('Final Grade By guardian', fontsize=20)
    plt.ylabel('Percentage of Student Counts ', fontsize=16)
    plt.xlabel('Final Grade', fontsize=16)
    plt.show()
    # so guardian has grat impact on grades

    sns.boxplot(x="internet", y="total_grades", data=stu)
    plt.show()
    # internet also have great impact on performance of individual
def printInfos(stu):
    print("\nDATASET INFORMATIONS TYPES")
    print(stu.dtypes)

    # print(stu.describe().to_string())

    # describing categorical data
    print("\nDATASET DESCRIPTION")
    print(stu.describe(include="all").to_string())

    stu.info()

    # checking for null values
    print("\nCHECKING NULL VALUE")
    print(stu.isnull().any())

pd.set_option('display.expand_frame_repr', False)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#import data
stu_por = pd.read_csv("input/student-por.csv", sep=";")
stu_mat = pd.read_csv("input/student-mat.csv", sep=";")

#Merging two datasets
stu = pd.concat([stu_por, stu_mat])

#Calculating Total Grade
stu["total_grades"] = (stu["G1"]+stu["G2"]+stu["G3"]) / 3

#print("\nDATASET MERGED")
#print(stu.to_string())

#dropping columns
stu=stu.drop(["G1","G2","G3"],axis=1)
#max=stu["total_grades"].max()
#min=stu["total_grades"].min()

#ranging the grade in three parts
def marks(total_grades):
    if(total_grades<7):
        return("low")
    elif(total_grades>=7 and total_grades<14):
        return("average")
    elif(total_grades>=14):
        return("high")

stu["grades"]=stu["total_grades"].apply(marks)

print("\nDATASET  WITH TOTAL GRADE,RANK OF GRADE AND NO G1,G2,G3")
print(stu.to_string())

printInfos(stu)
showGraphics(stu)

############################# Features Selection #######################################################################

stu = stu.drop(["sex"], axis=1)
stu = stu.drop(['Pstatus'], axis=1)
stu = stu.drop(["Medu"], axis=1)
stu = stu.drop(['Walc'], axis=1)

############################# Transform Categorical Features ###########################################################
'''
transformers=[
    ['category vectorizer', OneHotEncoder(), [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]]
]
ct=ColumnTransformer(transformers, remainder='passthrough')
ct.fit(stu)
stu=ct.transform(stu)
stu=pd.DataFrame(stu)
print("\nDATASET TRANSFORMED")
print(stu.to_string())
'''

stu['school'], _ = pd.factorize(stu['school'], sort=True)
stu['address'], _ = pd.factorize(stu['address'], sort=True)
stu['famsize'], _ = pd.factorize(stu['famsize'], sort=True)
stu['Fedu'], _ = pd.factorize(stu['Fedu'], sort=True)
stu['Mjob'], _ = pd.factorize(stu['Mjob'], sort=True)
stu['Fjob'], _ = pd.factorize(stu['Fjob'], sort=True)
stu['reason'], _ = pd.factorize(stu['reason'], sort=True)
stu['guardian'], _ = pd.factorize(stu['guardian'], sort=True)
stu['schoolsup'], _ = pd.factorize(stu['schoolsup'], sort=True)
stu['famsup'], _ = pd.factorize(stu['famsup'], sort=True)
stu['paid'], _ = pd.factorize(stu['paid'], sort=True)
stu['activities'], _ = pd.factorize(stu['activities'], sort=True)
stu['nursery'], _ = pd.factorize(stu['nursery'], sort=True)
stu['higher'], _ = pd.factorize(stu['higher'], sort=True)
stu['internet'], _ = pd.factorize(stu['internet'], sort=True)
stu['romantic'], _ = pd.factorize(stu['romantic'], sort=True)

#stu['grades'], _ = pd.factorize(stu['grades'], sort=True)
#stu['sex'], _ = pd.factorize(stu['sex'], sort=True)
#stu['Pstatus'], _ = pd.factorize(stu['Pstatus'], sort=True)
#stu['Medu'], _ = pd.factorize(stu['Medu'], sort=True)

#print("DATASET NO CATEGORIZED")
#print(stu.to_string())


############################# Splitting ################################################################################
X=stu.drop(["total_grades","grades"],axis=1).values
y = stu['grades'].values
y, _ = pd.factorize(y, sort=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=5)

############################# Scalamento of data #######################################################################

scaler = StandardScaler().fit(X)
X_train=  scaler.transform(X)
scaler = StandardScaler().fit(X_test)
X_test=  scaler.transform(X_test)

############################# Models Definition ########################################################################

models = [KNeighborsClassifier(weights='distance'),
          LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced'),
          SVC(class_weight='balanced'),
          DecisionTreeClassifier(class_weight='balanced')]

models_names = ['K-NN',
                'Softmax Reg.',
                'SVM',
                'DT']

models_hparametes = [{'n_neighbors': list(range(1, 10, 2))},  # KNN
                     {'penalty': ['l1', 'l2'], 'C': [1e-5, 5e-5, 1e-4, 5e-4, 1]},
                     {'C': [1e-4, 1e-2, 1, 1e1, 1e2], 'gamma': [0.001, 0.0001], 'kernel': ['linear', 'rbf']},  # SMV
                     {'criterion': ['gini', 'entropy']},  # DT
                     ]

chosen_hparameters = []
estimators = []

for model, model_name, hparameters in zip(models, models_names, models_hparametes):
    print('\n ', model_name)
    clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy', cv=5)
    clf.fit(X_train, y)
    chosen_hparameters.append(clf.best_params_)
    estimators.append((model_name, clf))
    print('Accuracy:  ', clf.best_score_)

print('############ Ensemble  ############ \n')

estimators.pop(1)

clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

scores = cross_validate(clf_stack, X_train, y, cv=5, scoring=('f1_weighted', 'accuracy'))
print('The cross-validated weighted F1-score of the Stacking Ensemble is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Accuracy of the Stacking Ensemble is ', np.mean(scores['test_accuracy']))

############################# Final Choice of Model ####################################################################

final_model= clf_stack

############################# Final Training  ##########################################################################
final_model.fit(X_train, y)

############################# Prediction and Valutation ################################################################

y_pred = final_model.predict(X_test)

########################################################################################################################

print('/-------------------------------------------------------------------------------------------------------- /')
print('Final Testing RESULTS')
print('/-------------------------------------------------------------------------------------------------------- /')
print('Accuracy is ', accuracy_score(y_test, y_pred))
print('Precision is ', precision_score(y_test, y_pred, average='weighted'))
print('Recall is ', recall_score(y_test, y_pred, average='weighted'))
print('F1-Score is ', f1_score(y_test, y_pred, average='weighted'))

print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))
