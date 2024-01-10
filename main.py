from helpers import download_file, extract_file, input_Wset_Lset, positions
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as px
px.init_notebook_mode(connected=True)
px.offline.init_notebook_mode(connected=True)
import plotly.express as px
import os.path as osp
import os
import logging
from glob import glob
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error 
  
logging.getLogger().setLevel('INFO')


download_file
extract_file

BASE_URL = 'http://tennis-data.co.uk'
DATA_DIR = "tennis_data"
ATP_DIR = './{}/ATP'.format(DATA_DIR)


ATP_URLS = [BASE_URL + "/%i/%i.zip" % (i,i) for i in range(2000,2022)]
os.makedirs(osp.join(ATP_DIR, 'archives'), exist_ok=True)


for files, directory in ((ATP_URLS, ATP_DIR)):
    for dl_path in files:
        logging.info("downloading & extracting file %s", dl_path)
        archive_path = osp.join(directory, 'archives', osp.basename(dl_path))
        download_file(dl_path, archive_path)
        extract_file(archive_path, directory)
    
ATP_FILES = sorted(glob("%s/*.xls*" % ATP_DIR))

df_atp = pd.concat([pd.read_excel(f) for f in ATP_FILES], ignore_index=True)

logging.info("%i matches ATP in df_atp", df_atp.shape[0])

df_atp['Lsets'].replace('`1', 1,inplace = True)


input_Wset_Lset

positions
print(df_atp[df_atp["Winner_position"] == 1].shape[0])
print(df_atp[df_atp["Winner_position"] == 0].shape[0])

df_atp_X = df_atp.loc[:, ['AvgL', 'AvgW', 'B&WL', 'B&WW', 'B365L', 'B365W', 'CBL', 'CBW', 'EXL', 'EXW', 'GBL', 'GBW', \
    'IWL', 'IWW','LBL', 'LBW', 'LRank', 'MaxL', 'MaxW', 'PSL', 'PSW', \
    'SBL', 'SBW', 'SJL', 'SJW', 'UBL', 'UBW', 'WRank','Best of', \
    "Date",'ATP','Series','Court','Surface', 'Winner_position']]



df_atp_X.iloc[:, :-4] = df_atp_X.apply(pd.to_numeric, errors='coerce') 
df_atp_X["WRank"] = df_atp_X["WRank"].fillna(df_atp_X["WRank"].max())
df_atp_X["LRank"] = df_atp_X["LRank"].fillna(df_atp_X["LRank"].max())

cols_1=["AvgL", "AvgW", "B&WL", "B&WW", "B365L", "B365W", "CBL", "CBW", "EXL", "EXW", "GBL", "GBW", "IWL", "IWW",
    "LBL", "LBW", "MaxL", "MaxW", "PSL", "PSW", "SBL", "SBW", "SJL", "SJW", "UBL", "UBW"]
df_atp_X[cols_1]=df_atp_X[cols_1].fillna(1.0)

def assing():
    df_atp_X["P1Rank"] = df_atp_X.apply(lambda row: row["WRank"] if row["Winner_position"] == 1 else row["LRank"], axis=1)
    df_atp_X["P0Rank"] = df_atp_X.apply(lambda row: row["WRank"] if row["Winner_position"] == 0 else row["LRank"], axis=1)
    df_atp_X=df_atp_X.drop("WRank", axis=1)
    df_atp_X=df_atp_X.drop("LRank", axis=1)
    for cols in ( ('AvgL', 'AvgW'), ('B&WL', 'B&WW'), ('B365L', 'B365W'), ('CBL', 'CBW'), ('EXL', 'EXW'), ('GBL', 'GBW'), \
    ('IWL', 'IWW'),('LBL', 'LBW'), ('MaxL', 'MaxW'), ('PSL', 'PSW'), \
    ('SBL', 'SBW'), ('SJL', 'SJW'), ('UBL', 'UBW')):
        suffix=cols[1][:-1]
        df_atp_X["P1"+suffix] = df_atp_X.apply(lambda row: row[cols[1]] if row["Winner_position"] == 1 else row[cols[0]], axis=1)
        df_atp_X["P0"+suffix] = df_atp_X.apply(lambda row: row[cols[1]] if row["Winner_position"] == 0 else row[cols[0]], axis=1)
        df_atp_X=df_atp_X.drop(cols[0], axis=1)
        df_atp_X=df_atp_X.drop(cols[1], axis=1) 

column_names_for_onehot = df_atp_X.columns[3:6]
encoded_atp_df = pd.get_dummies(df_atp_X, columns=column_names_for_onehot, drop_first=True)

def feature_engineer():
    encoded_atp_df['Date'] = pd.to_datetime(encoded_atp_df['Date'], format = '%Y-%m-%dT', errors = 'coerce')
    encoded_atp_df['Date_year'] = encoded_atp_df['Date'].dt.year
    encoded_atp_df['Date_month'] = encoded_atp_df['Date'].dt.month
    encoded_atp_df['Date_week'] = encoded_atp_df['Date'].dt.week
    encoded_atp_df['Date_day'] = encoded_atp_df['Date'].dt.day
    encoded_atp_df=encoded_atp_df.drop("Date", axis=1)



f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(encoded_atp_df.drop("Winner_position", axis=1).corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

corr_matrix = encoded_atp_df.drop("Winner_position", axis=1).corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.8
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
print("Features to be dropped: ",to_drop)
# Drop features 
encoded_atp_df.drop(to_drop, axis=1, inplace=True)


encoded_atp_df.hist()
plt.gcf().set_size_inches(25, 25)
sns.set(color_codes=True)
encoded_atp_df = encoded_atp_df.loc[:,encoded_atp_df.apply(pd.Series.nunique) != 1]

year_to_predict = 2019

df_train = encoded_atp_df.iloc[df_atp[df_atp["Date"].dt.year != year_to_predict].index]
df_test = encoded_atp_df.iloc[df_atp[df_atp["Date"].dt.year == year_to_predict].index]

X_train = df_train.drop(["Winner_position"], axis=1)
y_train = df_train["Winner_position"]

X_test = df_test.drop(["Winner_position"], axis=1)
y_test = df_test["Winner_position"]

print("Training Set Shape: ",X_train.shape,  y_train.shape)
print("Test Set Shape:     ",X_test.shape,  y_test.shape)


sc = StandardScaler()  
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


names_of_classifier = ["Linear SVM", "LogisticRegression"]

classifier = [
    
    SVC(kernel="linear", C=0.03),
    LogisticRegression(multi_class='over', random_state=0, solver='liblinear', penalty='12')
]

for name, classifier in zip(names_of_classifier, classifier):
    classifier.fit(X_train_scaled,y_train)
    
    y_predict=classifier.predict(X_test_scaled)
    y_Train_predict=classifier.predict(X_train_scaled)
    print("Classifier: ",name)
    print("\nAccuracy for Test Set: ",accuracy_score(y_test, y_predict))
    print( "Mean Squared Error for Test Set: ",round(mean_squared_error(y_test,y_predict), 3))
    print("Confusion matrix for Test Set \n",confusion_matrix(y_test,y_predict))
    print(classification_report(y_test,y_predict))
    fpr, tpr, thresholds= metrics.roc_curve(y_test,y_predict)
    auc = metrics.roc_auc_score(y_test,y_predict, average='macro', sample_weight=None)
    print("ROC Curve for for Test Set \n")
    sns.set_style('darkgrid')
    sns.lineplot(fpr,tpr,color ='blue')
    plt.show()
    
    
    print("\nAccuracy for Train Set: ",accuracy_score(y_train, y_Train_predict))
    print( "Mean Squared Error for Train Set: ",round(mean_squared_error(y_train,y_Train_predict), 3))
    print("Confusion matrix for Train Set \n",confusion_matrix(y_train,y_Train_predict))
    print(classification_report(y_train,y_Train_predict))
    fpr_train, tpr_train, thresholds_train= metrics.roc_curve(y_train,y_Train_predict)
    auc_train = metrics.roc_auc_score(y_train,y_Train_predict, average='macro', sample_weight=None)
    print("ROC Curve for for Train Set \n")
    sns.set_style('darkgrid')
    sns.lineplot(fpr_train,tpr_train,color ='red')
    plt.show()
    
    print("--------------------------------------xxx--------------------------------------\n\n")
    

