# heart-disease-classification


Binary Classifier model buit for https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression/code?datasetId=222487&sortBy=voteCount
using Pyspark with Logisitc Regression , RandomForestClassifier & Naive Bayes.


To run, download data and load to HDFS by ``` hdfs dfs -copyFromLocal framingham.csv .
sprak-submit classifier.py


TODO: Fill find significant column finder function with p-value calc,  Confusion matrix in the end after predictions.