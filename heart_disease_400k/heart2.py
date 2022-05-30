#imports and setup
from pyspark.sql import SparkSession
from pyspark.ml.feature import (VectorAssembler, OneHotEncoder, StringIndexer)
from pyspark.ml import Pipeline
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, NaiveBayes)
from pyspark.sql.functions import (col, explode, array, lit)
from pyspark.ml.evaluation import (BinaryClassificationEvaluator, MulticlassClassificationEvaluator)
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
 

import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
 
spark = SparkSession.builder.appName('HeartDiseaseClassification').getOrCreate()

df = spark.read.csv('heart_2020_cleaned.csv',inferSchema=True,header=True)
display(df.head(5))

#Schema of the table
df.printSchema()



label = 'HeartDisease'
numerical_cols = ['BMI', 'PhysicalHealth','MentalHealth','SleepTime']
categorical_cols = list(set(df.columns) - set(numerical_cols) -set([label]))


# data distrivuton
# stats of numerical variables
df.select(numerical_cols).describe().show()


# check number of observations of differente samples
# df.groupBy(label).count().toPandas().plot.bar(x='HeartDisease', rot=0, title='Number of Observations per label')


# Preparing Data for Classification Models
'''
Working with an Unbalanced Dataset. Oversampling Smallest Class
As can be seen above, this dataset is extremely imbalanced. This is frequent in disease-related datasets.

In this section, I will perform oversampling on the smaller class to lessen the bias of the classification models.
'''

#splitting data into train and test sets before Oversampling
train_df, test_df = df.randomSplit([.7,.3])


#spliting df by classes
major_df = train_df.filter(col(label) == 'No')
minor_df = train_df.filter(col(label) == 'Yes')
#ratio of number observation major vs minor class
r = int(major_df.count()/minor_df.count())
 
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in range(r)]))).drop('dummy')
 
# combine both oversampled minority rows and previous majority rows 
combined_train_df = major_df.unionAll(oversampled_df)


# combined_train_df.groupBy(label).count().toPandas().plot.bar(x='HeartDisease', rot=0, title='Number of Observations in Train subset after Oversampling')



# Processing Categorical Columns for Spark Pipeline

'''
String columns cannot be used as input to Spark. To address this, I'll need to employ an indexer on these columns, followed by an encoding.

I also need to vectorize all the features into a features column after I have all columns with numerical values.

'''

# Indexers for categorical columns
indexers = [StringIndexer(inputCol=col, outputCol=col+'_indexed') for col in categorical_cols]
# Encoders for categorical columns
encoders = [OneHotEncoder(inputCol=col+'_indexed', outputCol=col+'_encoded') for col in categorical_cols]
 
# Indexer for classification label:
label_indexer = StringIndexer(inputCol=label, outputCol=label+'_indexed')


#assemble all features as vector to be used as input for Spark MLLib
assembler = VectorAssembler(inputCols= [col+'_encoded' for col in categorical_cols] + numerical_cols, outputCol='features') 


# Creating data processing pipeline
pipeline = Pipeline(stages= indexers + encoders + [label_indexer, assembler])


# Applying Classification Models
# Models Implemented:
# lr - Logistic Regression
# rfc - Random Forest Classifier
# nb - Naive Bayes


lr = LogisticRegression(featuresCol='features', labelCol=label+'_indexed')
rfc = RandomForestClassifier(featuresCol='features', labelCol=label+'_indexed', numTrees=100)
nb = NaiveBayes(featuresCol='features', labelCol=label+'_indexed')



# creating pipelines with machine learning models
pipeline_lr = Pipeline(stages=[pipeline, lr])
pipeline_rfc = Pipeline(stages=[pipeline, rfc])
pipeline_nb = Pipeline(stages=[pipeline, nb])


#fitting models with train subset
lr_fit = pipeline_lr.fit(combined_train_df)
rfc_fit = pipeline_rfc.fit(combined_train_df)
nb_fit = pipeline_nb.fit(combined_train_df)


# predictions for test subset
pred_lr = lr_fit.transform(test_df)
pred_rfc = rfc_fit.transform(test_df)
pred_nb = nb_fit.transform(test_df)


# Evaluating Results

'''
Area Under Curve - AUC
The AUC of a random selection of labels is 0.5. The closer this metric is to one, the better the model predicts the data labels.

Regarding this metric, the logistic regression model outperforms the Random Forest Classifier. The Naive Bayes Classifier performed the poorest.
'''

pred_AUC = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol=label+'_indexed')

AUC_lr = pred_AUC.evaluate(pred_lr)
AUC_rfc = pred_AUC.evaluate(pred_rfc)
AUC_nb = pred_AUC.evaluate(pred_nb)
print(AUC_lr, AUC_rfc, AUC_nb)

# Accuracy - A poor Evaluation Metric for Unbalanced Classification


'''
Accuracy is a common metric used when evaluating classification problems. It is calculated by

\frac{TP + TN}{\textit{All Samples}} 
All Samples
TP+TN
​
 
Where TP = True Positives and TN = True Negatives

Note that for this particular case this is not the best metric because the Negative label represents the grand majority of the observation.

As an extreme example, if I predicted that all observations would be negative for heart disease, the accuracy for this test subgroup would be 91.48 percent.

Looking at the results for these three models, the Naive Bayes has the best accuracy while being the lowest performing model in terms of TP. When analyzing these models, special emphasis should be placed on the TP cases.
'''

# calculating accuracy for all negative prediction mentioned above
acc_all_negative = test_df.filter(test_df[label]=='No').count() / test_df.count()
acc_all_negative


acc_evaluator = MulticlassClassificationEvaluator(labelCol=label+'_indexed', predictionCol="prediction", metricName="accuracy")


acc_lr = acc_evaluator.evaluate(pred_lr)
acc_rfc = acc_evaluator.evaluate(pred_rfc)
acc_nb = acc_evaluator.evaluate(pred_nb)
 
print('Logistic Regression accuracy: {:.2f}'.format(acc_lr*100))
print('Random Forest accuracy: {:.2f}'.format(acc_rfc*100))
print('Naive Bayes accuracy: {:.2f}'.format(acc_nb*100))


# Confusion Matrices/

def confusion_matrix(pred_df):
    preds_labels = pred_df.select(['prediction',label+'_indexed']).withColumn(label+'_indexed', F.col(label+'_indexed').cast(FloatType())).orderBy('prediction')
    preds_labels = preds_labels.select(['prediction',label+'_indexed'])
    metrics = MulticlassMetrics(preds_labels.rdd.map(tuple))
    return metrics.confusionMatrix().toArray()



# def confusion_matrix_plot(conf_mat, ax, title = 'Confusion Matrix'):
#     names = ['True Negative','False Positive','False Negative','True Positive']
 
#     number = ["{0:0.0f}".format(value) for value in conf_mat.flatten()]
 
#     percent = ["{0:.2%}".format(value) for value in conf_mat.flatten()/np.sum(conf_mat)]
 
#     labels = [f"{v1}\n\n{v2}\n\n{v3}" for v1, v2, v3 in zip(names, number, percent)]
 
#     labels = np.asarray(labels).reshape(2,2)
 
#     ax = sns.heatmap(conf_mat, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax)
 
#     ax.set_title(title+'\n');
#     ax.set_xlabel('\nPredicted Labels')
#     ax.set_ylabel('Real Labels');
 
#     ax.xaxis.set_ticklabels(['No','Yes'])
#     ax.yaxis.set_ticklabels(['No','Yes'])
    
#     return ax


conf_lr = confusion_matrix(pred_lr)
conf_rfc = confusion_matrix(pred_rfc)
conf_nb = confusion_matrix(pred_nb)


# fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
 
# ax1 = confusion_matrix_plot(conf_lr, ax1,'Logistic Regression - Confusion Matrix')
# ax2 = confusion_matrix_plot(conf_rfc, ax2,'Random Forest Classifier - Confusion Matrix')
# ax3 = confusion_matrix_plot(conf_nb, ax3, 'Naive Bayes - Confusion Matrix')
 
# plt.show()


# Sensitivity Metric
'''
Sensitivity is the True Positive Rate of the classification:

\frac{TP}{TP + FN} 
TP+FN
TP
​
 
where TP = True Positive and FN = False Negative.

It is a measure of how well the Positive label is predicted.
'''


def sensitivity(conf_mat):
    TP = conf_mat[1][1]
    FN = conf_mat[1][0]
    result = TP / (TP + FN)
    return result


print('Logistic Regression sensitivity: {}'.format((sensitivity(conf_lr)*100).round(2)))
print('Random Forest sensitivity: {}'.format((sensitivity(conf_rfc)*100).round(2)))
print('Naive Bayes sensitivity: {}'.format((sensitivity(conf_nb)*100).round(2)))

# Results

'''


The best performing model was Logistic Regression;
The true positive rate was 77%. This indicates that 77 percent of heart disease patients were appropriately identified;
The model's False Positive rate (or Specificity) is high, although lowering this statistic is not the primary goal.
Overall, the Logistic Regression model yields useful results (sensitivity higher than 50% = better than random guess). Despite this, 77 percent is still a low percentage for classification algorithms.

'''