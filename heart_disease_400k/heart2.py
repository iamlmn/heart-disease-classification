from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, array, lit
from pyspark.sql.types import FloatType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
import numpy as np

#creating the spark session and setting up the pyspark data frames
spark = SparkSession.builder.appName('Heart-Disease-Prediction').getOrCreate()
df = spark.read.csv('project/heart_2020_cleaned.csv',inferSchema=True,header=True)
df.show(5)
df.count()
df.printSchema()

#checking for null values and removing them
df = df.na.drop()
df.count()

#selecting the numerical and categorical columns in the dataset
numericalCols = [col for col, type in df.dtypes if type != 'string'] 
categoricalCols = [col for col, type in df.dtypes if type == 'string']
categoricalCols.remove('HeartDisease') 
print(numericalCols, categoricalCols)

#splitting data into training and test datasets 
train_df, test_df = df.randomSplit([.75,.25])

#splitting the training dataset into major and minor datasets
major_df = train_df.filter(df.HeartDisease == 'No')
minor_df = train_df.filter(df.HeartDisease == 'Yes')

#random oversampling of minor dataset

def oversampling(major_df, minor_df):
    ratio = int(major_df.count()/minor_df.count())
    oversampled_minor_df = minor_df.withColumn('dummy', explode(array([lit(row) for row in range(ratio)]))).drop('dummy')
    return major_df.unionAll(oversampled_minor_df)

oversampled_df = oversampling(major_df, minor_df)

#data preparation before using classification models

#StringIndexer to convert categorical columns into label indices
string_indexers = [StringIndexer(inputCol=col, outputCol=col+'_index') for col in categoricalCols]

#OneHotEncoder to convert label indices into binary vectors
one_hot_encoders = [OneHotEncoder(inputCol=col+'_index', outputCol=col+'_vector') for col in categoricalCols]

#StringIndexer to convert label into label index
label_index = StringIndexer(inputCol='HeartDisease', outputCol='HeartDisease_index')

#VectorAssembler to transform multiple columns into a single vector column
inputCols = [col+'_vector' for col in categoricalCols] + numericalCols
vector_assembler = VectorAssembler(inputCols=inputCols, outputCol='features') 

#pipeline to implement all the stages of data preparation
pipeline = Pipeline(stages=string_indexers + one_hot_encoders + [label_index, vector_assembler])

#classification models - logistic regression, random forest and naive bayes
logistic_regression = LogisticRegression(featuresCol='features', labelCol='HeartDisease_index')
random_forest = RandomForestClassifier(featuresCol='features', labelCol='HeartDisease_index', numTrees=100)
naive_bayes = NaiveBayes(featuresCol='features', labelCol='HeartDisease_index')

#creating pipelines for the classification models
logistic_regression_pipeline = Pipeline(stages=[pipeline, logistic_regression])
random_forest_pipeline = Pipeline(stages=[pipeline, random_forest])
naive_bayes_pipeline = Pipeline(stages=[pipeline, naive_bayes])


#fitting models with the oversampled training dataset
logistic_regression_fit = logistic_regression_pipeline.fit(oversampled_df)
random_forest_fit = random_forest_pipeline.fit(oversampled_df)
naive_bayes_fit = naive_bayes_pipeline.fit(oversampled_df)


#predictions for the test dataset
logistic_regression_prediction = logistic_regression_fit.transform(test_df)
random_forest_prediction = random_forest_fit.transform(test_df)
naive_bayes_prediction = naive_bayes_fit.transform(test_df)


#results evaluation

#area_under_curve

AUC_prediction = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='HeartDisease_index')

AUC_logistic_regression = AUC_prediction.evaluate(logistic_regression_prediction)
AUC_random_forest = AUC_prediction.evaluate(random_forest_prediction)
AUC_naive_bayes = AUC_prediction.evaluate(naive_bayes_prediction)

print('Logistic Regression AUC: {:.2f}'.format(AUC_logistic_regression*100))
print('Random Forest AUC: {:.2f}'.format(AUC_random_forest*100))
print('Naive Bayes AUC: {:.2f}'.format(AUC_naive_bayes*100))


#accuracy

accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='HeartDisease_index', predictionCol="prediction", metricName="accuracy")

accuracy_logistic_regression = accuracy_evaluator.evaluate(logistic_regression_prediction)
accuracy_random_forest = accuracy_evaluator.evaluate(random_forest_prediction)
accuracy_naive_bayes = accuracy_evaluator.evaluate(naive_bayes_prediction)
 
print('Logistic Regression accuracy: {:.2f}'.format(accuracy_logistic_regression*100))
print('Random Forest accuracy: {:.2f}'.format(accuracy_random_forest*100))
print('Naive Bayes accuracy: {:.2f}'.format(accuracy_naive_bayes*100))


#confusion matrices

def confusion_matrix(prediction_df):
    prediction_labels = prediction_df.select(['prediction','HeartDisease_index']).withColumn('HeartDisease_index', F.col('HeartDisease_index').cast(FloatType())).orderBy('prediction')
    prediction_labels = prediction_labels.select(['prediction','HeartDisease_index'])
    metrics = MulticlassMetrics(prediction_labels.rdd.map(tuple))
    return metrics.confusionMatrix().toArray()


confusion_matrix_logistic_regression = confusion_matrix(logistic_regression_prediction)
confusion_matrix_random_forest = confusion_matrix(random_forest_prediction)
confusion_matrix_naive_bayes = confusion_matrix(naive_bayes_prediction)


#sensitivity

def sensitivity(conf_matrix):
    TP = conf_matrix[1][1]
    FN = conf_matrix[1][0]
    return TP/(TP+FN)


print('Logistic Regression sensitivity: {}'.format((sensitivity(confusion_matrix_logistic_regression)*100).round(2)))
print('Random Forest sensitivity: {}'.format((sensitivity(confusion_matrix_random_forest)*100).round(2)))
print('Naive Bayes sensitivity: {}'.format((sensitivity(confusion_matrix_naive_bayes)*100).round(2)))

