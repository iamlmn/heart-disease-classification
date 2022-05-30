from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col
from pyspark.sql.types import IntegerType,BooleanType,DateType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.classification import (LogisticRegression, RandomForestClassifier, NaiveBayes)
# from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

# If SparkSession already exists it returns otherwise create a new SparkSession.
spark = SparkSession.builder.appName('heart-disease-prediction').getOrCreate()

# load dataset
df = spark.read.csv('framingham.csv', inferSchema=True, header=True) 

#view five records
df.show(5)   

# print dataframe columns and count
print(df.columns)
print(df.count())

# print schema
df.printSchema()

# TODO : Add P-value finding signficant columns
def getSignificantColumns(df):
    ignore_cols = ['currentSmoker',  'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'diaBP', 'BMI', 'heartRate']
    df = df.drop("education")
    for c in ignore_cols:
        df = df.drop(c)
    return df
# Ignore insignificant columns
df = getSignificantColumns(df)

# get feature columns to scale.
inputCols = [col for col in df.columns if col != "TenYearCHD"]

# Check missing value for single column
# df.filter(df['age'].isNull()).show()

# Check missing value for all columns
df.select([count(when(isnan(c), c)).alias(c) for c in df.columns]).show()

# Drop null records
df = df.replace("NA",None)
df = df.na.drop()

# update df count
print(df.count())

# set schema right by making everything as Integer.
inputCols
stringCols = []
for col, type in df.dtypes:
	if type == 'string':
		stringCols.append(col)

stringCols

for col in stringCols: df = df.withColumn(col, df[col].cast(IntegerType()))

# print update schema
df.printSchema()

# Vectorize (Feature transformer — VectorAssembler) + Scale feature cols
assembler = VectorAssembler(inputCols=inputCols, outputCol="inputVector")
# imputer = Imputer(inputCols = "inputVector", outputCols=["inputVector_imputed"], strategy='mean')
scaler = MinMaxScaler(inputCol="inputVector", outputCol="scaled")
pipeline = Pipeline(stages=[assembler, scaler])
scalerModel = pipeline.fit(df)
scaledData = scalerModel.transform(df)   

# show scaled data
scaledData.show(5)
scaledData.select(["scaled","TenYearCHD"]).show(5)

# get required columns into df_final
df_final = scaledData.select(["scaled","TenYearCHD"])
df_final = df_final.withColumnRenamed("scaled", "features").withColumnRenamed("TenYearCHD", "label")
df_final.show()

# Split data into train and test
train_df, test_df = df_final.randomSplit([.75,.25])
train_df.count()
test_df.count()

# create models
lr = LogisticRegression(labelCol='label')
lr
rfc = RandomForestClassifier(labelCol='label', numTrees=100)
rfc
nb = NaiveBayes(labelCol='label')
nb

def getTrainTestAccuracy(train_predictions, string = "Test"):
    '''
    Return model accuracy for the given prediction set vs ground truth
    '''
    #Evaluate predictions under ROC probability curve for binary classification problems
    # evaluator = BinaryClassificationEvaluator()
    print("Training dataset Model accuracy \n")
    # print("Test Area Under ROC", evaluator.evaluate(train_predictions))
    accuracy = train_predictions.filter(train_predictions.label == train_predictions.prediction).count() / float(train_predictions.count())
    print(string + " Accuracy : ",accuracy)
    return accuracy

models = [lr, rfc, nb]
best_training_val, best_training_model = -1, None
best_test_val, best_test_model = -1, None

# train and test all models and get the best model.
for model in models:
    print("Model : ", model)
    trained_model = model.fit(train_df)
    train_predictions = trained_model.transform(train_df)
    # train_predictions.show(5)
    cv_training_acc = getTrainTestAccuracy(train_predictions, "Training")
    if cv_training_acc > best_training_val:
        best_training_val =cv_training_acc
        best_training_model = model
    # test predictions
    test_predictions = trained_model.transform(test_df)
    # test_predictions.show(5)
    cv_test_acc = getTrainTestAccuracy(test_predictions)
    if cv_test_acc > best_test_val:
        best_test_val = cv_test_acc
        best_test_model = model
    print('***************ENDing', model , '********************')


print("Best test result are found with model {} with accuracy of {}".format(best_test_model, best_test_val))

# TODO CONFUSION MATRIX

'''
# ignore below code

# # # decision tree

# # from pyspark.ml.classification import DecisionTreeClassifier
# # dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
# # dtModel = dt.fit(train_df)
# # dt_test_predictions = dtModel.transform(test_df)


# # eval

# evaluator = BinaryClassificationEvaluator()
# print("Test Area Under ROC: " + str(evaluator.evaluate(dt_test_predictions, {evaluator.metricName: "areaUnderROC"})))
'''