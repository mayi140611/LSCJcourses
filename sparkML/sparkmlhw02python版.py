>>> df1 = spark.read.csv("/home/hadoop/Desktop/doc_class.dat", sep='|', header=True)
2018-06-28 04:01:56 WARN  ObjectStore:568 - Failed to get database global_temp, returning NoSuchObjectException
>>> df1.count()
334500                                                                          
>>> df1.show(1)
+--------+----------+--------+--------------------+--------------------+
|myapp_id|typenameid|typename|          myapp_word|      myapp_word_all|
+--------+----------+--------+--------------------+--------------------+
| 1376533|         2|  action|game, android, world|game, android, wo...|
+--------+----------+--------+--------------------+--------------------+
only showing top 1 row
>>> from pyspark.ml import Pipeline, PipelineModel
>>> from pyspark.ml.classification import LogisticRegression
>>> from pyspark.ml.feature import HashingTF, Tokenizer
>>> from pyspark.sql import Row
>>> from pyspark.ml.linalg import Vector
>>> tokenizer = Tokenizer(inputCol='myapp_word_all', outputCol='words')
>>> hashingTF = HashingTF(numFeatures=1000, inputCol='words', outputCol='features')
training = df1.withColumnRenamed('typenameid','label')
>>> from pyspark.sql.functions import col
>>> from pyspark.sql.types import IntegerType

>>> training = training.withColumn('label', col('label').cast(IntegerType()))

>>> lr = LogisticRegression(maxIter=10, regParam=0.001)
>>> pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
>>> model = pipeline.fit(training)
>>> test = training.limit(4)
>>> p = model.transform(test)
>>> p.select("myapp_id","label","typename","probability","prediction").show()
+--------+-----+--------+--------------------+----------+
|myapp_id|label|typename|         probability|prediction|
+--------+-----+--------+--------------------+----------+
| 1376533|    2|  action|[1.95620442461924...|       2.0|
| 1376542|    2|  action|[1.95620442461924...|       2.0|
| 1376603|    2|  action|[1.95620442461924...|       2.0|
| 1376792|    2|  action|[1.95620442461924...|       2.0|
                          
#模型保存和加载
>>> model.save('/tmp/testModel')
2018-06-28 04:57:27 WARN  TaskSetManager:66 - Stage 31 contains a task of very large size (282 KB). The maximum recommended task size is 100 KB.
>>> savedModel = PipelineModel.load('/tmp/testModel')
>>> pipeline.save('/temp/pp')
>>> pp = Pipeline.load('/temp/pp')
