##sparkmlhw02
采用ML Pipelines构建一个文档分类器，需要将模型进行保存，并且加载模型后对测试样本进行预测，考查点：  
1）  spark读取文件  
2）  数据清洗，考查Datasets的基本操作  
3）  构建分类器的管道，考查构建各种转换操作  
4）  读取模型，读取测试数据，并且进行模型测试  
 
数据格式：  
myapp_id|typenameid|typename|myapp_word|myapp_word_all  


	scala> val ds1 = spark.read.options(Map(("sep","|"),("header","true"))).csv("/home/hadoop/Desktop/doc_class.dat")
	ds1: org.apache.spark.sql.DataFrame = [myapp_id: string, typenameid: string ... 3 more fields]
	
	scala> ds1.count()
	res1: Long = 334500  
	
	scala> ds1.show(1)
	+--------+----------+--------+--------------------+--------------------+
	|myapp_id|typenameid|typename|          myapp_word|      myapp_word_all|
	+--------+----------+--------+--------------------+--------------------+
	| 1376533|         2|  action|game, android, world|game, android, wo...|
	
	
	import org.apache.spark.ml.{Pipeline, PipelineModel}
	import org.apache.spark.ml.classification.LogisticRegression
	import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
	import org.apache.spark.ml.linalg.Vector
	import org.apache.spark.sql.Row
	
	scala> val tokenizer = new Tokenizer().setInputCol("myapp_word_all").setOutputCol("words")
	tokenizer: org.apache.spark.ml.feature.Tokenizer = tok_b46f36a01c75
	
	
	scala> val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol)setOutputCol("features")
	hashingTF: org.apache.spark.ml.feature.HashingTF = hashingTF_5005fdc59df6
	
	
	scala> import org.apache.spark.sql.types._
	import org.apache.spark.sql.types._
	
	scala> val training = ds1.withColumn("typenameid",col("typenameid").cast(IntegerType)).withColumnRenamed("typenameid","label")
	training: org.apache.spark.sql.DataFrame = [myapp_id: string, label: int ... 3 more fields]
	
	scala> val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.001)
	
	
	
	scala> val pipeline = new Pipeline().setStages(Array(tokenizer, hashingTF, lr))
	
	scala> val model = pipeline.fit(training)
	
	
	
	
	scala> model.write.overwrite().save("/tmp/spark-logistic-regression-model")
	2018-06-17 08:12:14 WARN  TaskSetManager:66 - Stage 25 contains a task of very large size (282 KB). The maximum recommended task size is 100 KB.
	
	// We can also save this unfit pipeline to disk
	scala> pipeline.write.overwrite().save("/tmp/unfit-lr-model")
	
	scala> val test = training.limit(4)
	test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [myapp_id: string, label: int ... 3 more fields]
	
	scala> test.count()
	res8: Long = 4
	
	// Make predictions on test documents.
	scala> val p = model.transform(test)
	p: org.apache.spark.sql.DataFrame = [myapp_id: string, label: int ... 8 more fields]
	
	
	scala> p.select("myapp_id","label","typename","probability","prediction").show()
	+--------+-----+--------+--------------------+----------+                       
	|myapp_id|label|typename|         probability|prediction|
	+--------+-----+--------+--------------------+----------+
	| 1376533|    2|  action|[1.95620442461929...|       2.0|
	| 1376542|    2|  action|[1.95620442461929...|       2.0|
	| 1376603|    2|  action|[1.95620442461929...|       2.0|
	| 1376792|    2|  action|[1.95620442461929...|       2.0|
	+--------+-----+--------+--------------------+----------+


