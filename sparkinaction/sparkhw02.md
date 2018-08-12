##sparkhw02

###1. 通过spark-submit提交org.apache.spark.examples.JavaSparkPi，并将DAG截图。要求将Spark应用提交到Yarn上（即--master=yarn）。
	`[ian@h1 conf]$ spark-submit --master yarn \
	--name javasparkpitest \
	--num-executors 6 \
	--class org.apache.spark.examples.JavaSparkPi 
	$SPARK_HOME/examples/jars/spark-examples_2.11-2.3.1.jar`

![image.png](https://upload-images.jianshu.io/upload_images/2850424-378802d97a238335.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/2850424-bda7601f2b62fa8c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/2850424-bd9fe63dd2d9ac5e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


###2. 在IDE中使用Java实现WordCount，Partition数设置为3（也即numSlices=3），并过滤掉部分单词（如 of  with）。使用单步调试，验证同一Partition(或者说task)内的不同数据，是保证每条数据从前到后完全处理完，再处理下一条数据，还是对所有数据同时进行某种操作，结合后再进行某种操作。
>老师，我使用的是python.    
>根据课上所讲，由于map,flatMap,filter都是Transformation类型的算子  
>应该是：1）对A顺序执行map,flatMap,filter，然后对B顺序执行map,flatMap,filter，最后对C顺序执行map,flatMap,filter  
>
>python代码如下

	from __future__ import print_function
	
	import sys
	from operator import add
	
	from pyspark.sql import SparkSession
	
	
	lines = spark.read.text('hdfs://h1:9000/test.py')
	
	type(lines)
	
	
	#将df对象转成rdd对象，每一行为一个rdd元素
	rdd1 = lines.rdd
	
	rdd1.getNumPartitions()
	
	rdd2 = rdd1.map(lambda r: r[0])
	
	
	rdd3 = rdd2.flatMap(lambda x: x.split(' '))
	
	
	#过滤掉一些常见词汇
	rdd4 = rdd3.filter(lambda x: x not in ['=','of','a','#','with'])
	
	
	rdd5 = rdd4.map(lambda x: (x, 1))
	
	
	rdd6 = rdd5.reduceByKey(add)
	
	
	counts = lines.flatMap(lambda x: x.split(' '))                   .map(lambda x: (x, 1))                   .reduceByKey(add)
	
	
	output = rdd6.collect()
	
	for (word, count) in output:
	        print("%s: %i" % (word, count))














