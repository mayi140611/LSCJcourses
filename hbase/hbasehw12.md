##hbasehw12
完成课程中的命令练习测试

## 1 快照操作
### 1.1 配置快照
在HBase的配置文件hbase-site.xml中配置快照

	<property>
	<name>hbase.snapshot.enabled</name>
	<value>true</value>
	</property>
### 1.2 创建快照、查看快照

	hbase(main):005:0> snapshot 'table2','table2snapshot'
	0 row(s) in 0.7820 seconds
	
	hbase(main):006:0> list_snapshots
	SNAPSHOT                       TABLE + CREATION TIME                                                                   
	 table2snapshot                table2 (Thu Sep 13 09:27:06 +0800 2018) 

查看在hdfs上生成的文件  
![image.png](https://upload-images.jianshu.io/upload_images/2850424-f50bbdc78de412f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/840)
### 1.3 利用快照克隆表

	hbase(main):007:0> clone_snapshot 'table2snapshot', 'table21'
	0 row(s) in 0.8830 seconds
	
	hbase(main):008:0> scan 'table21'
	ROW                            COLUMN+CELL                                                                             
	 rk001                         column=info:name, timestamp=1532504076816, value=zhangsan                               
	 rk002                         column=info:name, timestamp=1532504088934, value=zhangsan2                              
	 rk003                         column=info:name, timestamp=1532504096930, value=zhangsan3                              
	 rk004                         column=info:name, timestamp=1532504468927, value=zhangsan4                              
	 rk005                         column=info:name, timestamp=1534485871811, value=zhangsan5                              
	 rk007                         column=info:name, timestamp=1534486147906, value=zhangsan71                             
	 rk008                         column=info:name, timestamp=1532504076817, value=zhangsan81                             
	7 row(s) in 0.1220 seconds

### 1.4 利用快照恢复表

	hbase(main):009:0> put 'table2','rk009','info:name','zhangsan9'
	0 row(s) in 0.1150 seconds
	
	hbase(main):010:0> scan 'table2'
	ROW                            COLUMN+CELL                                                                             
	 rk001                         column=info:name, timestamp=1532504076816, value=zhangsan                               
	 rk002                         column=info:name, timestamp=1532504088934, value=zhangsan2                              
	 rk003                         column=info:name, timestamp=1532504096930, value=zhangsan3                              
	 rk004                         column=info:name, timestamp=1532504468927, value=zhangsan4                              
	 rk005                         column=info:name, timestamp=1534485871811, value=zhangsan5                              
	 rk007                         column=info:name, timestamp=1534486147906, value=zhangsan71                             
	 rk008                         column=info:name, timestamp=1532504076817, value=zhangsan81                             
	 rk009                         column=info:name, timestamp=1536802661998, value=zhangsan9                              
	8 row(s) in 0.0350 seconds
	
	hbase(main):011:0> disable 'table2'
	0 row(s) in 2.4300 seconds
	
	hbase(main):012:0> restore_snapshot 'table2snapshot'
	0 row(s) in 0.7870 seconds
	
	hbase(main):014:0> scan 'table2'
	ROW                            COLUMN+CELL                                                                             
	 rk001                         column=info:name, timestamp=1532504076816, value=zhangsan                               
	 rk002                         column=info:name, timestamp=1532504088934, value=zhangsan2                              
	 rk003                         column=info:name, timestamp=1532504096930, value=zhangsan3                              
	 rk004                         column=info:name, timestamp=1532504468927, value=zhangsan4                              
	 rk005                         column=info:name, timestamp=1534485871811, value=zhangsan5                              
	 rk007                         column=info:name, timestamp=1534486147906, value=zhangsan71                             
	 rk008                         column=info:name, timestamp=1532504076817, value=zhangsan81                             
	7 row(s) in 0.1820 seconds

### 1.5 删除快照

	hbase(main):015:0> delete
	delete                delete_all_snapshot   delete_snapshot       deleteall
	hbase(main):015:0> delete_snapshot 'table2snapshot'
	0 row(s) in 0.0420 seconds
	
	hbase(main):016:0> list_snapshots
	SNAPSHOT                       TABLE + CREATION TIME                                                                   
	0 row(s) in 0.0170 seconds
	
	=> []

## 2 hbase集群复制

	hbase(main):017:0> list_peers
	 PEER_ID CLUSTER_KEY STATE TABLE_CFS
	0 row(s) in 0.1450 seconds
	
	hbase(main):018:0> status 'replication'
	version 1.2.6.1
	4 live servers
	    h3:
	       SOURCE:
	       SINK  : AgeOfLastAppliedOp=0, TimeStampsOfLastAppliedOp=Thu Sep 13 09:25:16 CST 2018
	    h2:
	       SOURCE:
	       SINK  : AgeOfLastAppliedOp=0, TimeStampsOfLastAppliedOp=Thu Sep 13 09:25:16 CST 2018
	    h1:
	       SOURCE:
	       SINK  : AgeOfLastAppliedOp=0, TimeStampsOfLastAppliedOp=Thu Sep 13 09:25:16 CST 2018
	    h4:
	       SOURCE:
	       SINK  : AgeOfLastAppliedOp=0, TimeStampsOfLastAppliedOp=Thu Sep 13 09:25:17 CST 2018



