##hbasehw11
练习HBase备份操作， distcp, export/import, CopyTable.  

----------
###用distcp进行备份与恢复
任务：将hbase中的table2表删除后恢复  

	```
	hbase(main):004:0> scan 'table2'
	ROW                                     COLUMN+CELL                                                                                                        
	 rk001                                  column=info:name, timestamp=1532504076816, value=zhangsan                                                          
	 rk002                                  column=info:name, timestamp=1532504088934, value=zhangsan2                                                         
	 rk003                                  column=info:name, timestamp=1532504096930, value=zhangsan3                                                         
	 rk004                                  column=info:name, timestamp=1532504468927, value=zhangsan4                                                         
	 rk005                                  column=info:name, timestamp=1534485871811, value=zhangsan5                                                         
	 rk007                                  column=info:name, timestamp=1534486147906, value=zhangsan71                                                        
	 rk008                                  column=info:name, timestamp=1532504076817, value=zhangsan81                                                        
	7 row(s) in 0.0820 seconds
	
	hbase(main):005:0> list
	TABLE                                                                                                                                                      
	mysql_hbase                                                                                                                                                
	mytable                                                                                                                                                    
	mytable2                                                                                                                                                   
	oracle_hbase                                                                                                                                               
	table2                                                                                                                                                     
	tablesplit                                                                                                                                                 
	tbl_1                                                                                                                                                      
	tbl_2                                                                                                                                                      
	test4 
	
	[ian@h1 ~]$ stop-hbase.sh     
	[ian@h1 ~]$ hadoop distcp /hbase /hbasebackup
	[ian@h1 ~]$ hdfs dfs -ls /
	Found 19 items
	drwxr-xr-x   - ian supergroup          0 2018-08-26 14:38 /checkpoints
	drwxr-xr-x   - ian supergroup          0 2018-08-31 17:16 /hbase
	drwxr-xr-x   - ian supergroup          0 2018-09-05 15:45 /hbasebackup
	
	[ian@h1 ~]$ start-hbase.sh 
	[ian@h1 ~]$ hbase shell
	
	hbase(main):002:0> disable 'table2'
	0 row(s) in 2.6370 seconds
	
	hbase(main):003:0> drop 'table2'
	0 row(s) in 2.3420 seconds        
	
	
	[ian@h1 ~]$ stop-hbase.sh    
	[ian@h1 ~]$ hdfs dfs -mv /hbase /hbasetmp
	[ian@h1 ~]$ hadoop distcp -overwrite /hbasebackup /hbase
	
	[ian@h1 ~]$ start-hbase.sh 
	[ian@h1 ~]$ hbase shell
	
	hbase(main):001:0> list
	TABLE                                                                                                                                                      
	mysql_hbase                                                                                                                                                
	mytable                                                                                                                                                    
	mytable2                                                                                                                                                   
	oracle_hbase                                                                                                                                               
	table2                                                                                                                                                     
	tablesplit                                                                                                                                                 
	tbl_1                                                                                                                                                      
	tbl_2                                                                                                                                                      
	test4                                                                                                                                                      
	9 row(s) in 0.1840 seconds
	
	=> ["mysql_hbase", "mytable", "mytable2", "oracle_hbase", "table2", "tablesplit", "tbl_1", "tbl_2", "test4"]
	hbase(main):002:0> scan 'table2'
	ROW                                     COLUMN+CELL                                                                                                        
	 rk001                                  column=info:name, timestamp=1532504076816, value=zhangsan                                                          
	 rk002                                  column=info:name, timestamp=1532504088934, value=zhangsan2                                                         
	 rk003                                  column=info:name, timestamp=1532504096930, value=zhangsan3                                                         
	 rk004                                  column=info:name, timestamp=1532504468927, value=zhangsan4                                                         
	 rk005                                  column=info:name, timestamp=1534485871811, value=zhangsan5                                                         
	 rk007                                  column=info:name, timestamp=1534486147906, value=zhangsan71                                                        
	 rk008                                  column=info:name, timestamp=1532504076817, value=zhangsan81                                                             
	```
###export/import
任务：export table2表，清空table2表，然后import恢复  

	```
	[ian@h1 ~]$ hbase org.apache.hadoop.hbase.mapreduce.Export table2 /tmp/table2
	[ian@h1 ~]$ hdfs dfs -ls /tmp
	Found 3 items
	drwxrwx---   - ian supergroup          0 2018-08-12 12:40 /tmp/hadoop-yarn
	drwx-wx-wx   - ian supergroup          0 2018-08-03 11:58 /tmp/hive
	drwxr-xr-x   - ian supergroup          0 2018-09-05 15:59 /tmp/table2
	
	#清空表数据
	hbase(main):001:0> scan 'table2'
	ROW                                     COLUMN+CELL                                                                                                        
	 rk001                                  column=info:name, timestamp=1532504076816, value=zhangsan                                                          
	 rk002                                  column=info:name, timestamp=1532504088934, value=zhangsan2                                                         
	 rk003                                  column=info:name, timestamp=1532504096930, value=zhangsan3                                                         
	 rk004                                  column=info:name, timestamp=1532504468927, value=zhangsan4                                                         
	 rk005                                  column=info:name, timestamp=1534485871811, value=zhangsan5                                                         
	 rk007                                  column=info:name, timestamp=1534486147906, value=zhangsan71                                                        
	 rk008                                  column=info:name, timestamp=1532504076817, value=zhangsan81                                                        
	7 row(s) in 0.2670 seconds
	
	hbase(main):002:0> truncate 'table2'
	Truncating 'table2' table (it may take a while):
	 - Disabling table...
	 - Truncating table...
	0 row(s) in 4.7890 seconds
	
	hbase(main):003:0> scan 'table2'
	ROW                                     COLUMN+CELL                                                                                                        
	0 row(s) in 0.1420 seconds
	
	###恢复数据
	[ian@h1 ~]$ hbase org.apache.hadoop.hbase.mapreduce.Import table2 /tmp/table2
	hbase(main):001:0> scan 'table2'
	ROW                                     COLUMN+CELL                                                                                                        
	 rk001                                  column=info:name, timestamp=1532504076816, value=zhangsan                                                          
	 rk002                                  column=info:name, timestamp=1532504088934, value=zhangsan2                                                         
	 rk003                                  column=info:name, timestamp=1532504096930, value=zhangsan3                                                         
	 rk004                                  column=info:name, timestamp=1532504468927, value=zhangsan4                                                         
	 rk005                                  column=info:name, timestamp=1534485871811, value=zhangsan5                                                         
	 rk007                                  column=info:name, timestamp=1534486147906, value=zhangsan71                                                        
	 rk008                                  column=info:name, timestamp=1532504076817, value=zhangsan81                                                        
	7 row(s) in 0.2650 seconds
	```
###CopyTable
将hbase的表table2的数据拷贝到表table2copy中  

	```
	hbase(main):002:0> create 'table2copy','info'
	0 row(s) in 2.4220 seconds
	
	=> Hbase::Table - table2copy
	hbase(main):003:0> scan 'table2copy'
	ROW                                     COLUMN+CELL                                                                                                        
	0 row(s) in 0.0800 seconds
	
	[ian@h1 ~]$ hbase org.apache.hadoop.hbase.mapreduce.CopyTable --new.name=table2copy table2
	
	
	hbase(main):001:0> scan 'table2copy'
	ROW                                     COLUMN+CELL                                                                                                        
	 rk001                                  column=info:name, timestamp=1532504076816, value=zhangsan                                                          
	 rk002                                  column=info:name, timestamp=1532504088934, value=zhangsan2                                                         
	 rk003                                  column=info:name, timestamp=1532504096930, value=zhangsan3                                                         
	 rk004                                  column=info:name, timestamp=1532504468927, value=zhangsan4                                                         
	 rk005                                  column=info:name, timestamp=1534485871811, value=zhangsan5                                                         
	 rk007                                  column=info:name, timestamp=1534486147906, value=zhangsan71                                                        
	 rk008                                  column=info:name, timestamp=1532504076817, value=zhangsan81                                                        
	7 row(s) in 0.2730 seconds
	```








