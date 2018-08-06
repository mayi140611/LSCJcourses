##HBASEhw07
按文教程中操作指引，完成TTL试验

	[ian@h1 ~]$ hbase shell

	Version 1.2.6.1, rUnknown, Sun Jun  3 23:19:26 CDT 2018
	
	hbase(main):001:0> create 'tbl_1','cf1'
	0 row(s) in 2.4880 seconds
	
	=> Hbase::Table - tbl_1
	hbase(main):002:0> list
	TABLE                                                                                                               
	mytable                                                                                                             
	mytable2                                                                                                            
	table2                                                                                                              
	tablesplit                                                                                                          
	tbl_1                                                                                                               
	5 row(s) in 0.0330 seconds
	
	=> ["mytable", "mytable2", "table2", "tablesplit", "tbl_1"]

	hbase(main):003:0> describe 'tbl_1'
	Table tbl_1 is ENABLED                                                                                              
	tbl_1                                                                                                               
	COLUMN FAMILIES DESCRIPTION                                                                                         
	{NAME => 'cf1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'FALSE', DATA_BLO
	CK_ENCODING => 'NONE', TTL => 'FOREVER', COMPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE
	 => '65536', REPLICATION_SCOPE => '0'}                                                                              
	1 row(s) in 0.1340 seconds
	
	hbase(main):004:0> alter 'tbl_1',NAME=>'cf1',TTL=>60
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	0 row(s) in 2.2660 seconds
	
	hbase(main):005:0> describe 'tbl_1'
	Table tbl_1 is ENABLED                                                                                              
	tbl_1                                                                                                               
	COLUMN FAMILIES DESCRIPTION                                                                                         
	{NAME => 'cf1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KEEP_DELETED_CELLS => 'FALSE', DATA_BLO
	CK_ENCODING => 'NONE', TTL => '60 SECONDS (1 MINUTE)', COMPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'tr
	ue', BLOCKSIZE => '65536', REPLICATION_SCOPE => '0'}                                                                
	1 row(s) in 0.0260 seconds
	
	hbase(main):006:0> put 'tbl_1','rowkey1','cf1:col1','content of col1'
	0 row(s) in 0.3040 seconds
	
	hbase(main):007:0> scan 'tbl_1'
	ROW                            COLUMN+CELL                                                                          
	 rowkey1                       column=cf1:col1, timestamp=1533553279873, value=content of col1                      
	1 row(s) in 0.1040 seconds
	
	hbase(main):008:0> scan 'tbl_1'
	ROW                            COLUMN+CELL                                                                          
	 rowkey1                       column=cf1:col1, timestamp=1533553279873, value=content of col1                      
	1 row(s) in 0.0300 seconds
	
	hbase(main):009:0> scan 'tbl_1'
	ROW                            COLUMN+CELL                                                                          
	0 row(s) in 0.0230 seconds
	
	hbase(main):010:0> create 'tbl_2','cf2'
	0 row(s) in 2.2910 seconds
	
	=> Hbase::Table - tbl_2
	hbase(main):011:0> alter 'tbl_2',NAME=>'cf2',TTL=>10,MIN_VERSIONS=>1
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	0 row(s) in 2.7000 seconds
	
	hbase(main):012:0> list
	TABLE                                                                                                               
	mytable                                                                                                             
	mytable2                                                                                                            
	table2                                                                                                              
	tablesplit                                                                                                          
	tbl_1                                                                                                               
	tbl_2                                                                                                               
	6 row(s) in 0.0210 seconds
	
	=> ["mytable", "mytable2", "table2", "tablesplit", "tbl_1", "tbl_2"]
	hbase(main):013:0> put 'tbl_2','rowkey1','cf2:col1','content of col1'
	0 row(s) in 0.0260 seconds
	
	hbase(main):014:0> scan 'tbl_2'
	ROW                            COLUMN+CELL                                                                          
	 rowkey1                       column=cf2:col1, timestamp=1533553561765, value=content of col1                      
	1 row(s) in 0.0310 seconds
	
	hbase(main):015:0> scan 'tbl_2'
	ROW                            COLUMN+CELL                                                                          
	 rowkey1                       column=cf2:col1, timestamp=1533553561765, value=content of col1                      
	1 row(s) in 0.0320 seconds
	
	hbase(main):016:0> alter 'tbl_2',NAME=>'cf2',MIN_VERSIONS=>0
	Updating all regions with the new schema...
	0/1 regions updated.
	1/1 regions updated.
	Done.
	0 row(s) in 3.2150 seconds
	
	hbase(main):017:0> scan 'tbl_2'
	ROW                            COLUMN+CELL                                                                          
	0 row(s) in 0.0270 seconds