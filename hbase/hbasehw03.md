##实验一

	[ian@h1 ~]$ hbase shell
	SLF4J: Class path contains multiple SLF4J bindings.
	SLF4J: Found binding in [jar:file:/home/ian/installed/hbase/lib/slf4j-log4j12-1.7.5.jar!/org/slf4j/impl/StaticLoggerBinder.class]
	SLF4J: Found binding in [jar:file:/home/ian/installed/hadoop/share/hadoop/common/lib/slf4j-log4j12-1.7.10.jar!/org/slf4j/impl/StaticLoggerBinder.class]
	SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
	SLF4J: Actual binding is of type [org.slf4j.impl.Log4jLoggerFactory]
	HBase Shell; enter 'help<RETURN>' for list of supported commands.
	Type "exit<RETURN>" to leave the HBase Shell
	Version 1.2.6.1, rUnknown, Sun Jun  3 23:19:26 CDT 2018
	
	hbase(main):001:0> help
	HBase Shell, version 1.2.6.1, rUnknown, Sun Jun  3 23:19:26 CDT 2018
	Type 'help "COMMAND"', (e.g. 'help "get"' -- the quotes are necessary) for help on a specific command.
	Commands are grouped. Type 'help "COMMAND_GROUP"', (e.g. 'help "general"') for help on a command group.
	
	COMMAND GROUPS:
	  Group name: general
	  Commands: status, table_help, version, whoami
	
	  Group name: ddl
	  Commands: alter, alter_async, alter_status, create, describe, disable, disable_all, drop, drop_all, enable, enable_all, exists, get_table, is_disabled, is_enabled, list, locate_region, show_filters
	
	  Group name: namespace
	  Commands: alter_namespace, create_namespace, describe_namespace, drop_namespace, list_namespace, list_namespace_tables
	
	  Group name: dml
	  Commands: append, count, delete, deleteall, get, get_counter, get_splits, incr, put, scan, truncate, truncate_preserve
	
	  Group name: tools
	  Commands: assign, balance_switch, balancer, balancer_enabled, catalogjanitor_enabled, catalogjanitor_run, catalogjanitor_switch, close_region, compact, compact_rs, flush, major_compact, merge_region, move, normalize, normalizer_enabled, normalizer_switch, split, trace, unassign, wal_roll, zk_dump
	
	  Group name: replication
	  Commands: add_peer, append_peer_tableCFs, disable_peer, disable_table_replication, enable_peer, enable_table_replication, list_peers, list_replicated_tables, remove_peer, remove_peer_tableCFs, set_peer_tableCFs, show_peer_tableCFs
	
	  Group name: snapshots
	  Commands: clone_snapshot, delete_all_snapshot, delete_snapshot, list_snapshots, restore_snapshot, snapshot
	
	  Group name: configuration
	  Commands: update_all_config, update_config
	
	  Group name: quotas
	  Commands: list_quotas, set_quota
	
	  Group name: security
	  Commands: grant, list_security_capabilities, revoke, user_permission
	
	  Group name: procedures
	  Commands: abort_procedure, list_procedures
	
	  Group name: visibility labels
	  Commands: add_labels, clear_auths, get_auths, list_labels, set_auths, set_visibility
	
	SHELL USAGE:
	Quote all names in HBase Shell such as table and column names.  Commas delimit
	command parameters.  Type <RETURN> after entering a command to run it.
	Dictionaries of configuration used in the creation and alteration of tables are
	Ruby Hashes. They look like this:
	
	  {'key1' => 'value1', 'key2' => 'value2', ...}
	
	and are opened and closed with curley-braces.  Key/values are delimited by the
	'=>' character combination.  Usually keys are predefined constants such as
	NAME, VERSIONS, COMPRESSION, etc.  Constants do not need to be quoted.  Type
	'Object.constants' to see a (messy) list of all constants in the environment.
	
	If you are using binary keys or values and need to enter them in the shell, use
	double-quote'd hexadecimal representation. For example:
	
	  hbase> get 't1', "key\x03\x3f\xcd"
	  hbase> get 't1', "key\003\023\011"
	  hbase> put 't1', "test\xef\xff", 'f1:', "\x01\x33\x40"
	
	The HBase shell is the (J)Ruby IRB with the above HBase-specific commands added.
	For more on the HBase Shell, see http://hbase.apache.org/book.html
	hbase(main):002:0> list
	TABLE                                                                             
	table_2                                                                           
	1 row(s) in 0.1730 seconds
	
	=> ["table_2"]
	hbase(main):003:0> create 'table_1', 'colfam_1'
	0 row(s) in 2.5700 seconds
	
	=> Hbase::Table - table_1
	hbase(main):004:0> list
	TABLE                                                                             
	table_1                                                                           
	table_2                                                                           
	2 row(s) in 0.0140 seconds
	
	=> ["table_1", "table_2"]
	hbase(main):005:0> describe
	describe             describe_namespace
	hbase(main):005:0> describe 'table_1'
	Table table_1 is ENABLED                                                          
	table_1                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	1 row(s) in 0.1580 seconds
	
	hbase(main):006:0> drop 'table_1'
	
	ERROR: Table table_1 is enabled. Disable it first.
	
	Here is some help for this command:
	Drop the named table. Table must first be disabled:
	  hbase> drop 't1'
	  hbase> drop 'ns1:t1'
	
	
	hbase(main):007:0> disable 'table_1'
	0 row(s) in 2.3370 seconds
	
	hbase(main):008:0> dop 'table_1'
	NoMethodError: undefined method `dop' for #<Object:0x6b5ab2f2>
	
	hbase(main):009:0> drop 'table_1'
	0 row(s) in 2.3450 seconds
	
	hbase(main):010:0> list
	TABLE                                                                             
	table_2                                                                           
	1 row(s) in 0.0140 seconds
	
	=> ["table_2"]
	hbase(main):011:0> disable 'table_2'
	0 row(s) in 2.3040 seconds
	
	hbase(main):012:0> drop 'table_2'
	0 row(s) in 1.2940 seconds
	
	hbase(main):013:0> list
	TABLE                                                                             
	0 row(s) in 0.0140 seconds
	
	=> []
	hbase(main):014:0> create 'table_2','colfam_1'
	0 row(s) in 2.2950 seconds
	
	=> Hbase::Table - table_2
	hbase(main):015:0> list
	TABLE                                                                             
	table_2                                                                           
	1 row(s) in 0.0270 seconds
	
	=> ["table_2"]
	hbase(main):016:0> decribe 'table_2'
	NoMethodError: undefined method `decribe' for #<Object:0x6b5ab2f2>
	
	hbase(main):017:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	1 row(s) in 0.0230 seconds
	
	hbase(main):018:0> alter 'table_2', 'clofam_2'
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	0 row(s) in 2.2540 seconds
	
	hbase(main):019:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'clofam_2', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	2 row(s) in 0.0290 seconds
	
	hbase(main):020:0> alter 'table_2', 'colfam_2', {MATHOD='delete'}
	SyntaxError: (hbase):20: odd number list for Hash.alter 'table_2', 'colfam_2', {MATHOD='delete'}
	                                             ^
	
	hbase(main):021:0> alter 'table_2', 'colfam_2', MATHOD='delete'
	Updating all regions with the new schema...
	0/1 regions updated.
	1/1 regions updated.
	Done.
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	0 row(s) in 5.3790 seconds
	
	hbase(main):022:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'clofam_2', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'colfam_2', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'delete', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', KE
	EP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', COMP
	RESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '65536'
	, REPLICATION_SCOPE => '0'}                                                       
	4 row(s) in 0.0390 seconds
	
	hbase(main):023:0> alter 'table_2', 'colfam_2', MATHOD=>'delete'
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	Updating all regions with the new schema...
	0/1 regions updated.
	1/1 regions updated.
	Done.
	0 row(s) in 5.9460 seconds
	
	hbase(main):024:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'clofam_2', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'colfam_2', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	3 row(s) in 0.0390 seconds
	
	hbase(main):025:0> alter 'table_2', 'colfam_2', MATHOD=>'delete'
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	
	ERROR: org.apache.hadoop.hbase.InvalidFamilyOperationException: Family 'delete' does not exist, so it cannot be deleted
		at sun.reflect.NativeConstructorAccessorImpl.newInstance0(Native Method)
		at sun.reflect.NativeConstructorAccessorImpl.newInstance(NativeConstructorAccessorImpl.java:62)
		at sun.reflect.DelegatingConstructorAccessorImpl.newInstance(DelegatingConstructorAccessorImpl.java:45)
		at java.lang.reflect.Constructor.newInstance(Constructor.java:423)
		at org.apache.hadoop.ipc.RemoteException.instantiateException(RemoteException.java:106)
		at org.apache.hadoop.ipc.RemoteException.unwrapRemoteException(RemoteException.java:95)
		at org.apache.hadoop.hbase.util.ForeignExceptionUtil.toIOException(ForeignExceptionUtil.java:45)
		at org.apache.hadoop.hbase.procedure2.RemoteProcedureException.fromProto(RemoteProcedureException.java:114)
		at org.apache.hadoop.hbase.master.procedure.ProcedureSyncWait.waitForProcedureToComplete(ProcedureSyncWait.java:85)
		at org.apache.hadoop.hbase.master.HMaster$7.run(HMaster.java:2020)
		at org.apache.hadoop.hbase.master.procedure.MasterProcedureUtil.submitProcedure(MasterProcedureUtil.java:133)
		at org.apache.hadoop.hbase.master.HMaster.deleteColumn(HMaster.java:2006)
		at org.apache.hadoop.hbase.master.MasterRpcServices.deleteColumn(MasterRpcServices.java:475)
		at org.apache.hadoop.hbase.protobuf.generated.MasterProtos$MasterService$2.callBlockingMethod(MasterProtos.java:55658)
		at org.apache.hadoop.hbase.ipc.RpcServer.call(RpcServer.java:2196)
		at org.apache.hadoop.hbase.ipc.CallRunner.run(CallRunner.java:112)
		at org.apache.hadoop.hbase.ipc.RpcExecutor.consumerLoop(RpcExecutor.java:133)
		at org.apache.hadoop.hbase.ipc.RpcExecutor$1.run(RpcExecutor.java:108)
		at java.lang.Thread.run(Thread.java:748)
	Caused by: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.hbase.InvalidFamilyOperationException): Family 'delete' does not exist, so it cannot be deleted
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.prepareDelete(DeleteColumnFamilyProcedure.java:279)
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.executeFromState(DeleteColumnFamilyProcedure.java:91)
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.executeFromState(DeleteColumnFamilyProcedure.java:48)
		at org.apache.hadoop.hbase.procedure2.StateMachineProcedure.execute(StateMachineProcedure.java:119)
		at org.apache.hadoop.hbase.procedure2.Procedure.doExecute(Procedure.java:498)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execProcedure(ProcedureExecutor.java:1147)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execLoop(ProcedureExecutor.java:942)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execLoop(ProcedureExecutor.java:895)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.access$400(ProcedureExecutor.java:77)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor$2.run(ProcedureExecutor.java:497)
	
	Here is some help for this command:
	Alter a table. If the "hbase.online.schema.update.enable" property is set to
	false, then the table must be disabled (see help 'disable'). If the 
	"hbase.online.schema.update.enable" property is set to true, tables can be 
	altered without disabling them first. Altering enabled tables has caused problems 
	in the past, so use caution and test it before using in production. 
	
	You can use the alter command to add, 
	modify or delete column families or change table configuration options.
	Column families work in a similar way as the 'create' command. The column family
	specification can either be a name string, or a dictionary with the NAME attribute.
	Dictionaries are described in the output of the 'help' command, with no arguments.
	
	For example, to change or add the 'f1' column family in table 't1' from 
	current value to keep a maximum of 5 cell VERSIONS, do:
	
	  hbase> alter 't1', NAME => 'f1', VERSIONS => 5
	
	You can operate on several column families:
	
	  hbase> alter 't1', 'f1', {NAME => 'f2', IN_MEMORY => true}, {NAME => 'f3', VERSIONS => 5}
	
	To delete the 'f1' column family in table 'ns1:t1', use one of:
	
	  hbase> alter 'ns1:t1', NAME => 'f1', METHOD => 'delete'
	  hbase> alter 'ns1:t1', 'delete' => 'f1'
	
	You can also change table-scope attributes like MAX_FILESIZE, READONLY, 
	MEMSTORE_FLUSHSIZE, DURABILITY, etc. These can be put at the end;
	for example, to change the max size of a region to 128MB, do:
	
	  hbase> alter 't1', MAX_FILESIZE => '134217728'
	
	You can add a table coprocessor by setting a table coprocessor attribute:
	
	  hbase> alter 't1',
	    'coprocessor'=>'hdfs:///foo.jar|com.foo.FooRegionObserver|1001|arg1=1,arg2=2'
	
	Since you can have multiple coprocessors configured for a table, a
	sequence number will be automatically appended to the attribute name
	to uniquely identify it.
	
	The coprocessor attribute must match the pattern below in order for
	the framework to understand how to load the coprocessor classes:
	
	  [coprocessor jar file location] | class name | [priority] | [arguments]
	
	You can also set configuration settings specific to this table or column family:
	
	  hbase> alter 't1', CONFIGURATION => {'hbase.hregion.scan.loadColumnFamiliesOnDemand' => 'true'}
	  hbase> alter 't1', {NAME => 'f2', CONFIGURATION => {'hbase.hstore.blockingStoreFiles' => '10'}}
	
	You can also remove a table-scope attribute:
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'MAX_FILESIZE'
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'coprocessor$1'
	
	You can also set REGION_REPLICATION:
	
	  hbase> alter 't1', {REGION_REPLICATION => 2}
	
	There could be more than one alteration in one command:
	
	  hbase> alter 't1', { NAME => 'f1', VERSIONS => 3 }, 
	   { MAX_FILESIZE => '134217728' }, { METHOD => 'delete', NAME => 'f2' },
	   OWNER => 'johndoe', METADATA => { 'mykey' => 'myvalue' }
	
	
	hbase(main):026:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'clofam_2', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'colfam_2', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	3 row(s) in 0.0270 seconds
	
	hbase(main):027:0> alter 'table_2', NAME=>'colfam_2', MATHOD=>'delete'
	
	ERROR: org.apache.hadoop.hbase.InvalidFamilyOperationException: Family 'delete' does not exist, so it cannot be deleted
		at sun.reflect.NativeConstructorAccessorImpl.newInstance0(Native Method)
		at sun.reflect.NativeConstructorAccessorImpl.newInstance(NativeConstructorAccessorImpl.java:62)
		at sun.reflect.DelegatingConstructorAccessorImpl.newInstance(DelegatingConstructorAccessorImpl.java:45)
		at java.lang.reflect.Constructor.newInstance(Constructor.java:423)
		at org.apache.hadoop.ipc.RemoteException.instantiateException(RemoteException.java:106)
		at org.apache.hadoop.ipc.RemoteException.unwrapRemoteException(RemoteException.java:95)
		at org.apache.hadoop.hbase.util.ForeignExceptionUtil.toIOException(ForeignExceptionUtil.java:45)
		at org.apache.hadoop.hbase.procedure2.RemoteProcedureException.fromProto(RemoteProcedureException.java:114)
		at org.apache.hadoop.hbase.master.procedure.ProcedureSyncWait.waitForProcedureToComplete(ProcedureSyncWait.java:85)
		at org.apache.hadoop.hbase.master.HMaster$7.run(HMaster.java:2020)
		at org.apache.hadoop.hbase.master.procedure.MasterProcedureUtil.submitProcedure(MasterProcedureUtil.java:133)
		at org.apache.hadoop.hbase.master.HMaster.deleteColumn(HMaster.java:2006)
		at org.apache.hadoop.hbase.master.MasterRpcServices.deleteColumn(MasterRpcServices.java:475)
		at org.apache.hadoop.hbase.protobuf.generated.MasterProtos$MasterService$2.callBlockingMethod(MasterProtos.java:55658)
		at org.apache.hadoop.hbase.ipc.RpcServer.call(RpcServer.java:2196)
		at org.apache.hadoop.hbase.ipc.CallRunner.run(CallRunner.java:112)
		at org.apache.hadoop.hbase.ipc.RpcExecutor.consumerLoop(RpcExecutor.java:133)
		at org.apache.hadoop.hbase.ipc.RpcExecutor$1.run(RpcExecutor.java:108)
		at java.lang.Thread.run(Thread.java:748)
	Caused by: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.hbase.InvalidFamilyOperationException): Family 'delete' does not exist, so it cannot be deleted
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.prepareDelete(DeleteColumnFamilyProcedure.java:279)
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.executeFromState(DeleteColumnFamilyProcedure.java:91)
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.executeFromState(DeleteColumnFamilyProcedure.java:48)
		at org.apache.hadoop.hbase.procedure2.StateMachineProcedure.execute(StateMachineProcedure.java:119)
		at org.apache.hadoop.hbase.procedure2.Procedure.doExecute(Procedure.java:498)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execProcedure(ProcedureExecutor.java:1147)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execLoop(ProcedureExecutor.java:942)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execLoop(ProcedureExecutor.java:895)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.access$400(ProcedureExecutor.java:77)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor$2.run(ProcedureExecutor.java:497)
	
	Here is some help for this command:
	Alter a table. If the "hbase.online.schema.update.enable" property is set to
	false, then the table must be disabled (see help 'disable'). If the 
	"hbase.online.schema.update.enable" property is set to true, tables can be 
	altered without disabling them first. Altering enabled tables has caused problems 
	in the past, so use caution and test it before using in production. 
	
	You can use the alter command to add, 
	modify or delete column families or change table configuration options.
	Column families work in a similar way as the 'create' command. The column family
	specification can either be a name string, or a dictionary with the NAME attribute.
	Dictionaries are described in the output of the 'help' command, with no arguments.
	
	For example, to change or add the 'f1' column family in table 't1' from 
	current value to keep a maximum of 5 cell VERSIONS, do:
	
	  hbase> alter 't1', NAME => 'f1', VERSIONS => 5
	
	You can operate on several column families:
	
	  hbase> alter 't1', 'f1', {NAME => 'f2', IN_MEMORY => true}, {NAME => 'f3', VERSIONS => 5}
	
	To delete the 'f1' column family in table 'ns1:t1', use one of:
	
	  hbase> alter 'ns1:t1', NAME => 'f1', METHOD => 'delete'
	  hbase> alter 'ns1:t1', 'delete' => 'f1'
	
	You can also change table-scope attributes like MAX_FILESIZE, READONLY, 
	MEMSTORE_FLUSHSIZE, DURABILITY, etc. These can be put at the end;
	for example, to change the max size of a region to 128MB, do:
	
	  hbase> alter 't1', MAX_FILESIZE => '134217728'
	
	You can add a table coprocessor by setting a table coprocessor attribute:
	
	  hbase> alter 't1',
	    'coprocessor'=>'hdfs:///foo.jar|com.foo.FooRegionObserver|1001|arg1=1,arg2=2'
	
	Since you can have multiple coprocessors configured for a table, a
	sequence number will be automatically appended to the attribute name
	to uniquely identify it.
	
	The coprocessor attribute must match the pattern below in order for
	the framework to understand how to load the coprocessor classes:
	
	  [coprocessor jar file location] | class name | [priority] | [arguments]
	
	You can also set configuration settings specific to this table or column family:
	
	  hbase> alter 't1', CONFIGURATION => {'hbase.hregion.scan.loadColumnFamiliesOnDemand' => 'true'}
	  hbase> alter 't1', {NAME => 'f2', CONFIGURATION => {'hbase.hstore.blockingStoreFiles' => '10'}}
	
	You can also remove a table-scope attribute:
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'MAX_FILESIZE'
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'coprocessor$1'
	
	You can also set REGION_REPLICATION:
	
	  hbase> alter 't1', {REGION_REPLICATION => 2}
	
	There could be more than one alteration in one command:
	
	  hbase> alter 't1', { NAME => 'f1', VERSIONS => 3 }, 
	   { MAX_FILESIZE => '134217728' }, { METHOD => 'delete', NAME => 'f2' },
	   OWNER => 'johndoe', METADATA => { 'mykey' => 'myvalue' }
	
	
	hbase(main):028:0> alter 'table_2', {NAME=>'colfam_2', MATHOD=>'delete'}
	
	ERROR: org.apache.hadoop.hbase.InvalidFamilyOperationException: Family 'delete' does not exist, so it cannot be deleted
		at sun.reflect.NativeConstructorAccessorImpl.newInstance0(Native Method)
		at sun.reflect.NativeConstructorAccessorImpl.newInstance(NativeConstructorAccessorImpl.java:62)
		at sun.reflect.DelegatingConstructorAccessorImpl.newInstance(DelegatingConstructorAccessorImpl.java:45)
		at java.lang.reflect.Constructor.newInstance(Constructor.java:423)
		at org.apache.hadoop.ipc.RemoteException.instantiateException(RemoteException.java:106)
		at org.apache.hadoop.ipc.RemoteException.unwrapRemoteException(RemoteException.java:95)
		at org.apache.hadoop.hbase.util.ForeignExceptionUtil.toIOException(ForeignExceptionUtil.java:45)
		at org.apache.hadoop.hbase.procedure2.RemoteProcedureException.fromProto(RemoteProcedureException.java:114)
		at org.apache.hadoop.hbase.master.procedure.ProcedureSyncWait.waitForProcedureToComplete(ProcedureSyncWait.java:85)
		at org.apache.hadoop.hbase.master.HMaster$7.run(HMaster.java:2020)
		at org.apache.hadoop.hbase.master.procedure.MasterProcedureUtil.submitProcedure(MasterProcedureUtil.java:133)
		at org.apache.hadoop.hbase.master.HMaster.deleteColumn(HMaster.java:2006)
		at org.apache.hadoop.hbase.master.MasterRpcServices.deleteColumn(MasterRpcServices.java:475)
		at org.apache.hadoop.hbase.protobuf.generated.MasterProtos$MasterService$2.callBlockingMethod(MasterProtos.java:55658)
		at org.apache.hadoop.hbase.ipc.RpcServer.call(RpcServer.java:2196)
		at org.apache.hadoop.hbase.ipc.CallRunner.run(CallRunner.java:112)
		at org.apache.hadoop.hbase.ipc.RpcExecutor.consumerLoop(RpcExecutor.java:133)
		at org.apache.hadoop.hbase.ipc.RpcExecutor$1.run(RpcExecutor.java:108)
		at java.lang.Thread.run(Thread.java:748)
	Caused by: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.hbase.InvalidFamilyOperationException): Family 'delete' does not exist, so it cannot be deleted
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.prepareDelete(DeleteColumnFamilyProcedure.java:279)
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.executeFromState(DeleteColumnFamilyProcedure.java:91)
		at org.apache.hadoop.hbase.master.procedure.DeleteColumnFamilyProcedure.executeFromState(DeleteColumnFamilyProcedure.java:48)
		at org.apache.hadoop.hbase.procedure2.StateMachineProcedure.execute(StateMachineProcedure.java:119)
		at org.apache.hadoop.hbase.procedure2.Procedure.doExecute(Procedure.java:498)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execProcedure(ProcedureExecutor.java:1147)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execLoop(ProcedureExecutor.java:942)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.execLoop(ProcedureExecutor.java:895)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor.access$400(ProcedureExecutor.java:77)
		at org.apache.hadoop.hbase.procedure2.ProcedureExecutor$2.run(ProcedureExecutor.java:497)
	
	Here is some help for this command:
	Alter a table. If the "hbase.online.schema.update.enable" property is set to
	false, then the table must be disabled (see help 'disable'). If the 
	"hbase.online.schema.update.enable" property is set to true, tables can be 
	altered without disabling them first. Altering enabled tables has caused problems 
	in the past, so use caution and test it before using in production. 
	
	You can use the alter command to add, 
	modify or delete column families or change table configuration options.
	Column families work in a similar way as the 'create' command. The column family
	specification can either be a name string, or a dictionary with the NAME attribute.
	Dictionaries are described in the output of the 'help' command, with no arguments.
	
	For example, to change or add the 'f1' column family in table 't1' from 
	current value to keep a maximum of 5 cell VERSIONS, do:
	
	  hbase> alter 't1', NAME => 'f1', VERSIONS => 5
	
	You can operate on several column families:
	
	  hbase> alter 't1', 'f1', {NAME => 'f2', IN_MEMORY => true}, {NAME => 'f3', VERSIONS => 5}
	
	To delete the 'f1' column family in table 'ns1:t1', use one of:
	
	  hbase> alter 'ns1:t1', NAME => 'f1', METHOD => 'delete'
	  hbase> alter 'ns1:t1', 'delete' => 'f1'
	
	You can also change table-scope attributes like MAX_FILESIZE, READONLY, 
	MEMSTORE_FLUSHSIZE, DURABILITY, etc. These can be put at the end;
	for example, to change the max size of a region to 128MB, do:
	
	  hbase> alter 't1', MAX_FILESIZE => '134217728'
	
	You can add a table coprocessor by setting a table coprocessor attribute:
	
	  hbase> alter 't1',
	    'coprocessor'=>'hdfs:///foo.jar|com.foo.FooRegionObserver|1001|arg1=1,arg2=2'
	
	Since you can have multiple coprocessors configured for a table, a
	sequence number will be automatically appended to the attribute name
	to uniquely identify it.
	
	The coprocessor attribute must match the pattern below in order for
	the framework to understand how to load the coprocessor classes:
	
	  [coprocessor jar file location] | class name | [priority] | [arguments]
	
	You can also set configuration settings specific to this table or column family:
	
	  hbase> alter 't1', CONFIGURATION => {'hbase.hregion.scan.loadColumnFamiliesOnDemand' => 'true'}
	  hbase> alter 't1', {NAME => 'f2', CONFIGURATION => {'hbase.hstore.blockingStoreFiles' => '10'}}
	
	You can also remove a table-scope attribute:
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'MAX_FILESIZE'
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'coprocessor$1'
	
	You can also set REGION_REPLICATION:
	
	  hbase> alter 't1', {REGION_REPLICATION => 2}
	
	There could be more than one alteration in one command:
	
	  hbase> alter 't1', { NAME => 'f1', VERSIONS => 3 }, 
	   { MAX_FILESIZE => '134217728' }, { METHOD => 'delete', NAME => 'f2' },
	   OWNER => 'johndoe', METADATA => { 'mykey' => 'myvalue' }
	
	
	hbase(main):029:0> alter 'table_2', {NAME=>'colfam_2', METHOD=>'delete'}
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	0 row(s) in 2.6940 seconds
	
	hbase(main):030:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'clofam_2', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	2 row(s) in 0.0220 seconds
	
	hbase(main):031:0> alter 'table_2', 'clofam_2', METHOD=>'delete'
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	
	ERROR: NAME parameter missing for delete method
	
	Here is some help for this command:
	Alter a table. If the "hbase.online.schema.update.enable" property is set to
	false, then the table must be disabled (see help 'disable'). If the 
	"hbase.online.schema.update.enable" property is set to true, tables can be 
	altered without disabling them first. Altering enabled tables has caused problems 
	in the past, so use caution and test it before using in production. 
	
	You can use the alter command to add, 
	modify or delete column families or change table configuration options.
	Column families work in a similar way as the 'create' command. The column family
	specification can either be a name string, or a dictionary with the NAME attribute.
	Dictionaries are described in the output of the 'help' command, with no arguments.
	
	For example, to change or add the 'f1' column family in table 't1' from 
	current value to keep a maximum of 5 cell VERSIONS, do:
	
	  hbase> alter 't1', NAME => 'f1', VERSIONS => 5
	
	You can operate on several column families:
	
	  hbase> alter 't1', 'f1', {NAME => 'f2', IN_MEMORY => true}, {NAME => 'f3', VERSIONS => 5}
	
	To delete the 'f1' column family in table 'ns1:t1', use one of:
	
	  hbase> alter 'ns1:t1', NAME => 'f1', METHOD => 'delete'
	  hbase> alter 'ns1:t1', 'delete' => 'f1'
	
	You can also change table-scope attributes like MAX_FILESIZE, READONLY, 
	MEMSTORE_FLUSHSIZE, DURABILITY, etc. These can be put at the end;
	for example, to change the max size of a region to 128MB, do:
	
	  hbase> alter 't1', MAX_FILESIZE => '134217728'
	
	You can add a table coprocessor by setting a table coprocessor attribute:
	
	  hbase> alter 't1',
	    'coprocessor'=>'hdfs:///foo.jar|com.foo.FooRegionObserver|1001|arg1=1,arg2=2'
	
	Since you can have multiple coprocessors configured for a table, a
	sequence number will be automatically appended to the attribute name
	to uniquely identify it.
	
	The coprocessor attribute must match the pattern below in order for
	the framework to understand how to load the coprocessor classes:
	
	  [coprocessor jar file location] | class name | [priority] | [arguments]
	
	You can also set configuration settings specific to this table or column family:
	
	  hbase> alter 't1', CONFIGURATION => {'hbase.hregion.scan.loadColumnFamiliesOnDemand' => 'true'}
	  hbase> alter 't1', {NAME => 'f2', CONFIGURATION => {'hbase.hstore.blockingStoreFiles' => '10'}}
	
	You can also remove a table-scope attribute:
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'MAX_FILESIZE'
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'coprocessor$1'
	
	You can also set REGION_REPLICATION:
	
	  hbase> alter 't1', {REGION_REPLICATION => 2}
	
	There could be more than one alteration in one command:
	
	  hbase> alter 't1', { NAME => 'f1', VERSIONS => 3 }, 
	   { MAX_FILESIZE => '134217728' }, { METHOD => 'delete', NAME => 'f2' },
	   OWNER => 'johndoe', METADATA => { 'mykey' => 'myvalue' }
	
	
	hbase(main):032:0> alter 'table_2', NAME=>'clofam_2', METHOD=>'delete'
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	0 row(s) in 2.4410 seconds
	
	hbase(main):033:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	1 row(s) in 0.0440 seconds
	
	hbase(main):034:0> alter 'table_2', NAME=>'colfam_1', VERSIONS=>4
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	0 row(s) in 2.6970 seconds
	
	hbase(main):035:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '4', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	1 row(s) in 0.0210 seconds
	
	hbase(main):036:0> alter 'table_2', NAME=>'colfam_1', VERSIONS=>1
	Updating all regions with the new schema...
	1/1 regions updated.
	Done.
	0 row(s) in 2.1980 seconds
	
	hbase(main):037:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	1 row(s) in 0.0220 seconds

![1.PNG](1.PNG)

##实验二

	[ian@h1 ~]$ hbase shell
	SLF4J: Class path contains multiple SLF4J bindings.
	SLF4J: Found binding in [jar:file:/home/ian/installed/hbase/lib/slf4j-log4j12-1.7.5.jar!/org/slf4j/impl/StaticLoggerBinder.class]
	SLF4J: Found binding in [jar:file:/home/ian/installed/hadoop/share/hadoop/common/lib/slf4j-log4j12-1.7.10.jar!/org/slf4j/impl/StaticLoggerBinder.class]
	SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
	SLF4J: Actual binding is of type [org.slf4j.impl.Log4jLoggerFactory]
	HBase Shell; enter 'help<RETURN>' for list of supported commands.
	Type "exit<RETURN>" to leave the HBase Shell
	Version 1.2.6.1, rUnknown, Sun Jun  3 23:19:26 CDT 2018
	
	hbase(main):001:0> describe 'table_2'
	Table table_2 is ENABLED                                                          
	table_2                                                                           
	COLUMN FAMILIES DESCRIPTION                                                       
	{NAME => 'colfam_1', BLOOMFILTER => 'ROW', VERSIONS => '1', IN_MEMORY => 'false', 
	KEEP_DELETED_CELLS => 'FALSE', DATA_BLOCK_ENCODING => 'NONE', TTL => 'FOREVER', CO
	MPRESSION => 'NONE', MIN_VERSIONS => '0', BLOCKCACHE => 'true', BLOCKSIZE => '6553
	6', REPLICATION_SCOPE => '0'}                                                     
	1 row(s) in 0.2650 seconds
	
	hbase(main):002:0> get 'table_2','100'
	COLUMN                CELL                                                        
	0 row(s) in 0.0970 seconds
	
	hbase(main):003:0> put 'table_2','100','colfam_1:age','1'
	0 row(s) in 0.1310 seconds
	
	hbase(main):004:0> get 'table_2','100'
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531629601132, value=1                            
	1 row(s) in 0.0280 seconds
	
	hbase(main):005:0> put 'table_2','100','colfam_1:age','2'
	0 row(s) in 0.0120 seconds
	
	hbase(main):006:0> put 'table_2','100','colfam_1:age','3'
	0 row(s) in 0.0080 seconds
	
	hbase(main):007:0> get 'table_2','100'
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531629644158, value=3                            
	1 row(s) in 0.0160 seconds
	
	hbase(main):008:0> scan 'table_2','100'
	ROW                   COLUMN+CELL                                                 
	
	ERROR: Args should be a Hash
	
	Here is some help for this command:
	Scan a table; pass table name and optionally a dictionary of scanner
	specifications.  Scanner specifications may include one or more of:
	TIMERANGE, FILTER, LIMIT, STARTROW, STOPROW, ROWPREFIXFILTER, TIMESTAMP,
	MAXLENGTH or COLUMNS, CACHE or RAW, VERSIONS, ALL_METRICS or METRICS
	
	If no columns are specified, all columns will be scanned.
	To scan all members of a column family, leave the qualifier empty as in
	'col_family'.
	
	The filter can be specified in two ways:
	1. Using a filterString - more information on this is available in the
	Filter Language document attached to the HBASE-4176 JIRA
	2. Using the entire package name of the filter.
	
	If you wish to see metrics regarding the execution of the scan, the
	ALL_METRICS boolean should be set to true. Alternatively, if you would
	prefer to see only a subset of the metrics, the METRICS array can be 
	defined to include the names of only the metrics you care about.
	
	Some examples:
	
	  hbase> scan 'hbase:meta'
	  hbase> scan 'hbase:meta', {COLUMNS => 'info:regioninfo'}
	  hbase> scan 'ns1:t1', {COLUMNS => ['c1', 'c2'], LIMIT => 10, STARTROW => 'xyz'}
	  hbase> scan 't1', {COLUMNS => ['c1', 'c2'], LIMIT => 10, STARTROW => 'xyz'}
	  hbase> scan 't1', {COLUMNS => 'c1', TIMERANGE => [1303668804, 1303668904]}
	  hbase> scan 't1', {REVERSED => true}
	  hbase> scan 't1', {ALL_METRICS => true}
	  hbase> scan 't1', {METRICS => ['RPC_RETRIES', 'ROWS_FILTERED']}
	  hbase> scan 't1', {ROWPREFIXFILTER => 'row2', FILTER => "
	    (QualifierFilter (>=, 'binary:xyz')) AND (TimestampsFilter ( 123, 456))"}
	  hbase> scan 't1', {FILTER =>
	    org.apache.hadoop.hbase.filter.ColumnPaginationFilter.new(1, 0)}
	  hbase> scan 't1', {CONSISTENCY => 'TIMELINE'}
	For setting the Operation Attributes 
	  hbase> scan 't1', { COLUMNS => ['c1', 'c2'], ATTRIBUTES => {'mykey' => 'myvalue'}}
	  hbase> scan 't1', { COLUMNS => ['c1', 'c2'], AUTHORIZATIONS => ['PRIVATE','SECRET']}
	For experts, there is an additional option -- CACHE_BLOCKS -- which
	switches block caching for the scanner on (true) or off (false).  By
	default it is enabled.  Examples:
	
	  hbase> scan 't1', {COLUMNS => ['c1', 'c2'], CACHE_BLOCKS => false}
	
	Also for experts, there is an advanced option -- RAW -- which instructs the
	scanner to return all cells (including delete markers and uncollected deleted
	cells). This option cannot be combined with requesting specific COLUMNS.
	Disabled by default.  Example:
	
	  hbase> scan 't1', {RAW => true, VERSIONS => 10}
	
	Besides the default 'toStringBinary' format, 'scan' supports custom formatting
	by column.  A user can define a FORMATTER by adding it to the column name in
	the scan specification.  The FORMATTER can be stipulated: 
	
	 1. either as a org.apache.hadoop.hbase.util.Bytes method name (e.g, toInt, toString)
	 2. or as a custom class followed by method name: e.g. 'c(MyFormatterClass).format'.
	
	Example formatting cf:qualifier1 and cf:qualifier2 both as Integers: 
	  hbase> scan 't1', {COLUMNS => ['cf:qualifier1:toInt',
	    'cf:qualifier2:c(org.apache.hadoop.hbase.util.Bytes).toInt'] } 
	
	Note that you can specify a FORMATTER by column only (cf:qualifier).  You cannot
	specify a FORMATTER for all columns of a column family.
	
	Scan can also be used directly from a table, by first getting a reference to a
	table, like such:
	
	  hbase> t = get_table 't'
	  hbase> t.scan
	
	Note in the above situation, you can still provide all the filtering, columns,
	options, etc as described above.
	
	
	
	hbase(main):009:0> scan 'table_2',{COLUMNS=>'colfam_1:age'}
	ROW                   COLUMN+CELL                                                 
	 100                  column=colfam_1:age, timestamp=1531629644158, value=3       
	1 row(s) in 0.0600 seconds
	
	hbase(main):010:0> alter 'table_2',NAME=>'colfam_1:age',VERSIONS=>3
	
	ERROR: Illegal character <58>. Family names cannot contain control characters or colons: colfam_1:age
	
	Here is some help for this command:
	Alter a table. If the "hbase.online.schema.update.enable" property is set to
	false, then the table must be disabled (see help 'disable'). If the 
	"hbase.online.schema.update.enable" property is set to true, tables can be 
	altered without disabling them first. Altering enabled tables has caused problems 
	in the past, so use caution and test it before using in production. 
	
	You can use the alter command to add, 
	modify or delete column families or change table configuration options.
	Column families work in a similar way as the 'create' command. The column family
	specification can either be a name string, or a dictionary with the NAME attribute.
	Dictionaries are described in the output of the 'help' command, with no arguments.
	
	For example, to change or add the 'f1' column family in table 't1' from 
	current value to keep a maximum of 5 cell VERSIONS, do:
	
	  hbase> alter 't1', NAME => 'f1', VERSIONS => 5
	
	You can operate on several column families:
	
	  hbase> alter 't1', 'f1', {NAME => 'f2', IN_MEMORY => true}, {NAME => 'f3', VERSIONS => 5}
	
	To delete the 'f1' column family in table 'ns1:t1', use one of:
	
	  hbase> alter 'ns1:t1', NAME => 'f1', METHOD => 'delete'
	  hbase> alter 'ns1:t1', 'delete' => 'f1'
	
	You can also change table-scope attributes like MAX_FILESIZE, READONLY, 
	MEMSTORE_FLUSHSIZE, DURABILITY, etc. These can be put at the end;
	for example, to change the max size of a region to 128MB, do:
	
	  hbase> alter 't1', MAX_FILESIZE => '134217728'
	
	You can add a table coprocessor by setting a table coprocessor attribute:
	
	  hbase> alter 't1',
	    'coprocessor'=>'hdfs:///foo.jar|com.foo.FooRegionObserver|1001|arg1=1,arg2=2'
	
	Since you can have multiple coprocessors configured for a table, a
	sequence number will be automatically appended to the attribute name
	to uniquely identify it.
	
	The coprocessor attribute must match the pattern below in order for
	the framework to understand how to load the coprocessor classes:
	
	  [coprocessor jar file location] | class name | [priority] | [arguments]
	
	You can also set configuration settings specific to this table or column family:
	
	  hbase> alter 't1', CONFIGURATION => {'hbase.hregion.scan.loadColumnFamiliesOnDemand' => 'true'}
	  hbase> alter 't1', {NAME => 'f2', CONFIGURATION => {'hbase.hstore.blockingStoreFiles' => '10'}}
	
	You can also remove a table-scope attribute:
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'MAX_FILESIZE'
	
	  hbase> alter 't1', METHOD => 'table_att_unset', NAME => 'coprocessor$1'
	
	You can also set REGION_REPLICATION:
	
	  hbase> alter 't1', {REGION_REPLICATION => 2}
	
	There could be more than one alteration in one command:
	
	  hbase> alter 't1', { NAME => 'f1', VERSIONS => 3 }, 
	   { MAX_FILESIZE => '134217728' }, { METHOD => 'delete', NAME => 'f2' },
	   OWNER => 'johndoe', METADATA => { 'mykey' => 'myvalue' }
	
	
	hbase(main):011:0> alter 'table_2',NAME=>'colfam_1',VERSIONS=>3
	Updating all regions with the new schema...
	0/1 regions updated.
	1/1 regions updated.
	Done.
	0 row(s) in 3.7360 seconds
	
	hbase(main):012:0> get 'table_2','100'
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531629644158, value=3                            
	1 row(s) in 0.0440 seconds
	
	hbase(main):013:0> get 'table_2','100', {COLUMNS=>'colfam_1:age',VERSIONS=>2}
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531629644158, value=3                            
	1 row(s) in 0.0340 seconds
	
	hbase(main):014:0> get 'table_2','100', {COLUMNS=>'colfam_1:age',VERSIONS=>3}
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531629644158, value=3                            
	1 row(s) in 0.0070 seconds
	
	hbase(main):015:0> put 'table_2','100','colfam_1:age','4'
	0 row(s) in 0.0120 seconds
	
	hbase(main):016:0> get 'table_2','100', {COLUMNS=>'colfam_1:age',VERSIONS=>3}
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531630140244, value=4                            
	 colfam_1:age         timestamp=1531629644158, value=3                            
	2 row(s) in 0.0180 seconds
	
	hbase(main):017:0> get 'table_2','100', {COLUMNS=>'colfam_1:age',VERSIONS=>2}
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531630140244, value=4                            
	 colfam_1:age         timestamp=1531629644158, value=3                            
	2 row(s) in 0.0160 seconds
	
	hbase(main):018:0> get 'table_2','100', {COLUMNS=>'colfam_1:age',VERSIONS=>1}
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531630140244, value=4                            
	1 row(s) in 0.0130 seconds
	
	hbase(main):019:0> get 'table_2','100', {COLOMNS=>'colfam_1:age'}
	NameError: uninitialized constant COLOMNS
	
	hbase(main):020:0> get 'table_2','100', {COLOMN=>'colfam_1:age'}
	NameError: uninitialized constant COLOMN
	
	hbase(main):021:0> get 'table_2','100', {COLUMNS=>'colfam_1:age'}
	COLUMN                CELL                                                        
	 colfam_1:age         timestamp=1531630140244, value=4                            
	1 row(s) in 0.0170 seconds
	
	hbase(main):022:0> delete 'table_2','100',{COLUMNS:'colfam_1:age'}
	SyntaxError: (hbase):22: syntax error, unexpected tSYMBEG
	
	delete 'table_2','100',{COLUMNS:'colfam_1:age'}
	                                ^
	
	hbase(main):023:0> delete 'table_2','100',{COLUMNS=>'colfam_1:age'}
	
	ERROR: undefined method `to_java_bytes' for {"COLUMNS"=>"colfam_1:age"}:Hash
	
	Here is some help for this command:
	Put a delete cell value at specified table/row/column and optionally
	timestamp coordinates.  Deletes must match the deleted cell's
	coordinates exactly.  When scanning, a delete cell suppresses older
	versions. To delete a cell from  't1' at row 'r1' under column 'c1'
	marked with the time 'ts1', do:
	
	  hbase> delete 'ns1:t1', 'r1', 'c1', ts1
	  hbase> delete 't1', 'r1', 'c1', ts1
	  hbase> delete 't1', 'r1', 'c1', ts1, {VISIBILITY=>'PRIVATE|SECRET'}
	
	The same command can also be run on a table reference. Suppose you had a reference
	t to table 't1', the corresponding command would be:
	
	  hbase> t.delete 'r1', 'c1',  ts1
	  hbase> t.delete 'r1', 'c1',  ts1, {VISIBILITY=>'PRIVATE|SECRET'}
	
	
	hbase(main):024:0> delete 'table_2','100','colfam_1:age'
	0 row(s) in 0.0430 seconds
	
	hbase(main):025:0> scan 'table_2'
	ROW                   COLUMN+CELL                                                 
	0 row(s) in 0.0270 seconds
	
	hbase(main):026:0> get 'table_2','100'
	COLUMN                CELL                                                        
	0 row(s) in 0.0160 seconds
	
	hbase(main):027:0> delete 'table_2','100'
	
	ERROR: wrong number of arguments (2 for 3)
	
	Here is some help for this command:
	Put a delete cell value at specified table/row/column and optionally
	timestamp coordinates.  Deletes must match the deleted cell's
	coordinates exactly.  When scanning, a delete cell suppresses older
	versions. To delete a cell from  't1' at row 'r1' under column 'c1'
	marked with the time 'ts1', do:
	
	  hbase> delete 'ns1:t1', 'r1', 'c1', ts1
	  hbase> delete 't1', 'r1', 'c1', ts1
	  hbase> delete 't1', 'r1', 'c1', ts1, {VISIBILITY=>'PRIVATE|SECRET'}
	
	The same command can also be run on a table reference. Suppose you had a reference
	t to table 't1', the corresponding command would be:
	
	  hbase> t.delete 'r1', 'c1',  ts1
	  hbase> t.delete 'r1', 'c1',  ts1, {VISIBILITY=>'PRIVATE|SECRET'}
	
	
	hbase(main):028:0> deleteall 'table_2','100'
	0 row(s) in 0.0070 seconds
	
	hbase(main):029:0> get 'table_2','100'
	COLUMN                CELL                                                        
	0 row(s) in 0.0140 seconds