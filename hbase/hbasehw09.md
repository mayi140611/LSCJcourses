## hbasehw09
HBase集群安装

我安装的集群是4个节点h1,h2,h3,h4  
hdfs已经安装好，namenode url为hdfs://h1:9000  
zookeeper使用外部安装好的，而不是hbase自带的。  
hbase版本1.2.6
###1、配置hbase
* conf/hbase-env.sh   

		# The java implementation to use.  Java 1.7+ required
		export JAVA_HOME=/home/ian/installed/java
		# Tell HBase whether it should manage it's own instance of Zookeeper or not.
		export HBASE_MANAGES_ZK=false

* hbase-site.xml  

		<configuration>
		    <property>
		        <name>hbase.rootdir</name>
		        <value>hdfs://h1:9000/hbase</value>
		    </property>
		    <property>
		        <name>hbase.cluster.distributed</name>
		        <value>true</value>
		    </property>
		    <property>
		        <name>hbase.zookeeper.quorum</name>
		        <value>h1,h2,h3,h4</value>
		    </property>
		    <property>
		      <name>hbase.zookeeper.property.clientPort</name>
		      <value>2181</value>
		      <description>Property from ZooKeeper's config zoo.cfg.
		      The port at which the clients will connect.
		      </description>
		    </property>
		</configuration>  

* conf/regionserver  
	
		h1
		h2
		h3
		h4  

* conf/backup-masters  

		h2

### 2 启动hbase

	[ian@h1 profile.d]# start-hbase.sh
	starting master
	h1: starting regionserver,
	h2: starting regionserver,
	h2: starting master
	[ian@h1 Desktop]# jps
	94417 SecondaryNameNode
	93891 NameNode
	87490 QuorumPeerMain
	96021 HRegionServer
	95124 NodeManager
	94105 DataNode
	96397 Jps
	94767 ResourceManager
	95727 HMaster
	[ian@h2 hbase]$ jps
	42949 Jps
	42470 HRegionServer
	42088 NodeManager
	39529 QuorumPeerMain
	42666 HMaster
	41899 DataNode

### 3 验证
访问http://h1:16010

![image.png](https://upload-images.jianshu.io/upload_images/2850424-1c0e59fd22848667.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)