## webhw05
### 通过Ajax技术实现类似搜索下拉的效果，要求如下：
1、文本框中输入文字后，通过Ajax与服务端交互，列出待选的结果  
2、选择某个待选结果后，拉下框消失，文本框中显示选择的结果  
3、服务端使用 web.py 实现  
4、服务端返回模拟的待选结果即可

在当前目录输入：`python .\server.py`

在浏览器输入：http://localhost:8080/showfile/webhw05.html


![效果图](./1.png)

### 问题
>No 'Access-Control-Allow-Origin' header is present on the requested resource.'

#### 什么是跨域访问
在A网站中，我们希望使用Ajax来获得B网站中的特定内容。如果A网站与B网站不在同一个域中，那么就出现了跨域访问问题。你可以理解为两个域名之间不能跨过域名来发送请求或者请求数据，否则就是不安全的。跨域访问违反了同源策略，
总而言之，同源策略规定，浏览器的ajax只能访问跟它的HTML页面同源（相同域名或IP）的资源。

在服务端设置
`web.header('Access-Control-Allow-Origin','*')`
