# 09 爬虫

## 09.01 Urllib 

### 01 Urllib 基本使用

使用urllib来获取百度首页的源码：

```python
	import urllib.request

# (1)定义一个url  就是你要访问的地址
url = 'http://www.baidu.com'

# (2)模拟浏览器向服务器发送请求 response响应
response = urllib.request.urlopen(url)

# （3）获取响应中的页面的源码  content 内容的意思
# read方法  返回的是字节形式的二进制数据
# 我们要将二进制的数据转换为字符串
# 二进制--》字符串  解码  decode('编码的格式')
content = response.read().decode('utf-8')

# （4）打印数据
print(content)
```

注意点：

- `rllib.request.urlopen() `模拟浏览器向服务器发送请求

- `rllib.request.urlopen(url) `返回的数据类型是HttpResponse。

- - 可以print输出，查看类型，如果是b开头的就是字节码数据，需要进行decode
  - 查看网页源码中的head部分的charset用的是什么编码格式，然后用相应的解码，例如utf-8



### 02 一个类型和六个方法

```python
import urllib.request

url = 'http://www.baidu.com'

# 模拟浏览器向服务器发送请求
response = urllib.request.urlopen(url)

# 一个类型和六个方法
# response是HTTPResponse的类型
print(type(response))
# 输出为：<class 'http.client.HTTPResponse'>

# 按照一个字节一个字节的去读
content = response.read()
print(content)

# 返回多少个字节
content = response.read(5)
print(content)

# 读取一行
content = response.readline()
print(content)
#一行一行读，直到读完
content = response.readlines()
print(content)

# 返回状态码  如果是200了 那么就证明我们的逻辑没有错
print(response.getcode())

# 返回的是url地址
print(response.geturl())

# 获取是一个状态信息（请求头）
print(response.getheaders())

```

- **一个类型 HTTPResponse**

- **六个方法 read  readline  readlines  getcode geturl getheaders**



### 03 下载

​	`urllib.request.urlretrieve(url,fileName)` url代表的是下载的路径  filename文件的名字

将爬取到的东西下载到本地：

```python
import urllib.request

# 下载网页
url_page = 'http://www.baidu.com'

# url代表的是下载的路径  filename文件的名字
# 在python中 可以变量的名字  也可以直接写值
urllib.request.urlretrieve(url_page,'baidu.html')

# 下载图片
url_img = 'https://img1.baidu.com/it/u=3004965690,4089234593&fm=26&fmt=auto&gp=0.jpg'

urllib.request.urlretrieve(url= url_img,filename='lisa.jpg')

# 下载视频
url_video = 'https://vd3.bdstatic.com/mda-mhkku4ndaka5etk3/1080p/cae_h264/1629557146541497769/mda-mhkku4ndaka5etk3.mp4?v_from_s=hkapp-haokan-tucheng&auth_key=1629687514-0-0-7ed57ed7d1168bb1f06d18a4ea214300&bcevod_channel=searchbox_feed&pd=1&pt=3&abtest='

urllib.request.urlretrieve(url_video,'hxekyyds.mp4')
```



### 04 请求对象的定制

需要向网页上请求数据，就需要将请求伪装成浏览器的样子，第一法反扒方法就是请求对象定制。一般网站都需要获取用户代理信息，以此表明这是个浏览器来请求数据。

> UA介绍：**User Agent**中文名为用户代理，简称 UA，它是一个特殊字符串头，使得服务器能够识别客户使用的操作系统及版本、CPU 类型、浏览器及版本。浏览器内核、浏览器渲染引擎、浏览器语言、浏览器插件等

urlopen方法中不能存储字典，headers不能传递进去。所以需要给他专门定制一个request，将这个request作为参数传到urlopen中。

语法：`equest = urllib.request.Request()`

```python
import urllib.request

url = 'https://www.baidu.com'

# url的组成
# https://www.baidu.com/s?wd=周杰伦

# http/https    www.baidu.com   80/443     s      wd = 周杰伦     #
#    协议             主机        端口号     路径     参数           锚点
# http   80
# https  443
# mysql  3306
# oracle 1521
# redis  6379
# mongodb 27017

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

# 因为urlopen方法中不能存储字典 所以headers不能传递进去
# 请求对象的定制
request = urllib.request.Request(url=url,headers=headers)
response = urllib.request.urlopen(request)
content = response.read().decode('utf8')
print(content)
```



### 05 编码解码

#### 1.get请求方式：urllib.parse.quote（）

我们需要将传进去的参数进行编码，即用`urllib.parse.quote（）`，不转成Unicode就会报错。然后拼接url

```python
# https://www.baidu.com/s?wd=%E5%91%A8%E6%9D%B0%E4%BC%A6  #这是unicode的编码

# 需求 获取 https://www.baidu.com/s?wd=周杰伦的网页源码

import urllib.request
import urllib.parse

url = 'https://www.baidu.com/s?wd='

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

# 将周杰伦三个字变成unicode编码的格式
# 我们需要依赖于urllib.parse
name = urllib.parse.quote('周杰伦')

url = url + name

# 请求对象的定制
request = urllib.request.Request(url=url,headers=headers)
# 模拟浏览器向服务器发送请求
response = urllib.request.urlopen(request)
# 获取响应的内容
content = response.read().decode('utf-8')
# 打印数据
print(content)
```

#### 2.get请求方式：urllib.parse.urlencode（）

urlencode应用场景：**多个参数**的时候。

当url有多个参数的时候，直接将参数存成一个字典，然后将字典作为urllib.parse.urlencode()的参数传进去。这个方法会将字典的value转换成unicode。然后将编码的参数与url拼接。

```python
# https://www.baidu.com/s?wd=周杰伦&sex=男

#获取https://www.baidu.com/s?wd=%E5%91%A8%E6%9D%B0%E4%BC%A6&sex=%E7%94%B7的网页源码

import urllib.request
import urllib.parse

base_url = 'https://www.baidu.com/s?'

data = {
    'wd':'周杰伦',
    'sex':'男',
    'location':'中国台湾省'
}

new_data = urllib.parse.urlencode(data)

# 请求资源路径
url = base_url + new_data

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}
request = urllib.request.Request(url=url,headers=headers)
response = urllib.request.urlopen(request)
content = response.read().decode('utf-8')
print(content)
```

#### 3.post请求方式

有些场景需要用到post请求，例如百度翻译。注意post中参数放置的位置——**定制一个request**

```python
import urllib.request
import urllib.parse

# 1.这个url需要在网页的开发者工具中network中仔细查找，找到数据请求的url，这一步比较难
url = 'https://fanyi.baidu.com/sug'		

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}
data = {
    'kw':'spider'
}

# 2.post请求的参数 必须要进行编码，并调用encode方法
data = urllib.parse.urlencode(data).encode('utf-8')

# 3.post的请求的参数 是不会拼接在url的后面的  而是需要放在请求对象定制的参数中
request = urllib.request.Request(url=url,data=data,headers=headers)

# 4.模拟浏览器向服务器发送请求
response = urllib.request.urlopen(request)
content = response.read().decode('utf-8')

# 5.此时获取到的数据是这样的：{"errno":0,"data":[{"k":"spider","v":"n. \u8718\u86db; \u661f\u5f62\u8f6e\uff0c\u5341\u5b57\u53c9;...  可以用json模块将字符串转成json对象便于读取
# 字符串--》json对象
import json
obj = json.loads(content)
print(obj)
```

post特点：

- post请求方式的参数 必须编码   data = urllib.parse.urlencode(data)

- 编码之后 必须调用encode方法 data = urllib.parse.urlencode(data).encode('utf-8')

- 参数是放在请求对象定制的方法中  request = urllib.request.Request(url=url,data=data,headers=headers)

#### 总结：post和get区别

1：get请求方式的参数必须编码，**参数是拼接到url后面，编码之后不需要调用encode方法**

2：post请求方式的参数必须编码，**参数是放在请求对象定制的方法中，编码之后需要调用encode方法**



#### 案例练习：百度详细翻译

熟练post；了解第二个反扒手段

也就是header部分，但是header中的不是所有都有用，可以一个一个测试，这里起作用的就是cookie

```python
import urllib.request
import urllib.parse

url = 'https://fanyi.baidu.com/v2transapi?from=en&to=zh'

headers = {
    # 'Accept': '*/*',
    # 'Accept-Encoding': 'gzip, deflate, br',
    # 'Accept-Language': 'zh-CN,zh;q=0.9',
    # 'Connection': 'keep-alive',
    # 'Content-Length': '135',
    # 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Cookie': 'BIDUPSID=DAA8F9F0BD801A2929D96D69CF7EBF50; PSTM=1597202227; BAIDUID=DAA8F9F0BD801A29B2813502000BF8E9:SL=0:NR=10:FG=1; __yjs_duid=1_c19765bd685fa6fa12c2853fc392f8db1618999058029; REALTIME_TRANS_SWITCH=1; FANYI_WORD_SWITCH=1; HISTORY_SWITCH=1; SOUND_SPD_SWITCH=1; SOUND_PREFER_SWITCH=1; BDUSS=R2bEZvTjFCNHQxdUV-cTZ-MzZrSGxhbUYwSkRkUWk2SkxxS3E2M2lqaFRLUlJoRVFBQUFBJCQAAAAAAAAAAAEAAAA3e~BTveK-9sHLZGF5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFOc7GBTnOxgaW; BDUSS_BFESS=R2bEZvTjFCNHQxdUV-cTZ-MzZrSGxhbUYwSkRkUWk2SkxxS3E2M2lqaFRLUlJoRVFBQUFBJCQAAAAAAAAAAAEAAAA3e~BTveK-9sHLZGF5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAFOc7GBTnOxgaW; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BAIDUID_BFESS=DAA8F9F0BD801A29B2813502000BF8E9:SL=0:NR=10:FG=1; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; PSINO=2; H_PS_PSSID=34435_31660_34405_34004_34073_34092_26350_34426_34323_22158_34390; delPer=1; BA_HECTOR=8185a12020018421b61gi6ka20q; BCLID=10943521300863382545; BDSFRCVID=boDOJexroG0YyvRHKn7hh7zlD_weG7bTDYLEOwXPsp3LGJLVJeC6EG0Pts1-dEu-EHtdogKK0mOTHv8F_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF=tR3aQ5rtKRTffjrnhPF3-44vXP6-hnjy3bRkX4Q4Wpv_Mnndjn6SQh4Wbttf5q3RymJ42-39LPO2hpRjyxv4y4Ldj4oxJpOJ-bCL0p5aHl51fbbvbURvD-ug3-7qqU5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-jBRIE3-oJqC8hMIt43f; BCLID_BFESS=10943521300863382545; BDSFRCVID_BFESS=boDOJexroG0YyvRHKn7hh7zlD_weG7bTDYLEOwXPsp3LGJLVJeC6EG0Pts1-dEu-EHtdogKK0mOTHv8F_2uxOjjg8UtVJeC6EG0Ptf8g0M5; H_BDCLCKID_SF_BFESS=tR3aQ5rtKRTffjrnhPF3-44vXP6-hnjy3bRkX4Q4Wpv_Mnndjn6SQh4Wbttf5q3RymJ42-39LPO2hpRjyxv4y4Ldj4oxJpOJ-bCL0p5aHl51fbbvbURvD-ug3-7qqU5dtjTO2bc_5KnlfMQ_bf--QfbQ0hOhqP-jBRIE3-oJqC8hMIt43f; Hm_lvt_64ecd82404c51e03dc91cb9e8c025574=1629701482,1629702031,1629702343,1629704515; Hm_lpvt_64ecd82404c51e03dc91cb9e8c025574=1629704515; __yjs_st=2_MDBkZDdkNzg4YzYyZGU2NTM5NzBjZmQ0OTZiMWRmZGUxM2QwYzkwZTc2NTZmMmIxNDJkYzk4NzU1ZDUzN2U3Yjc4ZTJmYjE1YTUzMTljYWFkMWUwYmVmZGEzNmZjN2FlY2M3NDAzOThhZTY5NzI0MjVkMmQ0NWU3MWE1YTJmNGE5NDBhYjVlOWY3MTFiMWNjYTVhYWI0YThlMDVjODBkNWU2NjMwMzY2MjFhZDNkMzVhNGMzMGZkMWY2NjU5YzkxMDk3NTEzODJiZWUyMjEyYTk5YzY4ODUyYzNjZTJjMGM5MzhhMWE5YjU3NTM3NWZiOWQxNmU3MDVkODExYzFjN183XzliY2RhYjgz; ab_sr=1.0.1_ZTc2ZDFkMTU5ZTM0ZTM4MWVlNDU2MGEzYTM4MzZiY2I2MDIxNzY1Nzc1OWZjZGNiZWRhYjU5ZjYwZmNjMTE2ZjIzNmQxMTdiMzIzYTgzZjVjMTY0ZjM1YjMwZTdjMjhiNDRmN2QzMjMwNWRhZmUxYTJjZjZhNTViMGM2ODFlYjE5YTlmMWRjZDAwZGFmMDY4ZTFlNGJiZjU5YzE1MGIxN2FiYTU3NDgzZmI4MDdhMDM5NTQ0MjQxNDBiNzdhMDdl',
    # 'Host': 'fanyi.baidu.com',
    # 'Origin': 'https://fanyi.baidu.com',
    # 'Referer': 'https://fanyi.baidu.com/?aldtype=16047',
    # 'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"',
    # 'sec-ch-ua-mobile': '?0',
    # 'Sec-Fetch-Dest': 'empty',
    # 'Sec-Fetch-Mode': 'cors',
    # 'Sec-Fetch-Site': 'same-origin',
    # 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    # 'X-Requested-With': 'XMLHttpRequest',
}

data = {
    'from': 'en',
    'to': 'zh',
    'query': 'love',
    'transtype': 'realtime',
    'simple_means_flag': '3',
    'sign': '198772.518981',
    'token': '5483bfa652979b41f9c90d91f3de875d',
    'domain': 'common',
}
#下面是常规操作
data = urllib.parse.quote(data).encode('utf-8')
request = urllib.request.Request(url=url,data=data,headers=headers)
response = urllib.request.urlopen(request)
content = response.read().decode('utf-8')

import json
obj = json.loads(content)
print(obj)
```



### 06 ajax的get请求

获取豆瓣电影前十页的数据。**将爬虫的三个步骤封装成三个函数。**即：

**（0）前置步骤：网页请求分析**

**（1） 请求对象的定制**

**（2） 获取响应的数据**

**（3） 下载数据**

```python
# 网页请求分析
# https://movie.douban.com/j/chart/top_list?type=5&interval_id=100%3A90&action=&
# start=0&limit=20
# https://movie.douban.com/j/chart/top_list?type=5&interval_id=100%3A90&action=&
# start=20&limit=20
# https://movie.douban.com/j/chart/top_list?type=5&interval_id=100%3A90&action=&
# start=40&limit=20
# page    1  2   3   4
# start   0  20  40  60
# start （page - 1）*20

import urllib.parse
import urllib.request

def create_request(page):
    base_url = 'https://movie.douban.com/j/chart/top_list?type=5&interval_id=100%3A90&action=&'

    data = {
        'start':(page - 1) * 20,
        'limit':20
    }
    data = urllib.parse.urlencode(data)
    url = base_url + data
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    }

    request = urllib.request.Request(url=url,headers=headers)
    return request


def get_content(request):
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    return content


def down_load(page,content):
    with open('douban_' + str(page) + '.json','w',encoding='utf-8')as fp:
        fp.write(content)

        
# 程序的入口
if __name__ == '__main__':
    start_page = int(input('请输入起始的页码'))
    end_page = int(input('请输入结束的页面'))

    for page in range(start_page,end_page+1):
#         每一页都有自己的请求对象的定制
        request = create_request(page)
#         获取响应的数据
        content = get_content(request)
#         下载
        down_load(page,content)
```

> 当面对多页数据时，可以在当前页清空network中的数据，然后再点击另一页，就可以快速找到这页的数据请求

### 07 ajax的post请求

KFC官网爬取

```python
# 网页分析
# 1页
# http://www.kfc.com.cn/kfccda/ashx/GetStoreList.ashx?op=cname
# post
# cname: 北京
# pid:
# pageIndex: 1
# pageSize: 10

# 2页
# http://www.kfc.com.cn/kfccda/ashx/GetStoreList.ashx?op=cname
# post
# cname: 北京
# pid:
# pageIndex: 2
# pageSize: 10

import urllib.request
import urllib.parse

def create_request(page):
    base_url = 'http://www.kfc.com.cn/kfccda/ashx/GetStoreList.ashx?op=cname'
    data = {
        'cname': '北京',
        'pid':'',
        'pageIndex': page,
        'pageSize': '10'
    }
    data = urllib.parse.urlencode(data).encode('utf-8')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
    }
    request = urllib.request.Request(url=base_url,headers=headers,data=data)
    return request

def get_content(request):
	pass	#与上一个一样


def down_load(page,content):
    with open('kfc_' + str(page) + '.json','w',encoding='utf-8')as fp:
        fp.write(content)
    
if __name__ == '__main__':
    pass 	#与上一个相同

```

### 08 异常：URLError\HTTPError

1.HTTPError类是URLError类的子类

2.导入的包urllib.error.HTTPError 	urllib.error.URLError

3.http错误：**http错误是针对浏览器无法连接到服务器而增加出来的错误提示**。引导并告诉浏览者该页是哪里出了问题。

4.通过urllib发送请求的时候，有可能会发送失败，这个时候如果想让你的代码更加的健壮，可以通**过tryexcept进行捕获异常，异常有两类，URLError\HTTPError**

```python
import urllib.request
import urllib.error

# url = 'https://blog.csdn.net/sulixu/article/details/1198189491'

url = 'http://www.doudan1111.com'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

try:
    request = urllib.request.Request(url = url, headers = headers)
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    print(content)
except urllib.error.HTTPError:
    print('系统正在升级。。。')
except urllib.error.URLError:
    print('我都说了 系统正在升级。。。')
```

### 09 cookie登录

**适用的场景**：数据采集的时候 需要绕过登陆 然后进入到某个页面

**常遇小问题**：个人信息页面是utf-8  但是还报错了编码错误  因为并没有进入到个人信息页面 而是跳转到了登陆页面，那么登陆页面不是utf-8  所以报错

总的来说，无论什么情况下，只要没编码等错误，但还是访问不成功。就是因为**请求头的信息不够**

```python
# 什么情况下访问不成功？
# 因为请求头的信息不够  所以访问不成功

import urllib.request

url = 'https://weibo.cn/6451491586/info'

headers = {
# ':authority': 'weibo.cn',
# ':method': 'GET',
# ':path': '/6451491586/info',
# ':scheme': 'https',
'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
# 'accept-encoding': 'gzip, deflate, br',
'accept-language': 'zh-CN,zh;q=0.9',
'cache-control': 'max-age=0',
#     cookie中携带着你的登陆信息   如果有登陆之后的cookie  那么我们就可以携带着cookie进入到任何页面
'cookie': '_T_WM=24c44910ba98d188fced94ba0da5960e; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFxxfgNNUmXi4YiaYZKr_J_5NHD95QcSh-pSh.pSKncWs4DqcjiqgSXIgvVPcpD; SUB=_2A25MKKG_DeRhGeBK7lMV-S_JwzqIHXVv0s_3rDV6PUJbktCOLXL2kW1NR6e0UHkCGcyvxTYyKB2OV9aloJJ7mUNz; SSOLoginState=1630327279',
# referer  判断当前路径是不是由上一个路径进来的    一般情况下 是做图片防盗链
'referer': 'https://weibo.cn/',
'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"',
'sec-ch-ua-mobile': '?0',
'sec-fetch-dest': 'document',
'sec-fetch-mode': 'navigate',
'sec-fetch-site': 'same-origin',
'sec-fetch-user': '?1',
'upgrade-insecure-requests': '1',
'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
}
# 请求对象的定制
request = urllib.request.Request(url=url,headers=headers)
# 模拟浏览器向服务器发送请求
response = urllib.request.urlopen(request)
# 获取响应的数据
content = response.read().decode('utf-8')

# 将数据保存到本地
with open('weibo.html','w',encoding='utf-8')as fp:
    fp.write(content)
```

### 10 Handler处理器

为什么要学习handler？
	urllib.request.urlopen(url)——不能定制请求头

​	urllib.request.Request(url,headers,data)——可以定制请求头

​	Handler——定制更高级的请求头（随着业务逻辑的复杂 请求对象的定制已经满足不了我们的需求（动态cookie和代理不能使用请求对象的定制）【**也就是应对header中参数动态变化的情况**】

```python
# 需求 使用handler来访问百度  获取网页源码
import urllib.request

url = 'http://www.baidu.com'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}
request = urllib.request.Request(url = url,headers = headers)

# handler   build_opener  open

# （1）获取hanlder对象
handler = urllib.request.HTTPHandler()
# （2）获取opener对象
opener = urllib.request.build_opener(handler)
# (3) 调用open方法
response = opener.open(request)

content = response.read().decode('utf-8')

print(content)
```

### 11 代理服务器

1.代理的常用功能?

- 1.突破自身IP访问限制，访问国外站点。

- 2.访问一些单位或团体内部资源

  ​		扩展：某大学FTP(前提是该代理地址在该资源的允许访问范围之内)，使用教育网内地址段免费代理服务器，就可以用于对教育网开放的各类FTP下载上传，以及各类资料查询共享等服务。

- 3.提高访问速度

  ​		扩展：通常代理服务器都设置一个较大的硬盘缓冲区，当有外界的信息通过时，同时也将其保存到缓冲区中，当其他用户再访问相同的信息时， 则直接由缓冲区中取出信息，传给用户，以提高访问速度。

- 4.隐藏真实IP

  ​		扩展：上网者也可以通过这种方法隐藏自己的IP，免受攻击。

2.代码配置代理

- 创建Reuqest对象

- 创建ProxyHandler对象

- 用handler对象创建opener对象

- 使用opener.open函数发送请求

```python
import urllib.request

url = 'http://www.baidu.com/s?wd=ip'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

# 请求对象的定制
request = urllib.request.Request(url = url,headers= headers)

# 模拟浏览器访问服务器
# 代理ip
proxies = {
    'http':'118.24.219.151:16817'
}
# handler  build_opener  open
hander = urllib.request.URLHandler(proxies=proxies)
openner = urllib.request.build_opener(hander)
response = openner.open(request)

# 获取响应的信息
content = response.read().decode('utf-8')

# 保存
with open('daili.html','w',encoding='utf-8')as fp:
    fp.write(content)
```

#### 代理池

简单实现代理池

```python
import urllib.request

proxies_pool = [
    {'http':'118.24.219.151:16817'},
    {'http':'118.24.219.151:16817'},
]

import random
proxies = random.choice(proxies_pool)
url = 'http://www.baidu.com/s?wd=ip'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

request = urllib.request.Request(url = url,headers=headers)

handler = urllib.request.ProxyHandler(proxies=proxies)
opener = urllib.request.build_opener(handler)
response = opener.open(request)

content = response.read().decode('utf-8')

with open('daili.html','w',encoding='utf-8')as fp:
    fp.write(content)
```



## 09.02 解析

### 01 xpath

```
xpath使用：
注意：提前安装xpath插件
（1）打开chrome浏览器
（2）点击右上角小圆点
（3）更多工具
（4）扩展程序
（5）拖拽xpath插件到扩展程序中
（6）如果crx文件失效，需要将后缀修改zip
（7）再次拖拽
（8）关闭浏览器重新打开
（9）ctrl + shift + x
（10）出现小黑框

1.安装lxml库
	pip install lxml ‐i https://pypi.douban.com/simple
2.导入lxml.etree
	from lxml import etree
	
3.etree.parse() 解析本地文件
	html_tree = etree.parse('XX.html')
4.etree.HTML() 服务器响应文件
	html_tree = etree.HTML(response.read().decode('utf‐8'))
5.html_tree.xpath(xpath路径)
```

#### xpath基本语法（以本地文件为例）

举个例子：

创建一个本地文件：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <title>Title</title>
</head>
<body>
    <ul>
        <li id="l1" class="c1">北京</li>
        <li id="l2">上海</li>
        <li id="c3">深圳</li>
        <li id="c4">武汉</li>
    </ul>

<!--    <ul>-->
<!--        <li>大连</li>-->
<!--        <li>锦州</li>-->
<!--        <li>沈阳</li>-->
<!--    </ul>-->
</body>
</html>
```

xpath基本语法应用：

```python
from lxml import etree
tree = etree.parse('070_尚硅谷_爬虫_解析_xpath的基本使用.html')

#1.路径查询
#	 //：查找所有子孙节点，不考虑层级关系
#	 / ：找直接子节点

#例子1.查找ul下面的li：
li_list = tree.xpath('//body/ul/li')


#2.谓词查询
#	//div[@id]
#	//div[@id="maincontent"]

#例子1.查找所有有id的属性的li标签
li_list = tree.xpath('//ul/li[@id]/text()')		#text()是获取标签中的内容
#输出为：['北京','上海','深圳',武汉]

#例子2.找到id为l1的li标签  注意引号的问题
li_list = tree.xpath('//ul/li[@id="l1"]/text()')
#输出为：['北京']


#3.属性查询
#	//@class

# 例子1.查找到id为l1的li标签的class的属性值
li = tree.xpath('//ul/li[@id="l1"]/@class')
# 输出为：['c1']


#4.模糊查询
#	//div[contains(@id, "he")]
#	//div[starts‐with(@id, "he")]

# 例子1.查询id中包含l的li标签
li_list = tree.xpath('//ul/li[contains(@id,"l")]/text()')
# 例子2.查询id的值以l开头的li标签
li_list = tree.xpath('//ul/li[starts-with(@id,"c")]/text()')
#输出：['深圳',武汉]


#5.内容查询
#	//div/h1/text()


#6.逻辑运算
#	//div[@id="head" and @class="s_down"]
#	//title | //price

#例子1.查询id为l1和class为c1的
li_list = tree.xpath('//ul/li[@id="l1" and @class="c1"]/text()')
#例子2
li_list = tree.xpath('//ul/li[@id="l1"]/text() | //ul/li[@id="l2"]/text()')
```

#### 解析服务器响应文件：获取百度网盘的百度一下

步骤：

- **（1） 获取网页的源码**

- **（2） 解析   解析的服务器响应的文件  etree.HTML**

-    **(3)  打印**

```python
import urllib.request

url = 'https://www.baidu.com/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}
# 请求对象的定制
request = urllib.request.Request(url = url,headers = headers)
# 模拟浏览器访问服务器
response = urllib.request.urlopen(request)
# 获取网页源码
content = response.read().decode('utf-8')

# 解析网页源码 来获取我们想要的数据
from lxml import etree

# 解析服务器响应的文件
tree = etree.HTML(content)

# 获取想要的数据  xpath的返回值是一个列表类型的数据
result = tree.xpath('//input[@id="su"]/@value')[0]

print(result)
```

#### 完整案例：站长素材爬取

需求：下载的前十页的图片

```python
# 网页源码分析：该网站第一页和后面页数的网址不同
# https://sc.chinaz.com/tupian/qinglvtupian.html   1
# https://sc.chinaz.com/tupian/qinglvtupian_page.html 2-10页

import urllib.request
from lxml import etree

def create_request(page):
    if(page == 1):
        url = 'https://sc.chinaz.com/tupian/qinglvtupian.html'
    else:
        url = 'https://sc.chinaz.com/tupian/qinglvtupian_' + str(page) + '.html'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    }

    request = urllib.request.Request(url = url, headers = headers)
    return request

def get_content(request):
    response = urllib.request.urlopen(request)
    content = response.read().decode('utf-8')
    return content


def down_load(content):
#     下载图片
    # urllib.request.urlretrieve('图片地址','文件的名字')
    tree = etree.HTML(content)

    name_list = tree.xpath('//div[@id="container"]//a/img/@alt')

    # 一般涉及图片的网站都会进行懒加载
    src_list = tree.xpath('//div[@id="container"]//a/img/@src2')

    for i in range(len(name_list)):
        name = name_list[i]
        src = src_list[i]
        url = 'https:' + src
		#下载图片
        urllib.request.urlretrieve(url=url,filename='./loveImg/' + name + '.jpg')


if __name__ == '__main__':
    start_page = int(input('请输入起始页码'))
    end_page = int(input('请输入结束页码'))

    for page in range(start_page,end_page+1):
        # (1) 请求对象的定制
        request = create_request(page)
        # （2）获取网页的源码
        content = get_content(request)
        # （3）下载
        down_load(content)

```

> 当遇到网站有懒加载的时候，获取数据的时候需要好好分析一下，**xpath路径应该是懒加载前的路径，而不是加载后的路径**





### 02 JsonPath

jsonpath用来解析json数据，不同于xpath，jsonpath只能解析本地的数据

```
jsonpath的使用：
	obj = json.load(open('json文件', 'r', encoding='utf‐8'))
	ret = jsonpath.jsonpath(obj, 'jsonpath语法')
```

教程连接（http://blog.csdn.net/luxideyao/article/details/77802389）

##### 举个例子

在本地创建一个json文件：

```json
{ "store": {
    "book": [
      { "category": "修真",
        "author": "六道",
        "title": "坏蛋是怎样练成的",
        "price": 8.95
      },
      { "category": "修真",
        "author": "天蚕土豆",
        "title": "斗破苍穹",
        "price": 12.99
      },
      { "category": "修真",
        "author": "唐家三少",
        "title": "斗罗大陆",
        "isbn": "0-553-21311-3",
        "price": 8.99
      },
      { "category": "修真",
        "author": "南派三叔",
        "title": "星辰变",
        "isbn": "0-395-19395-8",
        "price": 22.99
      }
    ],
    "bicycle": {
      "author": "老马",
      "color": "黑色",
      "price": 19.95
    }
  }
}
```

jsonpath解析：

```python

import json
import jsonpath


obj = json.load(open('073_尚硅谷_爬虫_解析_jsonpath.json','r',encoding='utf-8'))

# 书店所有书的作者
# author_list = jsonpath.jsonpath(obj,'$.store.book[*].author')
# print(author_list)

# 所有的作者
# author_list = jsonpath.jsonpath(obj,'$..author')
# print(author_list)

# store下面的所有的元素
# tag_list = jsonpath.jsonpath(obj,'$.store.*')
# print(tag_list)

# store里面所有东西的price
# price_list = jsonpath.jsonpath(obj,'$.store..price')
# print(price_list)

# 第三个书
# book = jsonpath.jsonpath(obj,'$..book[2]')
# print(book)

# 最后一本书
# book = jsonpath.jsonpath(obj,'$..book[(@.length-1)]')
# print(book)

# 	前面的两本书
# book_list = jsonpath.jsonpath(obj,'$..book[0,1]')
# book_list = jsonpath.jsonpath(obj,'$..book[:2]')
# print(book_list)

# 条件过滤需要在（）的前面添加一个？
# 	 过滤出所有的包含isbn的书。
# book_list = jsonpath.jsonpath(obj,'$..book[?(@.isbn)]')
# print(book_list)


# 哪本书超过了10块钱
book_list = jsonpath.jsonpath(obj,'$..book[?(@.price>10)]')
print(book_list)
```

#### 案例：淘票票数据解析

```python
import urllib.request

url = 'https://dianying.taobao.com/cityAction.json?activityId&_ksTS=1629789477003_137&jsoncallback=jsonp138&action=cityAction&n_s=new&event_submit_doGetAllRegion=true'

# header中需要注释了冒号开头的键值对，以及accept-encoding
headers = {
    # ':authority': 'dianying.taobao.com',
    # ':method': 'GET',
    # ':path': '/cityAction.json?activityId&_ksTS=1629789477003_137&jsoncallback=jsonp138&action=cityAction&n_s=new&event_submit_doGetAllRegion=true',
    # ':scheme': 'https',
    'accept': 'text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01',
    # 'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cookie': 'cna=UkO6F8VULRwCAXTqq7dbS5A8; miid=949542021157939863; sgcookie=E100F01JK9XMmyoZRigjfmZKExNdRHQqPf4v9NIWIC1nnpnxyNgROLshAf0gz7lGnkKvwCnu1umyfirMSAWtubqc4g%3D%3D; tracknick=action_li; _cc_=UIHiLt3xSw%3D%3D; enc=dA18hg7jG1xapfVGPHoQCAkPQ4as1%2FEUqsG4M6AcAjHFFUM54HWpBv4AAm0MbQgqO%2BiZ5qkUeLIxljrHkOW%2BtQ%3D%3D; hng=CN%7Czh-CN%7CCNY%7C156; thw=cn; _m_h5_tk=3ca69de1b9ad7dce614840fcd015dcdb_1629776735568; _m_h5_tk_enc=ab56df54999d1d2cac2f82753ae29f82; t=874e6ce33295bf6b95cfcfaff0af0db6; xlly_s=1; cookie2=13acd8f4dafac4f7bd2177d6710d60fe; v=0; _tb_token_=e65ebbe536158; tfstk=cGhRB7mNpnxkDmUx7YpDAMNM2gTGZbWLxUZN9U4ulewe025didli6j5AFPI8MEC..; l=eBrgmF1cOsMXqSxaBO5aFurza77tzIRb8sPzaNbMiInca6OdtFt_rNCK2Ns9SdtjgtfFBetPVKlOcRCEF3apbgiMW_N-1NKDSxJ6-; isg=BBoas2yXLzHdGp3pCh7XVmpja8A8S54lyLj1RySTHq14l7vRDNufNAjpZ2MLRxa9',
    'referer': 'https://dianying.taobao.com/',
    'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"',
    'sec-ch-ua-mobile': '?0',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
    'x-requested-with': 'XMLHttpRequest',
}

request = urllib.request.Request(url = url, headers = headers)
response = urllib.request.urlopen(request)
content = response.read().decode('utf-8')

# split 切割。		因为获取到的数据时jsonp的，这里暂时不解释jsonp是什么，直接采用split来获取数据
content = content.split('(')[1].split(')')[0]

with open('074_尚硅谷_爬虫_解析_jsonpath解析淘票票.json','w',encoding='utf-8')as fp:
    fp.write(content)

import json
import jsonpath

obj = json.load(open('074_尚硅谷_爬虫_解析_jsonpath解析淘票票.json','r',encoding='utf-8'))

city_list = jsonpath.jsonpath(obj,'$..regionName')

print(city_list)
```

### 03 beautifulSoup

#### 基本简介

1.BeautifulSoup简称：
		bs4
2.什么是BeatifulSoup？
		BeautifulSoup，和lxml一样，是一个html的解析器，主要功能也是解析和提取数据
3.优缺点？
		缺点：效率没有lxml的效率高
		优点：接口设计人性化，使用方便

#### 安装以及创建

1.安装
		pip install bs4
2.导入
		from bs4 import BeautifulSoup
3.创建对象
		服务器响应的文件生成对象
				soup = BeautifulSoup(response.read().decode(), 'lxml')
		本地文件生成对象
				soup = BeautifulSoup(open('1.html'), 'lxml')
		注意：默认打开文件的编码格式gbk所以需要指定打开编码格式（直接在open中新增一个encode参数）

#### beautifulSoup语法：以本地文件解析为例

创建一个本地html文件：

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
    <div>
        <ul>
            <li id="l1">张三</li>
            <li id="l2">李四</li>
            <li>王五</li>
            <a href="" id="" class="a1">尚硅谷</a>
            <span>嘿嘿嘿</span>
        </ul>
    </div>
    <a href="" title="a2">百度</a>
    <div id="d1">
        <span>
            哈哈哈
        </span>
    </div>
    <p id="p1" class="p1">呵呵呵</p>
</body>
</html>
```

语法及函数介绍：

```python
from bs4 import BeautifulSoup

# 默认打开的文件的编码格式是gbk 所以在打开文件的时候需要指定编码
soup = BeautifulSoup(open('075_尚硅谷_爬虫_解析_bs4的基本使用.html',encoding='utf-8'),'lxml')

# 1.根据标签名查找节点,找到的是第一个符合条件的数据
print(soup.a)
# 2.获取标签的属性和属性值
print(soup.a.attrs)

# bs4的一些函数
#一、节点定位：
# （1）find
# 例子1.返回的是第一个符合条件的数据
print(soup.find('a'))

# 例子2.根据title的值来找到对应的标签对象
print(soup.find('a',title="a2"))

# 例子3.根据class的值来找到对应的标签对象  注意的是class需要添加下划线
print(soup.find('a',class_="a1"))


# （2）find_all  
# 例子1.返回的是一个列表 并且返回了所有的a标签
print(soup.find_all('a'))

# 例子2.如果想获取的是多个标签的数据 那么需要在find_all的参数中添加的是列表的数据
print(soup.find_all(['a','span']))

# 例子3. limit的作用是查找前几个数据
print(soup.find_all('li',limit=2))


# （3）select（推荐）
# 例子1：select方法返回的是一个列表  并且会返回多个数据
print(soup.select('a'))

# 例子2：可以通过.代表class  我们把这种操作叫做类选择器
print(soup.select('.a1'))

# 例子3：可以通过#代表id
print(soup.select('#l1'))

# 例子4：属性选择器---通过属性来寻找对应的标签
# 查找到li标签中有id的标签
print(soup.select('li[id]'))
# 查找到li标签中id为l2的标签
print(soup.select('li[id="l2"]'))

# 层级选择器
# 例子5：后代选择器
# 找到的是div下面的li，获取所有的后代标签，不管多少级
print(soup.select('div li'))

# 例子6：子代选择器
#  某标签的第一级子标签
# 注意：很多的计算机编程语言中 如果不加空格不会输出内容  但是在bs4中 不会报错 会显示内容
print(soup.select('div > ul > li'))

# 例子7：找到a标签和li标签的所有的对象
print(soup.select('a,li'))


# 二、节点信息
# 1.获取节点内容：obj.string和obj.get_text()
obj = soup.select('#d1')[0]
# 如果标签对象中 只有内容 那么string和get_text()都可以使用
# 如果标签对象中 除了内容还有标签 那么string就获取不到数据 而get_text()是可以获取数据
# 我们一般情况下  推荐使用get_text()
print(obj.string)
print(obj.get_text())

# 2.节点的属性：tag.name获取标签名；tag.attrs将属性值作为一个字典返回
obj = soup.select('#p1')[0]
# name是标签的名字
print(obj.name)
# 将属性值左右一个字典返回
print(obj.attrs)

# 3.获取节点的属性
obj = soup.select('#p1')[0]

print(obj.attrs.get('class'))
print(obj.get('class'))
print(obj['class'])

```

>bs4中的常用函数：
>
>- find
>- find_all
>- select   注意select返回的是一个列表

#### 案例：星巴克数据爬取

```python
import urllib.request

url = 'https://www.starbucks.com.cn/menu/'
response = urllib.request.urlopen(url)
content = response.read().decode('utf-8')

from bs4 import BeautifulSoup

soup = BeautifulSoup(content,'lxml')

# //ul[@class="grid padded-3 product"]//strong/text()
name_list = soup.select('ul[class="grid padded-3 product"] strong')

for name in name_list:
    print(name.get_text())
```



## 09.03 selenium

### selenium

#### 01 基本介绍

1.什么是selenium？
（1）Selenium是一个用于Web应用程序测试的工具。
（2）Selenium 测试直接运行在浏览器中，就像真正的用户在操作一样。
（3）支持通过各种driver（FirfoxDriver，IternetExplorerDriver，OperaDriver，ChromeDriver）驱动
真实浏览器完成测试。
（4）selenium也是支持无界面浏览器操作的。

2.为什么使用selenium？
模拟浏览器功能，自动执行网页中的js代码，实现动态加载

#### 02 基本使用

```python
# （1）导入selenium
from selenium import webdriver

# (2) 创建浏览器操作对象
path = 'chromedriver.exe'
browser = webdriver.Chrome(path)

# （3）访问网站
url = 'https://www.jd.com/'
browser.get(url)

# page_source获取网页源码
content = browser.page_source
print(content)
```

#### 03 selenium的元素定位

##### 旧版方法

元素定位：自动化要做的就是模拟鼠标和键盘来操作来操作这些元素，点击、输入等等。操作这些元素前首先
要找到它们，WebDriver提供很多定位元素的方法

```
方法：
1.find_element_by_id			根据id来找到对象
	eg:button = browser.find_element_by_id('su')
2.find_elements_by_name			根据标签属性的属性值来获取对象的
	eg:name = browser.find_element_by_name('wd')
3.find_elements_by_xpath		根据xpath语句来获取对象
	eg:xpath1 = browser.find_elements_by_xpath('//input[@id="su"]')
4.find_elements_by_tag_name		根据标签的名字来获取对象
	eg:names = browser.find_elements_by_tag_name('input')
5.find_elements_by_css_selector	使用的bs4的语法来获取对象
	eg:my_input = browser.find_elements_by_css_selector('#kw')[0]
6.find_elements_by_link_text	根据链接的文本来获取对象
	eg:browser.find_element_by_link_text("新闻")
```

> 最常用的是根据id、xpath语法和bs4语法来获取对象

##### 新版方法

新版中已经废弃了上面的用法，但任然可以使用。

新版用的是find_element（by，value）和find_elements（by，value）方法。

- by是selenium.webdriver.common.by中的常量，有ID,TAG_NAME, CLASS_NAME, NAME

```python
from selenium.webdriver.common.by import By

find_element(str=By.TAG_NAME,value='su')
```



#### 04 selenium的访问元素信息与交互

##### 访问元素信息

- 获取元素属性：.get_attribute('class')

- 获取元素文本：.text

  ​	获取的是标签对中的文本

- 获取标签名：.tag_name

##### 交互

- 点击:click()
- 输入:send_keys()
- 后退操作:browser.back()
- 前进操作:browser.forword()
- 模拟JS滚动:
  - js='document.documentElement.scrollTop=100000'
  - browser.execute_script(js) 执行js代码
- 获取网页代码：page_source
- 退出：browser.quit()

```python
from selenium import webdriver

# 创建浏览器对象
path = 'chromedriver.exe'
browser = webdriver.Chrome(path)

# url
url = 'https://www.baidu.com'
browser.get(url)

import time
time.sleep(2)

# 获取文本框的对象
input = browser.find_element_by_id('kw')

# 在文本框中输入周杰伦
input.send_keys('周杰伦')

time.sleep(2)

# 获取百度一下的按钮
button = browser.find_element_by_id('su')

# 点击按钮
button.click()

time.sleep(2)

# 滑到底部
js_bottom = 'document.documentElement.scrollTop=100000'
browser.execute_script(js_bottom)

time.sleep(2)

# 获取下一页的按钮
next = browser.find_element_by_xpath('//a[@class="n"]')

# 点击下一页
next.click()

time.sleep(2)

# 回到上一页
browser.back()

time.sleep(2)

# 回去
browser.forward()

time.sleep(3)

# 退出
browser.quit()
```

### Phantomjs

1.什么是Phantomjs？

（1）是一个无界面的浏览器

（2）支持页面元素查找，js的执行等

（3）由于不进行css和gui渲染，运行效率要比真实的浏览器要快很多

2.如何使用Phantomjs？

（1）获取PhantomJS.exe文件路径path

（2）browser = webdriver.PhantomJS(path)

（3）browser.get(url)

扩展：保存屏幕快照:browser.save_screenshot('baidu.png')

> 在当前文件目录下放入PhantomJS.exe可执行文件，然后只用改两条代码，其他全一样：
>
> ```python
> path = 'phantomjs.exe'
> browser = webdriver.PhantomJS(path)
> ```
>
> 停更了，都不用了



### Chrome handless

Chrome-headless 模式， Google 针对 Chrome 浏览器 59版 新增加的一种模式，可以让你**不打开UI界面的情况下使用 Chrome 浏览器，所以运行效果与 Chrome 保持完美一致。**

**使用Chrome handless的方法可以直接封装好。封装成下面的share_browser函数，以后要用到，直接调用这个函数就行。如果是不同的服务器用的话，只用就该函数中的path为该服务器的chrome.exe安装位置就行**

```python
#导入包
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# 函数封装
def share_browser():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')

    # path是你自己的chrome浏览器的文件路径
    path = r'C:\Users\10277\AppData\Local\Google\Chrome\Application\chrome.exe'
    chrome_options.binary_location = path

    browser = webdriver.Chrome(chrome_options=chrome_options)
    return browser

#前面的直接copy，然后直接写下面的需求代码
browser = share_browser()
url = 'https://www.baidu.com'
browser.get(url)
```



## 09.04 requests

### 基本使用

#### 01.文档：

​	官方文档
​		http://cn.python‐requests.org/zh_CN/latest/
​	快速上手
​		http://cn.python‐requests.org/zh_CN/latest/user/quickstart.html

#### 02.response的属性以及类型

| 属性                 | 介绍                                   |
| -------------------- | -------------------------------------- |
| 返回类型             | models.Response                        |
| response.text        | 获取网站源码                           |
| response.encoding    | 访问或定制编码方式                     |
| response.url         | 获取请求的url                          |
| response.content     | 响应的字节类型（返回的是二进制的数据） |
| response.status_code | 响应的状态码                           |
| response.headers     | 响应的头信息                           |

#### 03.基本使用

```python
import requests

url = 'http://www.baidu.com'
response = requests.get(url=url)
# 以字符串的形式来返回了网页的源码
print(response.text)
```

### get请求

语法：requests.get(url,params,kwargs)

```python
import requests

url = 'https://www.baidu.com/s'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}
data = {
    'wd':'北京'
}

# url  请求资源路径
# params 参数
# kwargs 字典
response = requests.get(url=url,params=data,headers=headers)

content = response.text
print(content)
```

**总结：**

（1）参数使用params传递
（2）参数无需urlencode编码
（3）不需要请求对象的定制
（4）请求资源路径中的？可以加也可以不加

### post请求

语法：requests.post(url,data,kwargs)

```python
import requests

url = 'https://fanyi.baidu.com/sug'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}
data = {
    'kw': 'eye'
}

# url 请求地址
# data 请求参数
# kwargs 字典
response = requests.post(url=url,data=data,headers=headers)

content =response.text

import json
obj = json.loads(content,encoding='utf-8')
print(obj)
```

**总结：**

（1）post请求 是不需要编解码

（2）请求资源路径后面可以不加?

（3）不需要请求对象的定制

（4）get请求的参数名字是params；post请求的参数的名字是data

### 代理

直接在请求中设置proxies参数

```python
import requests

url = 'http://www.baidu.com/s?'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
}
data = {
    'wd':'ip'
}
proxy = {
    'http':'212.129.251.55:16816'
}

response = requests.get(url = url,params=data,headers = headers,proxies = proxy)

content = response.text
with open('daili.html','w',encoding='utf-8')as fp:
    fp.write(content)
```

### cookie定制：以中文诗词网为例

#### 需求：

通过登陆  然后进入到主页面。但登录界面有验证码，需要解决验证码问题。

#### 登录分析：

1.先找到登录的接口，也就是将登录数据发送到后端的接口。tips：输入一个错误的密码或验证码，然后别点下一步，在其中找login接口

2.通过找到的登陆接口我们发现 登陆的时候需要的参数很多：

```
_VIEWSTATE: /m1O5dxmOo7f1qlmvtnyNyhhaUrWNVTs3TMKIsm1lvpIgs0WWWUCQHl5iMrvLlwnsqLUN6Wh1aNpitc4WnOt0So3k6UYdFyqCPI6jWSvC8yBA1Q39I7uuR4NjGo=
__VIEWSTATEGENERATOR: C93BE1AE
from: http://so.gushiwen.cn/user/collect.aspx
email: 595165358@qq.com
pwd: action
code: PId7
denglu: 登录
```

我们观察到_VIEWSTATE   __VIEWSTATEGENERATOR  code三个值是可以变化的量

#### 难点分析:

1 _VIEWSTATE   __VIEWSTATEGENERATOR  

​		（1）**一般情况看不到的数据 都是在页面的源码中。**

​		（2）我们观察到这两个数据在页面的源码中 所以我们需要获取页面的源码 然后进行解析就可以获取了

2 验证码

​		使用requests中的session来获取验证码图片，以此保证和第一次请求页面时验证码相同而没有刷新

​		验证码是图片，下载文件时应该以二进制数据，即response.content



#### 实现

```python
import requests

# 一、这是登陆页面的url地址
url = 'https://so.gushiwen.cn/user/login.aspx?\from=http://so.gushiwen.cn/user/collect.aspx'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36'
}

# 获取页面的源码
response = requests.get(url = url,headers = headers)
content = response.text

# 解析页面源码  然后获取_VIEWSTATE   __VIEWSTATEGENERATOR
from bs4 import BeautifulSoup

soup = BeautifulSoup(content,'lxml')

# 获取_VIEWSTATE
viewstate = soup.select('#__VIEWSTATE')[0].attrs.get('value')
# 获取__VIEWSTATEGENERATOR
viewstategenerator = soup.select('#__VIEWSTATEGENERATOR')[0].attrs.get('value')

# 获取验证码图片
code = soup.select('#imgCode')[0].attrs.get('src')
code_url = 'https://so.gushiwen.cn' + code

# 有坑:这个方法在请求验证码时，会刷新验证码，与前面的不是同一个了，
# import urllib.request
# urllib.request.urlretrieve(url=code_url,filename='code.jpg')
# requests里面有一个方法 session（）  通过session的返回值 就能使用请求变成一个对象

session = requests.session()
# 验证码的url的内容
response_code = session.get(code_url)
# 注意此时要使用二进制数据  因为我们要使用的是图片的下载
content_code = response_code.content
# wb的模式就是将二进制数据写入到文件
with open('code.jpg','wb')as fp:
    fp.write(content_code)


# 获取了验证码的图片之后 下载到本地 然后观察验证码  观察之后 然后在控制台输入这个验证码 就可以将这个值给
# code的参数 就可以登陆

code_name = input('请输入你的验证码')


# 二、点击登陆
url_post = 'https://so.gushiwen.cn/user/login.aspx?from=http%3a%2f%2fso.gushiwen.cn%2fuser%2fcollect.aspx'
data_post = {
    '__VIEWSTATE': viewstate,
    '__VIEWSTATEGENERATOR': viewstategenerator,
    'from': 'http://so.gushiwen.cn/user/collect.aspx',
    'email': '595165358@qq.com',
    'pwd': 'action',
    'code': code_name,
    'denglu': '登录',
}

response_post = session.post(url = url, headers = headers, data = data_post)

content_post = response_post.text

with open('gushiwen.html','w',encoding= ' utf-8')as fp:
    fp.write(content_post)
```

#### 超级鹰打码平台的使用

这种平台可以帮助我们自动解决验证码的问题，而不用手动输入验证码。



## 09.05 scrapy

### scrapy

#### 1.scrapy是什么？

Scrapy是一个**为了爬取网站数据，提取结构性数据而编写的应用框架**。 可以应用在包括数据挖掘，信息处理
或存储历史数据等一系列的程序中。

#### 2.scrapy项目的创建以及运行

```
1. 创建爬虫的项目   scrapy startproject 项目的名字
                 注意：项目的名字不允许使用数字开头  也不能包含中文
2. 创建爬虫文件
                 要在spiders文件夹中去创建爬虫文件
                 cd 项目的名字\项目的名字\spiders
                 cd scrapy_baidu_091\scrapy_baidu_091\spiders

                 创建爬虫文件
                 scrapy genspider 爬虫文件的名字  要爬取网页
                 eg：scrapy genspider baidu  http://www.baidu.com
                 一般情况下不需要添加http协议  因为start_urls的值是根据allowed_domains
                 修改的  所以添加了http的话  那么start_urls就需要我们手动去修改了
3. 运行爬虫代码
                 scrapy crawl 爬虫的名字
                 eg：
                 scrapy crawl baidu
```

##### scrapy项目的结构

```
项目名字
	项目名字
		spiders文件夹
            __init__.py
            自定义的爬虫文件.py 	‐‐‐》由我们自己创建，是实现爬虫核心功能的文件
        __init__.py
        items.py 				‐‐‐》定义数据结构的地方，是一个继承自scrapy.Item的类
        middlewares.py 			‐‐‐》中间件 代理
        pipelines.py 			‐‐‐》管道文件，里面只有一个类，用于处理下载数据的后续处理
                                    默认是300优先级，值越小优先级越高（1‐1000）
        settings.py 			‐‐‐》配置文件 比如：是否遵守robots协议，User‐Agent定义等
```

##### 自定义的爬虫文件

```python
import scrapy

class BaiduSpider(scrapy.Spider):
    name = 'baidu'		# 爬虫的名字  用于运行爬虫的时候 使用的值
    # 允许访问的域名
    allowed_domains = ['https://car.autohome.com.cn/price/brand-15.html']
    # 起始的url地址  指的是第一次要访问的域名
    # start_urls 是在allowed_domains的前面添加一个http://
    #             在 allowed_domains的后面添加一个/
    # 注意如果你的请求的接口是html为结尾的  那么是不需要加/的
    start_urls = ['https://car.autohome.com.cn/price/brand-15.html']

    # 相当于框架默认执行了： response = urllib.request.urlopen()和
    #       response  = requests.get()
    def parse(self, response):			#‐‐‐》解析数据的回调函数
        name_list = response.xpath('//div[@class="main-title"]/a/text()')
        price_list = response.xpath('//div[@class="main-lever"]//span/span/text()')

        for i in range(len(name_list)):
            name = name_list[i].extract()
            price = price_list[i].extract()
            print(name,price)
```

##### response的属性和方法

```
response的属性和方法
    response.text   获取的是响应的字符串
    response.body   获取的是二进制数据
    response.xpath  可以直接是xpath方法来解析response中的内容
    response.extract()   提取seletor对象的data属性值
    response.extract_first() 提取的seletor列表的第一个数据
```

#### 3.scrapy架构组成

```
（1）引擎					‐‐‐》自动运行，无需关注，会自动组织所有的请求对象，分发给下载器
（2）下载器					‐‐‐》从引擎处获取到请求对象后，请求数据
（3）spiders					‐‐‐》Spider类定义了如何爬取某个(或某些)网站。包括了爬取的动作(例如:是否跟进链接)以及如何从网页的内容中提取结构化数据(爬取item)。 换句话说，Spider就是您定义爬取的动作及分析某个网页(或者是有些网页)的地方。
（4）调度器 					‐‐‐》有自己的调度规则，无需关注
（5）管道（Item pipeline） 	‐‐‐》最终处理数据的管道，会预留接口供我们处理数据
当Item在Spider中被收集之后，它将会被传递到Item Pipeline，一些组件会按照一定的顺序执行对Item的处理。
每个item pipeline组件(有时称之为“Item Pipeline”)是实现了简单方法的Python类。他们接收到Item并通过它执行一些行为，同时也决定此Item是否继续通过pipeline，或是被丢弃而不再进行处理。

以下是item pipeline的一些典型应用：
1. 清理HTML数据
2. 验证爬取的数据(检查item包含某些字段)
3. 查重(并丢弃)
4. 将爬取结果保存到数据库中
```



#### 4.scrapy工作原理

<img src="C:\Users\10277\AppData\Roaming\Typora\typora-user-images\image-20211113205215545.png" alt="image-20211113205215545" style="zoom:150%;" />

### scrapy shell

#### 1.什么是scrapy shell？

Scrapy终端，是一个交互终端，供您**在未启动spider的情况下尝试及调试您的爬取代码**。 其本意是用来测试提取

数据的代码，不过您可以将其作为正常的Python终端，在上面测试任何的Python代码。

该终端是用来测试XPath或CSS表达式，查看他们的工作方式及从爬取的网页中提取的数据。 在编写您的spider时，该终端提供了交互性测试您的表达式代码的功能，免去了每次修改后运行spider的麻烦。

一旦熟悉了Scrapy终端后，您会发现其在开发和调试spider时发挥的巨大作用。

#### 2.安装ipython

安装：pip install ipython
简介：如果您安装了 IPython ，Scrapy终端将使用 IPython (替代标准Python终端)。 IPython 终端与其他相
比更为强大，提供智能的自动补全，高亮输出，及其他特性。



**总之，也就是可以用这个工具做一些快速的调试。特别是在项目很大的时候，对一些小模块做调试**



### 案例1：当当网数据爬取

### 案例2：电影天堂数据爬取

### CrawlSpider

#### 介绍

```
1.继承自scrapy.Spider
2.独门秘笈
		CrawlSpider可以定义规则，再解析html内容的时候，可以根据链接规则提取出指定的链接，然后再向这些链接发送请求
		所以，如果有需要跟进链接的需求，意思就是爬取了网页之后，需要提取链接再次爬取，使用CrawlSpider是非常合适的
```

#### 提取链接

链接提取器，在这里就可以写规则提取指定链接

```python
from scrapy.linkextractors import LinkExtractor
scrapy.linkextractors.LinkExtractor(
allow = (), # 正则表达式 提取符合正则的链接			
deny = (), # (不用)正则表达式 不提取符合正则的链接
allow_domains = (), # （不用）允许的域名
deny_domains = (), # （不用）不允许的域名
restrict_xpaths = (), # xpath，提取符合xpath规则的链接
restrict_css = () # 提取符合选择器规则的链接
)#allow、restrict_xpaths、restrict_css这三个最常用

#模拟使用
#正则用法：links = LinkExtractor(allow=r'list_23_\d+\.html')
#xpath用法：links = LinkExtractor(restrict_xpaths=r'//div[@class="x"]')
#css用法：links = LinkExtractor(restrict_css='.x')

#提取连接
link.extract_links(response)
```

#### CrawlSpider运行原理

#### CrawlSpider案例

#### 数据入库



### 日志信息和日志等级

settings.py文件设置：
默认的级别为DEBUG，会显示上面所有的信息

在配置文件中 settings.py

- LOG_FILE : 将屏幕显示的信息全部记录到文件中，屏幕不再显示，注意文件后缀一定是.log

- LOG_LEVEL : 设置日志显示的等级，就是显示哪些，不显示哪些

### scrapy的post请求

```
# post请求 如果没有参数 那么这个请求将没有任何意义
# 所以start_urls 也没有用了。 parse方法也没有用了。所以注释了start_urls和parse方法

（1）重写start_requests方法：
		def start_requests(self)
(2) start_requests的返回值：
		scrapy.FormRequest(url=url, headers=headers, callback=self.parse_item, formdata=data)
		url: 要发送的post地址
		headers：可以定制头信息
		callback: 回调函数
		formdata: post所携带的数据，这是一个字典
```

举个例子：

```python
import scrapy
import json

class TestpostSpider(scrapy.Spider):
    name = 'testpost'
    allowed_domains = ['https://fanyi.baidu.com/sug']

    def start_requests(self):
        url = 'https://fanyi.baidu.com/sug'
        data = {
            'kw': 'final'
        }
        yield scrapy.FormRequest(url=url,formdata=data,callback=self.parse_second)

    def parse_second(self,response):
        content = response.text
        obj = json.loads(content,encoding='utf-8')
        print(obj)
```

### 代理

```
（1）到settings.py中，打开一个选项
DOWNLOADER_MIDDLEWARES = {
'postproject.middlewares.Proxy': 543,
}
（2）到middlewares.py中写代码
def process_request(self, request, spider):
request.meta['proxy'] = 'https://113.68.202.10:9999'
return None
```







# 10 pytorch



## 11 sklearn



