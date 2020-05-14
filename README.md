# Command Line Cheatsheet

- [Bash](#1-bash)
- [Hadoop](#2-hadoop)
- [SSH](#3-ssh)
- [Git](#4-git)
- [Docker](#5-docker)
- [Conda](#6-conda)

## 1. Bash
Source: <https://www.cs.cmu.edu/~15131/f17/>

### 1.1 Command
* ls / tree: -a 显示所有, a代表all; -l显示详细内容(权限)
* cat / less <filename>: 显示文件内容; /banana表示查抄; 按q键退出
* cp <source> <destination>: 拷贝文件
* mv <source> <destination>: 移动文件
* rm <filename>: 删除文件; rm -r folder2/ : 删除文件夹
* mkdir <directory>: 建立路径(文件夹)
* touch <file>: 建立空文件
* echo <text>: 打印文本
* fg/bg: foreground/background任务。例如fg 1(任务编号); 用jobs查看当前任务
* w/who/whoami: w/who查看已登陆用户; whoami查看目前登陆用户
* su: 切换用户; su -user; su root;
* chown -R user:usergroup filename
* chmod及权限: <https://www.cnblogs.com/peida/archive/2012/11/29/2794010.html>
* vim: https://www.runoob.com/linux/linux-vim.html
* |: 管道操作
* grep(global reg ex print) / sed(stream editor): 正则表达式和自动文件操作
* &gt;&gt;: append stdout; >: overwrite stdout; 2>, 2>>: stderr
    - cat asdf hello.txt 2>&1 > alloutput.txt:把2句柄赋值传递给1句柄的地址(0是stdin, 1是stdout, 2是stderr); 功能为把stderr发送到stdout, 然后把stdout发送到文件
    - 2>&1 要写到后面: ls a 1>&2 2> b.txt; ls a > b.txt 2>&1
    - /dev/null: 不打印任何东西, 可以理解为垃圾桶位置
* $: 把输出变成输入的一部分, 类比pipe; touch myfile-$(date +%s).txt
* make

### 1.2 Points
* setting variables:
    - myvariable="hello”: 不可被其他程序引用。获得变量值用$, 例如echo $myvariable,  echo lone${another_var}s    
    - export anothervar="some string”: 可被外部或者其他程序引用
    - permission: <https://www.cnblogs.com/peida/archive/2012/11/29/2794010.html>
* {} 生成序列: mv {1.txt,2.txt}; mv Bash/2{.txt,}

## 2. Hadoop
### 2.1 hdfs
* hdfs dfs -cat fileName
* hdfs dfs -mkdir msia
* hdfs dfs -ls
* hdfs dfs -copyFromLocal <localdir> <serverdir>

### 2.2 MapReduce
* ant: compile Java file, in the dir of build.xml

### 2.3 Example - 1st Homework
1. build and compile java code
ant
2. mkdir in hdfs  
    * hdfs dfs -mkdir wc   
    * hdfs dfs -mkdir wc/input   
    * hdfs dfs -copyFromLocal text.txt wc/input   
3. run local(linux system) java using yarn [hdfs input dir] [hdfs output dir]  # output folder should not exist previously   
    * yarn jar WordCount.jar /user/zzu8431/wc/input /user/zzu8431/wc/output   
4. check output  
    * hdfs dfs -ls wc/output  
    * hdfs dfs -cat wc/output part-00000   


## 3. SSH & Server
### 3.1 ssh connection without entering password
* eval \`ssh-agent\`  
* ssh-add 

### 3.2 Add SSH key to the server, so as to skip password
* ssh-copy-id -i .ssh/id_rsa.pub username@192.168.x.xxx

### 3.3 Remote server connection
* ssh zzu8431@wolf.analytics.private
* ssh zzu8431@msia423.analytics.northwestern.edu
* ssh -T git@github.com

### 3.4 SCP - secure copy
* scp [OPTION] [user@]Local host: file1 [user@]Remote host: file2

### 3.5 Keep codes running on the server
* nohup python -u model.py > log.out 2>&1 & # log.out is the log file

### 3.6 Additional tutor
* <https://meineke.github.io/workflows/>   # TA Session
* <https://blog.csdn.net/xlgen157387/article/details/50282483>   # Server Config (Renaming)
* <https://blog.csdn.net/liu_qingbo/article/details/78383892>   # SSH key config


## 4. Git
### 4.1 basic 
* git init
* git remote add origin git@github.com:michaelliao/learngit.git
* git clone git@github.com:michaelliao/learngit.git
* git add readme.txt
* git commit -m "wrote a readme file" 

### 4.2 branch
* git checkout -b dev   # -b means create and switch to that branch (git branch dev; git checkout dev)
* git branch; git branch -vv  # check the branches/and upstream branches; 
* git push -u origin master; git push --set-upstream origin master: 如果当前分支与多个远程主机存在追踪关系, 那么git push --set-upstream origin master(git push -u origin master 省略形式)将本地的master分支推送到origin主机(--set-upstream选项会指定一个默认主机), 同时指定该主机为默认主机, 后面使用可以不加任何参数使用git push 

### 4.3 github
* git push origin master
* git pull origin master    
* git remote -v   # 查看远程仓库信息
* git remote rm origin   # 移除远程仓库信息

## 5. Docker
### 5.1 Build the image
* docker images   # 查看所有image
* docker build -f app/Dockerfile -t pennylane .   # (创建一个tag为pennylane)的镜像。最后的. 号，其实是在指定镜像构建过程中的上下文环境的目录    

### 5.2 Run the container  
* docker run -p 5000:5000 --name test pennylane   # (创建并run一个name为test的容器)    

### 5.3 Restart and kill the docker/container     
* docker restart 777c866f0cbc       
* docker start 777c866f0cbc   # 启动容器       
* docker attach 777c866f0cbc   # 进入容器(命令行模式，前提是用-it去run容器)    
* docker stop test  
* docker kill test    

### 5.4 Others 
* docker ps -a   # 查看过去执行过的docker     
* docker image/container rm name   # 删除镜像/容器
* docker rm $(docker ps -aq)   # 删除所有容器    
* docker run -不同参数 <https://blog.csdn.net/qq_19381989/article/details/102781663>


## 6. Conda
* conda/source activate xxx_env   # source适用于低版本, 退出用deactivate
* conda env list
* conda env create -f Env.yml   # create env from file
* conda create -n spark python=3.7
* conda remove -n spark --all
* conda list   # 查看env下的包
* python -m ipykernel install --user --name new_env # Add new env to the Jupyter kernal. This should be done in the activated new env.



