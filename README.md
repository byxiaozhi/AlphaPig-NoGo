# AlphaPig
一个基于神经网络的不围棋AI

## 运行环境

[Visual Studio 2022](https://visualstudio.microsoft.com/zh-hans/) 

[oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html)

[Python](https://www.python.org/)

[Keras](https://keras.io/)

## 使用说明

1. 运行 Train.py 生成模型（需要得到model文件夹以及network.weights）
2. 编译 SelfPlay 程序（得到SelfPlay.exe）
3. 新建一个文件夹，里面新建一个dataset目录，然后将Train.py、Selfplay.exe、Trainloop.ps1、network.weights文件以及model文件夹放进去
4. 运行Trainloop.ps1开始训练

