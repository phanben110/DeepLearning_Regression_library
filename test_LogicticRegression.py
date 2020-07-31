"""
Name: Phan Ben 
Class: 18TDH1 _ DUT 
Email: phanben110@gmail.com
"""
import OPP_algorithmRegression as pb
import pandas as pd

data = pd.read_csv('D:\work\Mechine Learing\DL_Tutorial-master\L2\dataset.csv').values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)
myDeThuong = pb.LogicticRegression(x,y)
myDeThuong.drawData("True" , "False")
myDeThuong.run(1000 ,0.01)
print(myDeThuong.predicValue([2,1]))
myDeThuong.show()
