"""
Name: Phan Ben 
Class: 18TDH1 _ DUT 
Email: phanben110@gmail.com
"""
import OPP_algorithmRegression as pb
import pandas as pd

data = pd.read_csv('D:\work\Mechine Learing\simpleLinear.csv').values
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
benDeptrai = pb.LinearRegression( x , y )
benDeptrai.drawData("ben dep trai " ,"my dep gai ")
benDeptrai.run(100 ,  0.000000001)
print(benDeptrai.__doc__)
print(benDeptrai.predicValue(2800))

benDeptrai.show()
