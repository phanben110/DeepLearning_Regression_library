import numpy as np
import matplotlib.pyplot as plt

"""
This library was researched and developed by Phan Ben, with the aim of conquering AI
Basic deep learning Algorithm
31/7/2020 

"""

class LinearRegression:
	"""
	This library was researched and developed by Phan Ben, with the aim of conquering AI
	Algorithm Linear Regression is basic deep learning Algorithm

	"""
	w = 0
	def __init__(self , _dataX , _dataY ):
		self.x = _dataX.reshape(-1,1)
		self.y = _dataY.reshape(-1,1)
		self.N = self.x.shape[0]

	def drawData(self, nameLableX , nameLableY):
		plt.scatter(self.x , self.y )
		plt.xlabel(nameLableX)
		plt.ylabel(nameLableY)
		# plt.show()

	def run(self , numOfIteration , learningRate ):
		X = np.hstack((np.ones((self.N ,1 )) , self.x ))
		W = np.array([0.,1.]).reshape(-1,1)
		Y = self.y 
		cost = np.zeros((numOfIteration , 1))
		for i in range(1, numOfIteration):
			r = np.dot(X ,W ) - Y
			cost[i] = 0.5*np.sum(r*r)
			W[0] -= learningRate*np.sum(r)
			W[1] -= learningRate* np.sum(np.multiply(r, X[:, 1].reshape(-1, 1)))
			print (cost)
		predict = np.dot(X , W)
		plt.plot((X[0][1] , X[self.N - 1][1]) , (predict[0] ,predict[self.N -1 ]), 'r')
		LinearRegression.w = W
		return W

	def predicValue(self , ValueX  ):
		W = LinearRegression.w
		y = W[0] + W[1]*ValueX
		return y

	def show(self):
		plt.show()


class LogicticRegression:
	"""
	This library was researched and developed by Phan Ben, with the aim of conquering AI
	Algorithm Logictic Regression is basic deep learning Algorithm

	"""
	W = 0
	probabilityTrue = 0
	probabilityFalse = 0


	def __init__(self , x_data , y_data ):
		self.x  = x_data.reshape(-1,2)
		self.y  = y_data.reshape(-1,1)
		self.N  = self.y.shape[0]

	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	def drawData(self, _xTrue, _xFalse ):
		xTrue = self.x[self.y[:,0]==1]
		xFalse= self.x[self.y[:,0]==0]
		plt.scatter(xTrue[:,0]  , xTrue[:,1]  , c='red', edgecolors='none', s=30, label=_xTrue )
		plt.scatter(xFalse[:,0] , xFalse[:,1] , c='blue', edgecolors='none', s=30, label=_xFalse)
		plt.legend(loc = 1 )
		plt.xlabel("x1")
		plt.xlabel("x2")

	def run(self, numOfIteration , learningRate ):
		x = np.hstack((np.ones((self.N, 1)), self.x))
		w = np.array([0., 0.1, 0.1]).reshape(-1, 1)
		y = self.y
		cost = np.zeros((numOfIteration, 1))
		for i in range (1, numOfIteration):
			y_predict = LogicticRegression.sigmoid(np.dot(x, w))
			cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1 - y, np.log(1 - y_predict)))
			# Gradient descent
			w = w - learningRate * np.dot(x.T, y_predict - y)
			print(cost[i])
		t = 0.5
		plt.plot((4, 10),(-(w[0] + 4 * w[1] + np.log(1 / t - 1)) / w[2], -(w[0] + 10 * w[1] + np.log(1 / t - 1)) / w[2]), 'g')
		# giải thích (4,10) là lấy giá trị x từ ngưỡng từ 4 đến 10
		LogicticRegression.W = w
		return w

	def predicValue(self, ValueX):
		w = LogicticRegression.W
		y = LogicticRegression.sigmoid(w[0]+w[1]*ValueX[0] + w[2]*ValueX[1])
		if y >= 0.5 :
			a = LogicticRegression.probabilityTrue = (y - 0.5)*2
			print("probability True is {} % ".format(float(a*100)))
			LogicticRegression.probabilityFalse = 0
			return True
		else:
			a = LogicticRegression.probabilityFalse = (0.5-y) * 2
			print("probability False is {} % ".format(float(a*100)))
			LogicticRegression.probabilityTrue = 0
			return False

	def show(self):
		plt.show()









