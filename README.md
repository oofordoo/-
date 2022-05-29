def loss_function(X, y, weight, bias): # 手工定义一个MSE均方误差函数
y_hat = weight*X + bias # 这是假设函数,其中已经应用了Python的广播功能
loss = y_hat-y  # 求出每一个y’和训练集中真实的y之间的差异 
cost = np.sum(loss**2)/(2*len(X)) # 这是均方误差函数的代码实现
return cost # 返回当前模型的均方误差值
