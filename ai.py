
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score
dataset=pd.read_csv("E:/1.Ai/Ai Lab/project2 Linear REgression prediction/dataset/datasal.csv")
print(dataset)
X=dataset.iloc[:,1].values
print(X)
Y=dataset.iloc[:,0:1].values
print(Y)
X_train,X_test,Y_train,Y_test=tts(X,Y,train_size=0.6,random_state=1)
X_train=X_train.reshape(-1,1)
print(X_train)
X_test=X_test.reshape(-1,1)
print(X_test)
model=LR()
model.fit(X_train,Y_train)

res=model.predict(X_test)
plt.scatter(X_train,Y_train,marker="*",color="grey")
plt.plot(X_test,res)
print(r2_score(Y_test,res))
plt.show()
