#imoprt part
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv('Income_training.csv')

X=df.drop(['compositeHourlyWages'],axis=1)
y=df['compositeHourlyWages']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import nadam
from keras.optimizers import  SGD,Adadelta
import keras.optimizers
from keras.wrappers.scikit_learn import KerasRegressor


model=Sequential()
model.add(Dense(3,activation='selu'))
model.add(Dense(5,activation='selu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(11,activation='selu'))
model.add(Dense(5,activation='selu'))
#model.add(Dense(3,activation='selu'))
model.add(Dense(1,kernel_initializer='normal'))

nadam_mode=nadam(lr=0.01)
sgd=SGD( lr=0.001,momentum=0.3,decay=0.2)
adaDelta=Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adaDelta)

history=model.fit(x=X.values,y=y.values,batch_size=16,epochs=50,verbose=1)    

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

test=pd.read_csv('Income_testing.csv')
test=test.drop(['ID'],axis=1)
prediction=model.predict(test)

from sklearn.metrics import mean_squared_error

mse=mean_squared_error(y_test,predict)
print(sqrt(mse))

from keras.models import load_model

model.save('Income_prediction.hd5')

f = open("submission4.csv", "w")
f.write('ID,compositeHourlyWages\n')
for i in range(len(prediction)):
     f.write(str(i+1)+','+str(prediction[i][0])+'\n')
     
sub=pd.read_csv('submission4.csv')
    