
# coding: utf-8




import pandas as pd
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model, Sequential
from keras import optimizers
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from keras import regularizers
#from sklearn.preprocessing import StandardScaler




train = pd.read_csv('./data/train_100k.csv')
out = pd.read_csv('./data/train_100k.truth.csv')
test = pd.read_csv('./data/test_100k.csv')
train.head()





train.isnull().values.any()
out.isnull().values.any()
test.isnull().values.any()





train = train.drop(columns='id', axis=1)
out = out.drop(columns='id', axis=1)
test = test.drop(columns='id', axis=1)
#train.head()
#test.head()
#out.head()





#scaler = StandardScaler().fit(train)
#train = scaler.transform(train)
#train = pd.DataFrame(train)
#train.skew()
#train.kurtosis()
#train.head()


slope = out.iloc[:,0]
intercept = out.iloc[:,1]


input_layer = Input(shape = (20,))
layer1 = Dense(units=1000, activation = 'relu',kernel_regularizer=regularizers.l2(1))(input_layer)
layer4 = Dense(units=1000, activation = 'relu',kernel_regularizer=regularizers.l2(1))(layer1)
#layer3 = Dense(units=100, activation = 'relu')(layer2)
#layer3 = Dropout(0.2)(layer2) 
#layer4 = Dense(units=1000, activation = 'relu')(layer3)
#layer4 = Dropout(0.2)(layer4) 

output1 = Dense(units = 1, activation = 'linear')(layer4)
output2 = Dense(units = 1, activation = 'linear')(layer4)
               
regressor = Model(inputs = input_layer,
                  outputs = [output1, output2])

opt = optimizers.Adam(lr=0.001, decay = 0.00001)               
regressor.compile(optimizer = 'adam',
                  loss = 'mse', metrics=['mae', 'mse'])
               
fit = regressor.fit(train, [slope, intercept],
              epochs =1000, batch_size = 1000, verbose=2, validation_split=0.3)
               
pred_slope, pred_intercept = regressor.predict(train)
               



result = pd.DataFrame(pred_slope, columns=['slope'])
result.insert(1, column='intercept', value=pred_intercept)
result.index.name = 'id'
result.to_csv('./results/submission_train.csv') 
result.head()


pred_slope, pred_intercept = regressor.predict(test)

result = pd.DataFrame(pred_slope, columns=['slope'])
result.insert(1, column='intercept', value=pred_intercept)  
result.index.name = 'id'
result.to_csv('./results/submission_test.csv') 
result.head()


#hist = pd.DataFrame(fit.history)
#hist.to_csv('./results/history.csv')    #history loss data

try: 

    '''   Loss Charts    '''

    print("Teste de validação")
    #print(o.history['val_acc'])
    a = list(fit.history.keys())
    
    val_acc = fit.history[a[4]]
    acc = fit.history[a[5]]

    #loss = o.history['loss']
    #val_loss = o.history['val_loss']
    #sns.set()
    plt.figure(1)
    #set_figheight(200) # optional setting the height of the image
    #set_figwidth(200)

    #plt.subplot(121)
    plt.plot(val_acc, label='MSE Slope Teste',color='red')
    plt.legend()
    plt.figure(2)
    plt.plot(acc, label=' MAE Intercept Teste ',color='green')
    plt.legend()
    
    plt.figure(3)
    #set_figheight(200) # optional setting the height of the image
    #set_figwidth(200)
    val_acc2 = fit.history[a[11]]
    acc2 = fit.history[a[12]]

    #plt.subplot(121)
    plt.plot(val_acc2, label='MSE Slope ',color='red')
    plt.legend()
    plt.figure(4)
    plt.plot(acc2, label=' MAE Intercept ',color='green')
    plt.legend()





    plt.show()
    
except:
    
    pass







