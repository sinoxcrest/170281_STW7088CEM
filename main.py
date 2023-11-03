
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df=pd.read_csv('dataset_thyroid_sick.csv')
df.head()
df.info()
df = df.replace('?', np.nan)
df.info()
df=df.drop('TBG',axis=1)
df=df.dropna()
df.info()
df[['TSH','T3','TT4','T4U','FTI']]=df[['TSH','T3','TT4','T4U','FTI']].astype(float)
df['age']=df['age'].astype(int)
df1=df.drop('Class',axis=1)
df1=pd.get_dummies(df1)
df1['Class']=df['Class']
df1=df1.replace({'negative':0,'sick':1})
df1=df1.dropna()
df1.head()
print(df1.head())
df1.describe().T
q = df1.age.quantile(0.995)
print(q)
df1=df1.query('age < @q')
df1.describe().T
q2 = df1.TSH.quantile(0.95)
print(q2)
df1=df1.query('TSH < @q2')
df1.describe().T


import plotly.graph_objects as go
import matplotlib.pyplot as plt



from pycaret.classification import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns


labels = ['0=Negative','1=Sick']
values = df1['Class'].value_counts()
colors = ['royalblue','red']
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5)])
fig.update_traces(hoverinfo='label+value',textfont_size=15,marker=dict(colors=colors))
fig.update_layout(annotations=[dict(text='Thyroid Outcome',
                                    x=0.50, y=0.5, font_size=15,
                                    showarrow=False)])
fig.show()
plt.plot(df1)
plt.show()


plt.figure(figsize =(20,10))
sns.heatmap(df1.corr(),annot=True,fmt='.1f')
plt.show()

df_train, df_test = train_test_split(df1, random_state =100 , test_size = 0.3)
setup_df = setup(data= df_train, target = 'Class',
                session_id=100, data_split_stratify=True,
                  verbose=False,remove_outliers=True)


top3_models = compare_models(n_select=3)
top3_models
ct = create_model("catboost")
# plot_model(estimator = ct , plot= "learning")
# plot_model(estimator = ct , plot= "auc")
# plot_model(estimator = ct, plot= "confusion_matrix", plot_kwargs = {'percent' : True})
# plot_model(estimator = ct, plot = "error")
# plot_model(estimator = ct, plot = "boundary")
predict_model(ct);
pred = predict_model(ct, data=df_test)
pred.head()
print("HELLO")
print(pred.head())
accuracy_score(pred['Class'], pred['prediction_label'])

from sklearn.metrics import classification_report


import matplotlib.pyplot as plt

X=df1.drop('Class',axis=1)
y=df1['Class']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 47))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100,verbose = 0)

score, acc = classifier.evaluate(X_train, y_train,
                            batch_size=10)
print('Train score:', score)
print('Train accuracy:', acc)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print('*'*20)
score, acc = classifier.evaluate(X_test, y_test,
                            batch_size=10)
print('FNN Test score:', score)
print('FNN Test accuracy:', acc)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
p = sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt="g")
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_curve
y_pred_proba = classifier.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='ANN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.show()
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)

from dataprep.eda import *
plot(df1).show_browser()

#LSTM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Prepare your data and split it into training and test sets
X = df1.drop('Class', axis=1).values
y = to_categorical(df1['Class'])  # Convert your labels to one-hot encoded format
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize your data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define your LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(2, activation='softmax'))  # Two output classes ('negative' and 'sick')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape the data for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("LSTM Test Loss:", loss)
print("LSTM Test Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Assuming you have already trained your LSTM model and have predictions stored in y_pred
y_pred = model.predict(X_test)

# Convert predicted probabilities to class labels
y_pred_labels = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 1. Confusion Matrix and Visualization (Optional)
confusion = confusion_matrix(y_true, y_pred_labels)

# 2. Classification Report (Optional)
class_report = classification_report(y_true, y_pred_labels)

# 3. Overall Accuracy
accuracy = accuracy_score(y_true, y_pred_labels)

# Print the results
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", class_report)
print("Overall Accuracy:", accuracy)






