from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

digits = load_digits()

print(f'Image : {digits.data.shape}')
print(f'Label : {digits.target.shape}')

print(digits.data[0].shape)
# print(digits.data[0])

# df = pd.DataFrame(digits.data, columns=digits.feature_names)
# df['result'] = pd.Categorical.from_codes(digits.target, digits.target_names)
# print(df.head())

# index = 20
# img = digits.data[index]
# label = digits.target[index]
# img = np.reshape(img, (8, 8))
# plt.imshow(img, cmap=plt.cm.gray)
# plt.title(f'Label : {label}')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
print(x_train.shape)
print(x_test.shape)

model = LogisticRegression()
model.fit(x_train, y_train)

index = 20
img = x_train[index][:]
label = y_train[index]
img = np.reshape(img, (1, -1))
# print(img)
print(f'Predict : {model.predict(img)}')
print(f'Label : {label}')

score = model.score(x_test, y_test)
print(score)
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt='.0f', linewidths=5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predict label')
plt.title(f'Score : {score}', size=15)
plt.show()