from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn
df = pd.read_csv(r'C:\Users\Admin\Desktop\ML\diabetes_risk_dataset.csv')
df.head()
x = df.drop("diabetes_risk_category", axis=1)  # FEATURES
y = df["diabetes_risk_category"]  # TARGET
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
x_train
scaler = MinMaxScaler()  # StandardScaler() or MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train
# 4. Apply PCA
pca = "PCA"(n_components=6)
# FIT PCA on Training Data ONLY
# TRANSFORM both Training and Test Data
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(x_train.shape)
lr = LinearRegression()
model1 = lr.fit(x_train, y_train)
y_pred = model1.predict(x_test)
# Convert to binary (0 or 1) using threshold 0.5
y_pred = (y_pred >= 0.5).astype(int)
print(r2_score(y_test, y_pred)**0.5)
print(mean_squared_error(y_test, y_pred)**0.5)
model2 = LogisticRegression(max_iter=1000)
model2.fit(x_train, y_train)
y_pred = model2.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model3.fit(x_train, y_train)
y_pred = model3.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
model4 = svm.SVC(kernel='rbf', C=3.0)
model4.fit(x_train, y_train)
y_pred = model4.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
svm_cl = svm.SVC(kernel='rbf', C=3.0, probability=True)
rf_cl = RandomForestClassifier(n_estimators=100, random_state=42)
lr_cl = LogisticRegression(max_iter=50)
voting_clf = VotingClassifier(
    estimators=[('svm', svm_cl), ('rf', rf_cl), ('lr', lr_cl)],
    voting='soft'  # 'hard' for majority vote, 'soft' for probability-weighted
)
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# ===== Option 2: Stacking Classifier =====
stacking_clf = StackingClassifier(
    estimators=[('svm', svm_cl), ('rf', rf_cl), ('lr', lr_cl)],
    final_estimator=LogisticRegression(),  # Meta-learner
    cv=5
)
stacking_clf.fit(x_train, y_train)
y_pred = stacking_clf.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Adaboost in ensemble learning
# adaboost is a combination of multiple decision trees
# adaboost is a type of boosting
# you can use adaboost to reduce overfitting
# you can use adaboost to increase accuracy
# you can use adaboost on both classification and regression
# importing the Adaboost algoritm


# ada_model = AdaBoostClassifier()

dt_model = DecisionTreeClassifier()
ada_model = AdaBoostClassifier(estimator=dt_model, n_estimators=60)
# estimator is the model which you want to use for boosting
# model training

ada_model.fit(x_train, y_train)
# model prediction

ada_pred = ada_model.predict(x_test)

# model eveluation

print(accuracy_score(y_test, ada_pred)*100)
print(confusion_matrix(y_test, ada_pred))
print(classification_report(y_test, ada_pred))
# bagging in ensemble learning
# bagging is a combination of multiple decision trees
# randomforest is a type of bagging
# you can use bagging to reduce overfitting
# you can use bagging to increase accuracy
# you can use bagging on both classification and regression
# initializing a randomforest model


rf_model = RandomForestClassifier(n_estimators=22, random_state=75)
# model training

rf_model.fit(x_train, y_train)
# model prediction

rf_pred = rf_model.predict(x_test)

# model evaluation

print(accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
