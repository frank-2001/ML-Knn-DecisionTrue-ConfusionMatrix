# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# %%
# Loading Dataset
data=pd.read_csv("Iris.csv")
# Drop index
data=data.drop("Id",axis=1)
# Set X columns
X=data.drop("Species",axis=1)
# Set Y column
y=data["Species"]

# %%
# Split data to 2 part Training and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# Training Knn Model
k = 5  # Define number of neight
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train, y_train)

# %%
# Training Decision tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# %%
# Test prediction
y_knn_pred = knn_model.predict(X_test)
print(y_knn_pred)
y_dt_pred = dt_model.predict(X_test)
print(y_dt_pred)

# %%
# Evaluation of the modele with the occuracy
accuracy_dt = accuracy_score(y_test, y_dt_pred)
accuracy_knn = accuracy_score(y_test, y_knn_pred)
print(f"Précision du modèle k-NN : {accuracy_knn:.2f}")
print(f"Précision du modèle Desion tree : {accuracy_dt:.2f}")

# %%
# Generate a matrix confusion
preds=knn_model.predict(X_test)
# Resume prediction
resume=[]
for e in preds:
    if((e in resume)==False):
       resume.append(e)
# Genrate confusion matrix
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test, preds)

# %%
# Design a matrix in 2D with Dataframe
df=pd.DataFrame(confusion,index=resume,columns=resume)
print(df)


