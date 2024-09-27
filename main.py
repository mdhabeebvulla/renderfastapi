from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
iris = load_iris()
X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr.fit(X_train,y_train)
app = FastAPI()
@app.get("/")
def example(sl,sw,pl,pw):
    result = lr.predict([[sl,sw,pl,pw]])
    return{'message':int(result[0])}
