import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df=pd.read_csv("data/house_prices.csv")
X=df.drop("Price",axis=1); y=df["Price"]
cat=["Location"]; num=[c for c in X.columns if c not in cat]
prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),num),
("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
model=Pipeline([("preprocessor",prep),("regressor",RandomForestRegressor(n_estimators=250,random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(Xtr,ytr); pred=model.predict(Xte)
print("MAE:",mean_absolute_error(yte,pred)); print("R2:",r2_score(yte,pred))
joblib.dump(model,"model.pkl")
