import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('big4_financial_risk_compliance.csv')
df['Fraud_Level'] = (df['Fraud_Cases_Detected'] >
                     df['Fraud_Cases_Detected'].median()).astype(int)
le = LabelEncoder()
for col in ['Firm_Name','Industry_Affected','AI_Used_for_Auditing']:
    df[col] = le.fit_transform(df[col].astype(str))
features = ['Total_Audit_Engagements','High_Risk_Cases',
            'Compliance_Violations','Employee_Workload',
            'Audit_Effectiveness_Score','Firm_Name',
            'Industry_Affected','AI_Used_for_Auditing']
X = df[features];  y = df['Fraud_Level']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
for name, m in [('LR', LogisticRegression(max_iter=1000)),
                ('RF', RandomForestClassifier(100)),
                ('XGB', XGBClassifier(n_estimators=100))]:
    m.fit(X_tr, y_tr)
    print(name, accuracy_score(y_te, m.predict(X_te)))
# XGBoost 60% | RF 55% | LR 55%
