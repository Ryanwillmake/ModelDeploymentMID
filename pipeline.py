import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess(df):
    df = df.drop(columns=['student_id'])
    
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['extracurricular_activities'] = le.fit_transform(df['extracurricular_activities'])
    
    df['score_avg'] = (df['technical_skill_score'] + df['soft_skill_score']) / 2
    df['academic_avg'] = (df['ssc_percentage'] + df['hsc_percentage'] + df['degree_percentage']) / 3
    
    X = df.drop(columns=['placement_status', 'salary_package_lpa'])
    y_cls = df['placement_status']
    y_reg = df['salary_package_lpa']
    
    return X, y_cls, y_reg

cls_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])