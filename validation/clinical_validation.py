# Validation against patient data
import pandas as pd

def validate():
    df = pd.read_csv('patients.csv')
    print(df.head())