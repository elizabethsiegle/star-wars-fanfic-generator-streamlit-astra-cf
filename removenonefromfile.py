import pandas as pd

df_csv = pd.read_csv('planets.csv')
df = df_csv.fillna(0)
df_csv.to_csv('planets_parsed.csv', index = False)