# datastax toomanyindexes error
import pandas as pd

df_csv = pd.read_csv('planets_parsed.csv')
df = df_csv.drop('rotation_period', axis=1)
df = df_csv.drop('orbital_period', axis=1)
df_csv.to_csv('planets_parsed_fewer_cols.csv', index = False)