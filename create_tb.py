import pandas as pd
persons = pd.read_csv('source/persons_primier.csv',sep=',')
diaries = pd.read_csv('source/diaries_premier.csv',sep=',')

dp = persons.merge(diaries, left_on='prozhito_id', right_on='person')

dp.to_csv('persons_daries.csv')