import pandas as pd

df = pd.read_csv('YOUR .CSV FILE')
df_cleaned = df.dropna()
df_cleaned.to_csv('cleaned_file.csv', index=False)

print("Lines with missing data have been removed and saved in 'cleaned_file.csv'.")
