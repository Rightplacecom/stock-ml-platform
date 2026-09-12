#import pandas as pd

#df = pd.read_csv("train.csv")

#print("Columns:", df.columns.tolist())
#print("Rows:", len(df))
#print()
#print(df)
import pandas as pd

# 1. Read CSV
df = pd.read_csv("train.csv")

# 2. Remove empty columns (from double commas)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = [c.strip() for c in df.columns]

# 3. Clean "+" from points and make it number
pts = 'TOTAL POINTS UP/DOWN'
df[pts] = pd.to_numeric(df[pts].astype(str).str.replace('+', ''), errors='coerce')

# 4. Show rows and columns
print("Columns:", df.columns.tolist())
print("Rows:", len(df))
print()
print(df)
print()

# 5. Simple rule model: put more -> BUY CALL, call more -> BUY PUT
df['PREDICT'] = ['BUY CALL' if p > c else 'BUY PUT'
                 for c, p in zip(df['call_volume'], df['put_volume'])]
df['CORRECT'] = df['PREDICT'] == df['WINNER'].str.strip()

print("Rule accuracy:", round(df['CORRECT'].mean() * 100, 1), "%")