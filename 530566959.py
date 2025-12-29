import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

fp = 'Global YouTube Statistics.csv'
df = pd.read_csv(fp, encoding='ISO-8859-1')
df.head()

missingValues = df.isnull().sum()
dataTypes = df.dtypes
uniqueCategories = df['category'].unique()
uniqueCountries = df['Country'].unique()

print("missing value statistics:")
print(missingValues)
print("\n")
print("data type:")
print(dataTypes)
print("\n")
print("unique youtube channel category:")
print(uniqueCategories)
print("\n")
print("countries appearing in the data set:")
print(uniqueCountries)

df['Country'] = df['Country'].fillna('Unknown')
df['Abbreviation'] = df['Abbreviation'].fillna('Unknown')

df['video views'] = pd.to_numeric(df['video views'].str.replace('[^0-9.]', '', regex=True), errors='coerce')
df['uploads'] = pd.to_numeric(df['uploads'].str.replace('[^0-9]', '', regex=True), errors='coerce')

df.loc[df['Country'].isin(['Entertainment', 'Music', 'Film', 'Education', 'Howto']), 'Country'] = 'Unknown'

standardCategories = ['Music', 'Entertainment', 'Education', 'People & Blogs', 'Gaming', 'Sports', 'Comedy', 'Howto & Style', 'News & Politics', 'Film & Animation']
df.loc[~df['category'].isin(standardCategories), 'category'] = 'Unknown'

print(df.head(), df.isnull().sum(), df.dtypes)

cleaned_data_file_path = 'clear.csv'
df.to_csv(cleaned_data_file_path, index=False)

plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='category', order = df['category'].value_counts().index)
plt.title('youtube channel category distribution')
plt.xlabel('number of channels')
plt.ylabel('category')
plt.show()


top_countries_by_subscribers = df.groupby('Country')['subscribers'].mean().sort_values(ascending=False).head(10).index
plt.figure(figsize=(12, 6))
sns.barplot(data=df[df['Country'].isin(top_countries_by_subscribers)], x='subscribers', y='Country', estimator=sum, ci=None, order=top_countries_by_subscribers)

plt.title('total number of youtube channel subscribers in different countries')
plt.xlabel('Total number of subscribers')
plt.ylabel('Country')
plt.show()


df['created_year'] = pd.to_numeric(df['created_year'], errors='coerce')
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='created_year', y='subscribers', estimator='mean', ci=None)
plt.title('relationship between youtube channel creation year and average number of subscribers')
plt.xlabel('creation year')
plt.ylabel('average number of subscribers')
plt.xticks(rotation=45)
plt.show()



