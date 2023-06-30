import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mp

df = pd.read_csv('train.csv', header=0)

#Display the count,mean,std,min,max and various percentiles for the variables that describe our monster types.
df.describe()

###After examining the following countplot it appears that the value 'Color' has little significance
#when it comes to classifying the types of our monsters.
countplot = sns.countplot(y='color', hue='type', data=df) #Figure 7
fig = countplot.get_figure()
fig.savefig('color_countplot.png')

df = df.drop(["color"],axis='columns')


#This plot shows the correlation between pairs of the variables:
#'bone_length, rotting_flesh,hair_length,has_soul'
#It helps us see the trends in our data and it gives us an idea about which monsters are dependently classified easier
#by which variables.
sns.pairplot(df.drop('id', axis=1), hue='type')
mp.savefig('pairplot.png')

#The following creates  a boxplot for our various variables:
#'bone_length, rotting_flesh,hair_length,has_soul & color'
data = df.drop(['id','type'], axis=1)

fig, axes = mp.subplots(nrows=2, ncols=2, figsize=(6,6))
mp.tight_layout(w_pad=2.0, h_pad=2.0)

for i, column in zip(range(1,data.shape[1]+1), data.columns):
    mp.subplot(2,2,i)
    sns.boxplot(x=df['type'], y=df[column], linewidth=0.5)
    mp.ylabel(column)
    mp.ylim([0,1.1])
mp.savefig('boxplot.png')

