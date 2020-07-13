# -*- coding: utf-8 -*-

import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/train.csv')

# import sweetviz as sv
# advert_report = sv.analyze(df)
# advert_report.show_html('EDA.html')

import pandas_profiling as pdp
profile_train_df = pdp.ProfileReport(df, title="Pandas Profiling Report", explorative=True)

profile_train_df.to_file("your_report.html")
# sns.violinplot(x='SmokingStatus', y='Age', data=df, hue="Sex", palette='muted', split=True)
# plt.title('Age Distributions Across Smoking Groups', size=16)
# plt.legend(loc='lower left', ncol=2)

# sns.pointplot(x='Weeks', y='FVC', data=df[df['Patient']=='ID00007637202177411956430'])