import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot(data: pd.DataFrame):
    # solution
    plt.figure(figsize=(12, 6))
    if len(data) > 30:
        chart = sns.barplot(x='title', y='imdb_rating', data=data, color='blue')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        chart.set(title='IMDB rating of each movie', xlabel='Movie Title', ylabel='IMDB Rating')
        chart.set(ylim=(0, 10))
        chart.set(xlim=(-1, len(data)))
        chart.set_xticklabels(chart.get_xticklabels(), fontsize=8)
        chart.tick_params(axis='y', labelsize=8)
        chart.tick_params(axis='x', labelsize=8)
        chart.set(xlabel=None)
        chart.set(ylabel=None)
        chart.set(xticks=[])
        chart.set(yticks=[])
        chart.set_frame_on(False)
        chart.spines['top'].set_visible(False)
        chart.spines['right'].set_visible(False)
        chart.spines['bottom'].set_visible(False)
        chart.spines['left'].set_visible(False)
        chart.grid(True, axis='y', linestyle='--', alpha=0.7)
    else:
        chart = sns.barplot(x='title', y='imdb_rating', data=data, color='blue')
        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
        chart.set(title='IMDB rating of each movie', xlabel='Movie Title', ylabel='IMDB Rating')
        chart.set(ylim=(0, 10))
        chart.set(xlim=(-1, len(data)))
        chart.set_xticklabels(chart.get_xticklabels(), fontsize=8)
        chart.tick_params(axis='y', labelsize=8)
        chart.tick_params(axis='x', labelsize=8)
        chart.set(xlabel=None)
        chart.set(ylabel=None)
        chart.set(xticks=[])
        chart.set(yticks=[])
        chart.set_frame_on(False)
        chart.spines['top'].set_visible(False)
        chart.spines['right'].set_visible(False)
        chart.spines['bottom'].set_visible(False)
        chart.spines['left'].set_visible(False)
        chart.grid(True, axis='y', linestyle='--', alpha=0.7)
        for index, row in data.iterrows():
            chart.text(index, row.imdb_rating, round(row.imdb_rating, 2), color='black', ha="center", fontsize=8)

    return plt;

chart = plot(pd.read_csv('sub_table.csv'))
chart.show()