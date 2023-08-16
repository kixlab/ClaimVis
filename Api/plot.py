import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# plan -

data = pd.read_csv('../Datasets/housing.csv')

def plot(data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='housing_median_age', ax=ax, kde=True, color='blue', alpha=0.5)
    ax.axvline(x=data['housing_median_age'].median(), color='red', linestyle='--', label=f'Median: {data["housing_median_age"].median():,.2f}')
    ax.set_xlabel('Housing Median Age')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.title('What is the distribution of housing_median_age?', wrap=True)
    return plt

chart = plot(data)
chart.show()