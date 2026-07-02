import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['Churn'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    sns.barplot(x=counts.index, y=counts.values, palette=colors, ax=ax)
    ax.set_title('Churn Distribution')
    ax.set_ylabel('Number of Customers')
    for i, v in enumerate(counts.values):
        ax.text(i, v+20, str(v), ha='center', fontweight='bold')
    return fig

def plot_tenure_by_churn(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='Churn', y='tenure', palette=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title('Tenure by Churn Status')
    return fig

def plot_monthly_charges_by_churn(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title('Monthly Charges by Churn Status')
    return fig

def plot_categorical_churn(df, col):
    """Plot churn rate by categorical column."""
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=churn_rate.index, y=churn_rate.values, palette='RdYlGn_r', ax=ax)
    ax.set_title(f'Churn Rate by {col}')
    ax.set_ylabel('Churn Rate (%)')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(churn_rate.values):
        ax.text(i, v+0.5, f'{v:.1f}%', ha='center')
    return fig

def plot_contract_tenure_heatmap(df):
    # Create tenure segments
    bins = [0, 12, 24, 48, 72, 100]
    labels = ['0-12', '13-24', '25-48', '49-72', '72+']
    df_temp = df.copy()
    df_temp['tenure_segment'] = pd.cut(df_temp['tenure'], bins=bins, labels=labels, right=False)
    pivot = pd.crosstab(df_temp['Contract'], df_temp['tenure_segment'],
                        values=df_temp['Churn'], aggfunc=lambda x: (x == 'Yes').mean() * 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('Churn Rate (%) by Contract Type and Tenure Segment')
    return fig