import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def plot_pca(df):
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                        'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI','SpreadOC','SpreadLH']

    X = df[feature_cols]
    X= X[50:]
    # Standardize the data
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    # Fit PCA without specifying the number of components
    pca = PCA()
    pca.fit(X_standardized)

    # Plot the explained variance ratio
    plt.figure(figsize=(10, 5))

    # Scree plot
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot')

    # Cumulative explained variance plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Plot')

    plt.tight_layout()
   # plt.show()
    plt.savefig('explained_variance_plot.png')