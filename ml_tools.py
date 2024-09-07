import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def get_data():
    """Load and preprocess the data."""
    df = pd.read_csv('feature_set.csv').fillna(method='pad')
    with open("negative.lncRNA.glist.xls", "r") as f:
        raw_negative_list = list(map(lambda x: x.strip(), f.readlines()))

    with open("positive.lncRNA.glist.xls", "r") as f:
        raw_positive_list = list(map(lambda x: x.strip(), f.readlines()))

    if len(raw_negative_list) < len(raw_positive_list):
        positive_list = list(np.random.choice(raw_positive_list, 150))
        negative_list = raw_negative_list
    else:
        negative_list = list(np.random.choice(raw_negative_list, 150))
        positive_list = raw_positive_list
    
    positive_df = df.loc[df['Gene_ID'].isin(positive_list)]
    negative_df = df.loc[df['Gene_ID'].isin(negative_list)]
    positive_df['label'] = 1
    negative_df['label'] = -1

    new_df = pd.concat([positive_df, negative_df])
    cols = list(new_df.columns)
    new_cols = cols[:-1]
    new_cols.insert(0, 'label')
    new_df = new_df.loc[:, new_cols]

    # Feature selection
    new_df = feature_select(new_df)
    y = np.array(new_df.label)
    X = new_df.iloc[:, 1:]
    return X, y

def feature_select(df):
    """Select important features using ExtraTreesClassifier."""
    from sklearn.ensemble import ExtraTreesClassifier
    y = np.array(df.label)
    X = np.array(df.drop(columns=['Gene_ID']).values)

    forest = ExtraTreesClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    best_features = ['label']
    for f in range(X.shape[1]):
        best_features.append(df.columns[indices[f] + 1])

    n_best = len(best_features)
    df1 = df.loc[:, best_features[0:n_best]]  # 0 = label
    return df1

def feature_importance_display(clf, model_id, X):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    
    if model_id == 'NB':
        positivelog = clf.feature_log_prob_[0, :]
        positive = [math.exp(x) for x in positivelog]
        negativelog = clf.feature_log_prob_[1, :]
        negative = [math.exp(x) for x in negativelog]  # Fix this line
        importances = pd.DataFrame(
            {
                'positive_odds': positive,
                'negative_odds': negative
            },
            index=X.columns
        ).sort_values('positive_odds', ascending=False)
    elif model_id == 'LR':
        importances = pd.Series([abs(x) for x in clf.coef_[0]], index=X.columns).sort_values(ascending=True)
    elif model_id == 'RF':
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
    else:
        raise NotImplementedError(f'The feature importances module is not supporting model_id="{model_id}"..')
    
    importances.plot(kind='bar', ax=ax)
    ax.set_title('Feature Importances')
    
    return fig


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def run_ML_pipeline(report, file_path, model_id):
    """
    Runs a certain pipeline on the given training data.
    """
    # Load the data
    user_x = pd.read_csv(file_path, sep=',' if file_path.endswith('.csv') else '\t', index_col=0).fillna(method='pad')
    test_x = user_x.iloc[:, 1:]
    
    seed = 42
    np.random.seed(seed)

    model_id = model_id
    train_X, train_y = get_data()
    column_names = list(test_x.columns.values)
    train_X = train_X[column_names]

    # Model selection
    if model_id == 'NB':
        from sklearn.naive_bayes import BernoulliNB
        clf = BernoulliNB(alpha=5)
    elif model_id == 'kNN':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=3)
    elif model_id == 'LR':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=10, random_state=seed, n_jobs=-1)
    elif model_id == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None,
                                    min_samples_split=2, random_state=seed, n_jobs=-1)
    elif model_id == 'SVM':
        from sklearn.svm import SVC
        clf = SVC(kernel='rbf', probability=True, random_state=seed)
    else:
        raise NotImplementedError(f'The model_id={model_id} is not known!')

    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_x)

    if report == 'confusion_matrix':
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(clf, train_X, train_y, ax=ax)
        return fig

    elif report == 'roc_auc_curve':
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(clf, train_X, train_y, ax=ax)
        return fig

    elif report == 'feature_importance':
        return feature_importance_display(clf, model_id, train_X)
    
    elif report == 'prediction_result':
        user_x['predicted_value'] = pred_y
        return user_x[['Gene_ID', 'predicted_value']]
    
    elif report == 'pca_plot':
        return run_PCA(test_x, pred_y)
    
    else:
        raise NotImplementedError(f'The report={report} is not known!')


def run_PCA(user_x, pred_y):
    """Run PCA on the given data"""
    seed = 42
    np.random.seed(seed)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(user_x)
    pca = PCA().fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    f, ax = plt.subplots(figsize=(15, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=pred_y)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    legend1 = ax.legend(*scatter.legend_elements(), title="Label")
    ax.add_artist(legend1)

    return ax
