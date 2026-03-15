import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_PATH = os.path.join(BASE_DIR, "data", "jobs.csv")


def ensure_outputs():
    """Ensure the outputs directory exists next to this script."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str | None = None):
    """
    Load jobs data for offline analysis.

    By default this reads from the local data/jobs.csv file
    relative to this script, so it works regardless of the
    current working directory.
    """
    csv_path = path or DATA_PATH
    df = pd.read_csv(csv_path)
    # Create a text field combining title + skills + location
    df['skills'] = df['Skills Required'].fillna("")
    df['title'] = df['Job Title'].fillna("")
    df['text'] = (df['title'] + " "+ df['skills']).str.replace(',', ' ')
    df['job_type'] = df['Job Type'].fillna('Unknown')
    return df


def plot_class_distribution(df, target_col='job_type'):
    counts = df[target_col].value_counts()
    plt.figure(figsize=(8,5))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Class distribution: %s' % target_col)
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(out)
    plt.close()
    print('Saved', out)


def train_and_analyze(df, text_col='text', target_col='job_type'):
    X = df[text_col].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
    X_train_t = vec.fit_transform(X_train)
    X_test_t = vec.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_t, y_train)

    # Predictions
    y_pred = clf.predict(X_test_t)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out_cm = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(out_cm)
    plt.close()
    print('Saved', out_cm)

    # Classification report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
    print('Saved classification_report.txt')

    # Feature importance: map feature importances back to terms using forest's feature_importances_
    try:
        importances = clf.feature_importances_
        feature_names = vec.get_feature_names_out()
        top_n = 30
        idx = np.argsort(importances)[::-1][:top_n]
        top_feats = feature_names[idx]
        top_imp = importances[idx]

        plt.figure(figsize=(8,10))
        sns.barplot(x=top_imp, y=top_feats, palette='magma')
        plt.title("Top TF-IDF Feature Importances (RandomForest)")
        plt.xlabel("Importance")
        plt.tight_layout()
        out_fi = os.path.join(OUTPUT_DIR, "feature_importance.png")
        plt.savefig(out_fi)
        plt.close()
        print('Saved', out_fi)
    except Exception as e:
        print('Could not compute feature importances:', e)


def main():
    ensure_outputs()
    df = load_data()
    plot_class_distribution(df)
    train_and_analyze(df)


if __name__ == '__main__':
    main()
