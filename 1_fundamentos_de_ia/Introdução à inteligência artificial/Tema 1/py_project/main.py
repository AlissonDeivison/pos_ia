from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

#Carregamento do dataset
def main() -> None:
    # Caminho do arquivo CSV dentro do diretório `data/`
    csv_path = "data/creditcard.csv"

    # Carrega o dataset
    df = read_csv(csv_path)

    #70% dos dados para treinamento e 30% para teste
    train_size = 0.7
    
    #Dividindo os dados em treinamento e teste
    train_df = df.sample(frac=train_size, random_state=42)
    test_df = df.drop(train_df.index)

    #Usando um random forest para classificar os dados
    # Define features (X) e alvo (y)
    target_col = "Class"
    assert target_col in df.columns, "Coluna 'Class' não encontrada no dataset."

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Treinamento do RandomForest com compensação de desbalanceamento
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    clf.fit(X_train, y_train)

    # Avaliação
    y_pred = clf.predict(X_test)
    # Probabilidade da classe positiva (1) para AUC
    y_proba = None
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    print("\nResultados do RandomForest:")
    print(f"Acurácia: {acc:.4f}")
    if auc is not None:
        print(f"ROC AUC: {auc:.4f}")
    print("Matriz de confusão:\n", cm)
    print("\nRelatório de classificação:\n", report)

if __name__ == "__main__":
    main()
