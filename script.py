do_analise_preliminar = False
do_knn = False
do_dt = False
do_mlp = True

import numpy
import pandas as pd
import numpy as np
from sklearn import neighbors, tree
from sklearn.datasets import load_svmlight_file

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# Inicialização de parametros iguais para todos os Comites de Classificadores
seed = 10
num_trees = 21
max_features = 9

print('### Leitura dos Dados')

list_features = []
list_targets = []
for i in range(1, 11):
    X, y = load_svmlight_file(f='./Dataset/batch' + str(i) + '.dat', dtype=np.float64)
    X = pd.DataFrame(X.toarray())
    y = pd.Series(y)

    list_features.append(X)
    list_targets.append(y)

X = pd.concat(list_features, ignore_index=True)
y = pd.concat(list_targets, ignore_index=True)

# Todoo código dentro deste IF serve para análise dos dados
if do_analise_preliminar:
    print(X.shape)
    print(X.head())
    print(y)

    print('### Análise dos Dados')

    print(X.describe())

    print('### Distribuição das médias')

    X.mean().hist(bins=30)
    plt.show()

    print('### Médias das features')

    X.mean().plot(kind='bar', figsize=(18, 12))
    plt.show()

    print('### Correlação das features')

    correlation = X.corr()
    print(correlation.head())

    fig, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(correlation, cmap='seismic')
    plt.show()

    # ### Apenas correlações altas (acima de 0.7)

    high_correlation = correlation.applymap(lambda x: x if x > 0.85 or x < -0.85 else 0)

    fig, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(high_correlation, cmap='seismic')  # 'Blues'
    plt.show()

    print('### Série temporal')

    X[4].plot(figsize=(14, 8))
    plt.show()

    print('### Série temporal apenas para uma classe')

    X_4 = X[4].copy()
    X_4.loc[~X.index.isin(y[y == 1].index)] = 0
    X_4.plot(figsize=(14, 8))
    plt.show()

    print('### Análise de uma características nos 16 sensores')

    y.name = 'target'
    data = pd.concat([X, y], axis=1)

    cols_to_plot = [i for i in range(0, 128, 8)]
    cols_to_plot.append('target')
    print(cols_to_plot)

    sns.pairplot(data[cols_to_plot], hue='target')
    plt.show()

    print('### Outliers')

    sns.boxplot(X[0], orient='vertical')
    plt.show()

    cols_to_plot = [0, 5, 15, 50, 73, 100, 115, 127]
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))

    count = 0
    for i in range(2):
        for j in range(4):
            ax[i][j].boxplot(X[cols_to_plot[count]])
            ax[i][j].set_xticks([])
            ax[i][j].set_xlabel(cols_to_plot[count])
            count += 1

    plt.show()

    sns.boxplot(X[0], orient='vertical')
    plt.show()

    def count_outliers(column):
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)
        iqr = q3 - q1
        return column[(column < (q1 - 1.5 * iqr)) | (column > (q3 + 1.5 * iqr))].count()

    outliers_columns = X.apply(count_outliers)

    print('Features sem outliers')
    print(outliers_columns[outliers_columns == 0])

    outliers_columns.hist(bins=30)
    plt.show()

    print('### Distribuição da classe alvo')

    print(y.value_counts())

    y.value_counts().sort_index().plot(kind='bar', title='Distribuição da classe alvo')
    plt.show()

    print('### Análise da distribuição dos dados no tempo (meses)')

    examples_by_month = [248, 197, 217, 261, 20, 15, 731, 506, 526, 554, 113, 48, 197, 20, 3, 652, 1625, 3613, 163, 131, 54,
                         416, 3600]

    examples_by_month_complete = [0]
    for i, j in enumerate(examples_by_month):
        if i == 0:
            examples_by_month_complete.append(j - 1)
        else:
            examples_by_month_complete.append(j + examples_by_month_complete[i])

    months = [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 30, 36]

    X = X.reset_index()
    X['month'] = pd.cut(X['index'], bins=examples_by_month_complete, labels=months, include_lowest=True)

    X.drop('index', axis=1, inplace=True)

    print('### Quantidade de exemplos por mês')

    X['month'].value_counts().sort_index()

    print('### Quantidade de gases por mês')

    gass_count_by_month = []
    for idx, rows in X.groupby('month'):
        count_gass = y.loc[rows.index].value_counts()
        count_gass.name = idx
        count_gass
        gass_count_by_month.append(count_gass)

    df_gass_month = pd.concat(gass_count_by_month, axis=1).fillna(0).T

    print(df_gass_month)

    df_gass_month.plot(figsize=(18, 10))
    plt.show()

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ax[0][0].plot(df_gass_month[1])
    ax[0][1].plot(df_gass_month[2])
    ax[0][2].plot(df_gass_month[3])
    ax[1][0].plot(df_gass_month[4])
    ax[1][1].plot(df_gass_month[5])
    ax[1][2].plot(df_gass_month[6])

    ax[0][0].legend()
    ax[0][1].legend()
    ax[0][2].legend()
    ax[1][0].legend()
    ax[1][1].legend()
    ax[1][2].legend()

    plt.show()

    df_gass_month.plot(kind='bar', figsize=(18, 10))
    plt.show()

    # Analisando com PCA a varianca dos dados em relacao aos atributos, apenas para valores numericos
    print("PCA - Realizando a análise de componentes principais com os principais atributos a serem analisados")
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Numero de Componentes encontrados')
    plt.ylabel('Variancia acumulada')
    plt.show()
    print("################################################################################\n")

# Inicio da Etapa - Pré-processamento de Dados
print("Inicio da etapa de pre-processamento de dados")
#realiza a re-escala dos dados
print("Re-escala dos dados - MinMax")
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

print("Resumo dos dados modificados")
numpy.set_printoptions(precision=3)
print(X[0:5,:])

# usando o metodo para fazer uma unica divisao dos dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

print("Início dos classificadores de individuais (KNN, DT, RANDOMFOREST, MLP)\n")
# Criação dos classificadores

if do_knn:
    # KNN
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    clf_knn = clf_knn.fit(X_train, y_train)
    pred_knn = clf_knn.predict(X_test)

    # Imprimindo relatório de classificação do modelo inicial
    print("Relatorio de Classificação do modelo inicial (kNN)")
    print(classification_report(y_test, pred_knn), "\n")

    # Relizando testes com varios k e metricas diferentes
    for k in range(1, 11):

        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance', leaf_size=40)
        clf_knn = clf_knn.fit(X_train, y_train)
        pred_i = clf_knn.predict(X_test)
        print("Acuracia do clf (KNN-euclidean): K = %2d, Train: %0.3f, Teste: %0.3f" % (k, clf_knn.score(X_train, y_train), clf_knn.score(X_test, y_test)))

        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm = 'auto', leaf_size = 30, metric = 'minkowski',
        metric_params=None, n_jobs=None, p=2, weights='uniform')
        clf_knn = clf_knn.fit(X_train, y_train)
        pred_i = clf_knn.predict(X_test)
        print("Acuracia do clf (KNN - minkowski): K = %2d, Train: %0.3f, Teste: %0.3f" % (k, clf_knn.score(X_train, y_train), clf_knn.score(X_test, y_test)))

        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='manhattan', p=k)
        clf_knn = clf_knn.fit(X_train, y_train)
        pred_i = clf_knn.predict(X_test)
        print("Acuracia do clf (KNN - manhattan): K = %2d, Train: %0.3f, Teste: %0.3f" % (k, clf_knn.score(X_train, y_train), clf_knn.score(X_test, y_test)))

        clf_knn = neighbors.KNeighborsClassifier(n_neighbors=k, metric='chebyshev', p=k)
        clf_knn = clf_knn.fit(X_train, y_train)
        pred_i = clf_knn.predict(X_test)
        print("Acuracia do clf (KNN - chebyshev): K = %2d, Train: %0.3f, Teste: %0.3f" % (k, clf_knn.score(X_train, y_train), clf_knn.score(X_test, y_test)))


    # Treinando novamente o modelo com valores otimos de K
    clf_knn = neighbors.KNeighborsClassifier(n_neighbors=4, metric='euclidean')
    clf_knn = clf_knn.fit(X_train, y_train)
    pred = clf_knn.predict(X_test)


    # Imprimindo relatório de classificação do modelo final
    print("Relatorio de Classificação do modelo final (KNN)- melhores parâmetros ")
    print(classification_report(y_test, pred),"\n" )

    print("KNN- Confussion matrix:\n", confusion_matrix(y_test, pred_knn))
    print("KNN- Acuracia: (Treinamento) %0.3f" %  clf_knn.score(X_train, y_train))
    print("KNN- Acuracia: (Testes) %0.3f" %  clf_knn.score(X_test, y_test))
    print("###########################################################################\n")

if do_dt:
    # Decision Tree (DT)
    clf_dt = tree.DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf_dt = clf_dt.fit(X_train, y_train)
    pred_dt = clf_dt.predict(X_test)

    # Imprimindo relatório de classificação do modelo inicial
    print("Relatorio de Classificação do modelo inicial  (Decision Tree (DT)) ")
    print(classification_report(y_test, pred_dt), "\n")

    for i in range(1, 21):

        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=seed)
        clf2 = tree.DecisionTreeClassifier(max_depth=i, random_state=seed)
        clf3 = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, min_samples_split=5, max_depth=i, random_state=seed)

        clf = clf.fit(X_train, y_train)
        clf2 = clf2.fit(X_train, y_train)
        clf3 = clf3.fit(X_train, y_train)

        print("Acuracia do clf (DT - Entropia): K = %2d, Train: %0.3f, Teste: %0.3f" % (
        i, clf.score(X_train, y_train), clf.score(X_test, y_test)))

        print("Acuracia do clf2 (DT - Default): K = %2d, Train: %0.3f, Teste: %0.3f" % (
        i, clf2.score(X_train, y_train), clf2.score(X_test, y_test)))

        print("Acuracia do clf3 (DT - Gini): K = %2d, Train: %0.3f, Teste: %0.3f" % (
        i, clf3.score(X_train, y_train), clf3.score(X_test, y_test)))

        print("Profundidade das arvores criadas para os classificadores")
        print("%2d, %2d, %2d" % (clf.tree_.max_depth, clf2.tree_.max_depth, clf3.tree_.max_depth))


    # Criando a instância final do classificador DT
    clf_dt = tree.DecisionTreeClassifier(max_depth=9, random_state=seed)
    clf_dt = clf_dt.fit(X_train, y_train)
    pred_dt = clf_dt.predict(X_test)

    # Imprimindo relatório de classificação do modelo inicial
    print("Relatorio de Classificação do modelo final  (DT) \n")
    print(classification_report(y_test, pred_dt), "\n")

    print("DT - Confussion matrix:\n", confusion_matrix(y_test, pred_dt))
    print("DT - Acuracia: (Treinamento) %0.3f" %  clf_dt.score(X_train, y_train))
    print("DT - Acuracia: (Testes) %0.3f" %  clf_dt.score(X_test, y_test))
    print("###########################################################################\n")

if do_mlp:
    """
    # Rede Neural com 3 camadas ((neuronios variaveis, random_state=10)

    # Criando a instância inicial do classificador RNA
    clf_rna = MLPClassifier(hidden_layer_sizes=([2,2]), random_state=seed, max_iter=1000)
    clf_rna = clf_rna.fit(X_train, y_train)
    pred_mlp = clf_rna.predict(X_test)

    # Imprimindo relatório de classificação do modelo inicial
    print("Relatorio de Classificação do modelo inicial  (RNA) ")
    print(classification_report(y_test, pred_mlp), "\n")


    for i in range(1, 11):

        # Rede Neural com 1 camada (10 neuronios)
        mlp1 = MLPClassifier(hidden_layer_sizes=([10]), random_state=seed)

        # Rede Neural com 2 camadas (3 e 3 neuronios)
        mlp2 = MLPClassifier(hidden_layer_sizes=(3, 3), random_state=seed)

        # Rede Neural com 3 camadas ((neuronios variaveis, random_state=10)
        mlp3 = MLPClassifier(hidden_layer_sizes=([i, i, i]), random_state=seed)

        # Rede Neural com 1 camada (neuronios variaveis, random_state=10)
        mlp4 = MLPClassifier(hidden_layer_sizes=([i]), random_state=seed)

        # Rede Neural com 2 camadas ( neuronios variveis, random_state = 10)
        mlp5 = MLPClassifier(hidden_layer_sizes=([i, i]), random_state=seed)

        mlp1 = mlp1.fit(X_train, y_train)
        mlp2 = mlp2.fit(X_train, y_train)
        mlp3 = mlp3.fit(X_train, y_train)
        mlp4 = mlp4.fit(X_train, y_train)
        mlp5 = mlp5.fit(X_train, y_train)

        print("Resultados - Classificador RNA \n")
        print("Acuracia do MLP1: i = %2d, Train: %0.3f, Teste: %0.3f" % (i, mlp1.score(X_train, y_train), mlp1.score(X_test, y_test)))
        print("Acuracia do MLP2: i = %2d, Train: %0.3f, Teste: %0.3f"  % (i, mlp2.score(X_train, y_train), mlp2.score(X_test, y_test)))
        print("Acuracia do MLP3: i = %2d, Train: %0.3f, Teste: %0.3f" % (i, mlp3.score(X_train, y_train), mlp3.score(X_test, y_test)))
        print("Acuracia do MLP4: i = %2d, Train: %0.3f, Teste: %0.3f" % (i, mlp4.score(X_train, y_train), mlp4.score(X_test, y_test)))
        print("Acuracia do MLP5: i = %2d, Train: %0.3f, Teste: %0.3f" % (i, mlp5.score(X_train, y_train), mlp5.score(X_test, y_test)))

"""
    # Criando a instância final do classificador RNA
    clf_rna = MLPClassifier(hidden_layer_sizes=([128,6]), random_state=seed)
    clf_rna = clf_rna.fit(X_train, y_train)
    pred_mlp = clf_rna.predict(X_test)


    print("MLP - Clasification report:\n", classification_report(y_test, pred_mlp))
    print("MLP - Confussion matrix:\n", confusion_matrix(y_test, pred_mlp))

    print("MLP- Acuracia: (Treinamento) %0.3f" %  clf_rna.score(X_train, y_train))
    print("MLP- Acuracia: (Testes) %0.3f" %  clf_rna.score(X_test, y_test))
    print("###########################################################################\n")