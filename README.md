# Gas-Sensor-Array-Drift

Projeto da disciplina Soluções de Mineração de Dados da UFPE.

Grupo:
* Hermann Schuenemann
* Giorgio Onofre
* Milton Gama Neto
* Paulo Júnior de Moraes

Este projeto consiste construir uma solução seguinte todos os passos do CRISP-DM (CRoss Industry Standard Process for Data Mining) para detecção de gases. Foi utilizado o dataset do UCI Machine Learning Repository https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset.

Existe um aspecto temporal nos dados que é objeto de estudo deste trabalho. Os dados são coletados em meses diferentes, porém não existe detalhamente profundo disto e também não tem um processo uniformizado, existe um espaço entre as coletas e desbalanceamentos entre os meses. Apenas a análise exploratória dos dados contemplou esse tipo de informação, enquanto o resto do processo foi realizado com a base unificada, para fins acadêmicos.

Segue a ordem de execução dos notebooks e descrições:
1. **analise.ipynb** <br>
    Análise exploratória dos dados realizada após o entendimento do negócio.
2. **preprocessamento.ipynb** <br>
    Preparação dos dados para etapa de modelagem. Foi realizado a padronização dos dados e redução de dimensionalidade.
3. **modelagem.ipynb** <br>
    Experimentos com diversos algoritmos de Machine Learning e ajuste de hiperparâmetros.
4. **desempenhos.ipynb** <br>
    Comparação entre os classificadores com os melhores parâmetros encontrados. Testes estatísticos e análise no conjunto de treinamento.
    
