# 💸 Predição de Renda Anual: Classificação com Regressão Linear e Logística

Este projeto aborda um problema de **classificação binária** com o objetivo de prever se a renda anual de um indivíduo ultrapassa 50 mil dólares (`>50K`) ou não (`<=50K`), utilizando dados censitários. O principal destaque do projeto é a comparação didática e prática entre a **Regressão Linear** (adaptada como um Modelo de Probabilidade Linear - LPM) e a **Regressão Logística**.

## 🗂️ Sobre os Dados
O conjunto de dados (`ayessa/salary-prediction-classification`) possui 32.561 instâncias.
* **Variáveis Numéricas:** Idade (`age`), Anos de Estudo (`education-num`), Ganhos/Perdas de Capital (`capital-gain`, `capital-loss`), Horas por semana (`hours-per-week`).
* **Variáveis Categóricas:** Classe de trabalho (`workclass`), Estado Civil (`marital-status`), Ocupação (`occupation`), Relacionamento, Raça, Sexo e País de Origem.
* **Variável Alvo:** `salary` (transformada na feature binária `target`: 1 para `>50K`, 0 para `<=50K`).

## 📊 Análise Exploratória (EDA)
A análise exploratória revelou os seguintes padrões:
* **Desbalanceamento de Classes:** Cerca de 76% dos indivíduos ganham `<=50K`, enquanto apenas 24% ganham `>50K`.
* **Correlações visuais:** Gráficos de dispersão e boxplots mostraram que idade mais avançada, maior nível educacional e mais horas de trabalho semanais estão positivamente associados a salários mais altos.

## 🛠️ Pré-processamento (Pipeline)
Foi construído um pipeline robusto com o `scikit-learn` garantindo que não houvesse vazamento de dados (*data leakage*):
1. **Limpeza:** Remoção das colunas `fnlwgt` (peso amostral irrelevante para a predição), `education` (redundante devido à presença de `education-num`) e a coluna original de `salary`.
2. **Divisão Estratificada:** Treino (70%) e Teste (30%) mantendo a proporção da variável alvo desbalanceada (`stratify=y`).
3. **Transformações Numéricas:** Padronização com `StandardScaler`.
4. **Transformações Categóricas:** Codificação com `OneHotEncoder` (`drop='first'` para evitar multicolinearidade e `handle_unknown='ignore'`).

## 🤖 Modelagem e Avaliação
Foram testadas duas abordagens distintas para prever a classe do salário:

1. **Regressão Linear (Modelo de Probabilidade Linear - LPM):**
   * Por ser um modelo contínuo, aplicou-se um limiar (*threshold*) de `0.5` para forçar a classificação (valores > 0.5 foram considerados classe 1).
   * **Resultados:** Acurácia de 84% e AUC-ROC de 0.8953. Porém, apresentou dificuldades na classe minoritária (`>50K`), com **Recall de 0.53** e **F1-Score de 0.61**.

2. **Regressão Logística:**
   * Algoritmo nativo para classificação binária, estimando as probabilidades por meio da função sigmoide.
   * **Resultados:** Acurácia de 85% e AUC-ROC de 0.9084. O desempenho na classe minoritária foi significativamente superior, alcançando **Recall de 0.61** e **F1-Score de 0.67**.

## 📈 Conclusão
A comparação comprova que, embora um Modelo de Probabilidade Linear (Regressão Linear) possa oferecer uma acurácia global razoável, a **Regressão Logística é muito mais adequada e eficiente para problemas de classificação**, lidando melhor com as probabilidades e o desbalanceamento das classes (entregando um Recall e F1-Score superiores para o evento de interesse).

## 💻 Tecnologias Utilizadas
* `Python` (Jupyter Notebook)
* `Pandas` e `NumPy` para manipulação de matrizes e dataframes.
* `Scikit-Learn` para construção do `Pipeline`, transformadores (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), modelos e métricas.
* `Matplotlib` e `Seaborn` para visualização gráfica de dados e plotagem da matriz de métricas.
