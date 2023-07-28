import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Função para converter valores com vírgula para ponto
def convert_to_decimal(value):
    if isinstance(value, str) and ',' in value:
        return float(value.replace(',', '.'))
    return value

# Carregar a base de dados
data = pd.read_csv('Amostra_INMET_Fortaleza.csv', sep=';', decimal=',')

# Converter os valores com vírgula para ponto na coluna "VENTO, RAJADA MAXIMA (m/s)"
data['VENTO, RAJADA MAXIMA (m/s)'] = data['VENTO, RAJADA MAXIMA (m/s)'].apply(convert_to_decimal)

# Preencher os valores ausentes (NaN) na coluna "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)" com zero
data['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'].fillna(0, inplace=True)

# Converter a coluna "PRECIPITAÇÃO TOTAL, HORÁRIO (mm)" para valores numéricos
data['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'] = pd.to_numeric(data['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)'])

# Preencher os valores ausentes (NaN) nas outras colunas com zero
data.fillna(0, inplace=True)

# Separar as features (X) e o target (y)
X = data[['UMIDADE RELATIVA DO AR, HORARIA (%)', 'VENTO, RAJADA MAXIMA (m/s)']]
y = data['PRECIPITAÇÃO TOTAL, HORÁRIO (mm)']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

# Criar o modelo de Árvore de Decisão
model = DecisionTreeRegressor(random_state=42)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular o erro médio quadrático
mse = mean_squared_error(y_test, y_pred)
print(f'RMSE: {mse}')

# Calcular o RMSPE
rmspe = np.sqrt(mse) / y.max()  # Dividindo pelo valor máximo de y para obter o erro em percentual
print(f'RMSPE: {rmspe:.2%}')

# Gráfico de Dispersão (Opção 1)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Gráfico de Dispersão: Valores Reais vs. Valores Previstos')
plt.ylim(0, 1.5)  # Definindo limite máximo para o eixo Y
plt.xlim(0, 3)
plt.show()

# Curva de Aprendizado (Opção 6)
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')
    
    train_mean = -np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Treinamento')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    
    plt.plot(train_sizes, test_mean, color='red', marker='o', markersize=5, label='Teste')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='red')
    
    plt.xlabel('Conjunto de treinamento')
    plt.ylabel('RMSE')
    plt.title('Curva de aprendizado')
    plt.legend()
    plt.show()

# Plotar a curva de aprendizado
plot_learning_curve(model, X_train, y_train)
