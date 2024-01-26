# Passo a Passo
# 0 - Entender o que foi pedido pela empresa
# 1 - Importar a base de dados
# 2 - Preparar a base de dados para a IA
# 3 - Criar um modelo de IA --- para score de crédito: Ruim, Médio, Ótimo
# 4 - Escolher uma modelo do IA
# 5 - Usar a nossa IA para fazer novas previsões 
!pip install scikit-learn

import pandas as pd 

tabela = pd.read_csv("clientes.csv")
display(tabela)

display(tabela.info())

# Profissão
# mix_credito
# comportamento_pagamento

from sklearn.preprocessing import LabelEncoder
codificador = LabelEncoder()

# codificador aplica na coluna Profissão
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])

# codificador aplica na coluna mix_credito
tabela["mix_credito"] = codificador.fit_transform(tabela["mix_credito"])

#codificador aplica na coluna comportamento_pagamento
tabela["comportamento_pagamento"] = codificador.fit_transform(tabela["comportamento_pagamento"])

print(tabela.info())

# aprendizado de máquina 
# y colina que quero prever
# x coluna para ajudar na previsão 
    # Não vamos utilizar o id-cliente, pois é um número aleatório

y = tabela["score_credito"]
x = tabela.drop(columns=["score_credito", "id_cliente"])

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split (x, y, test_size=0.3)

# Criar a IA 
# Árvore de Decisão - RandomForest
# KNN - Vizinhos Próximos - Kneighbors

# Importar a IA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Criar os modelos
modelo_arvorededeciçao = RandomForestClassifier()
modelo_vizinhosproximos = KNeighborsClassifier()

# Treinar os modelos
modelo_arvorededeciçao.fit(x_treino, y_treino)
modelo_vizinhosproximos.fit(x_treino, y_treino)

# Testar os modelos 

from sklearn.metrics import accuracy_score

previsao_arvorededeciçao = modelo_arvorededeciçao.predict(x_teste)
previsao_vizinhosproximos = modelo_vizinhosproximos.predict(x_teste.to_numpy())

print(accuracy_score(y_teste, previsao_arvorededeciçao))
print(accuracy_score(y_teste, previsao_vizinhosproximos))

# melhor modelo: modelo_arvorededeciçao
# fazer novas previsões
# importar novos clientes 
tabela_novos_clientes = pd.read_csv("novos_clientes.csv")
display(tabela_novos_clientes)

# codificar os novos clientes 
codificador = LabelEncoder()

# codificador aplica na coluna Profissão
tabela_novos_clientes["profissao"] = codificador.fit_transform(tabela_novos_clientes["profissao"])

# codificador aplica na coluna mix_credito
tabela_novos_clientes["mix_credito"] = codificador.fit_transform(tabela_novos_clientes["mix_credito"])

#codificador aplica na coluna comportamento_pagamento
tabela_novos_clientes["comportamento_pagamento"] = codificador.fit_transform(tabela_novos_clientes["comportamento_pagamento"])


# fazer as previsões 

previsoes = modelo_arvorededeciçao.predict(tabela_novos_clientes)
print(previsoes)
