import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Inicializacao dos dados da base Iris
filepath = "iris/iris.data"

data = pd.read_csv(filepath, delimiter=",", dtype=None)

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

#Dividindo data na base de teste e de treinamento
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=1234)

#Definindo hiperparametros
hidden_layer = [(10,),(25,),(50,),(100,)]
learning_rates = [0.1,0.05,0.01,0.001]
batch_sizes = [16, 32, 64]

best_acc = 0.0
best_params = {"hidden_layer": None, "learning_rates": None, "batch_sizes": None}

#teste das diferentes combinacoes de hiperparametros
for bs in batch_sizes:
    for lr in learning_rates:
        for hl in hidden_layer:
            print(f"Testando hiperparametros:\nHidden Layer: {hl}\nLearning Rate: {lr}\nBatch Size: {bs}")
            #Criacao do classificador com os hiperparametros do loop
            classifier = MLPClassifier(hidden_layer_sizes=hl,learning_rate_init=lr, batch_size=bs, random_state=1234)

            #treinamento
            classifier.fit(X_treino, y_treino)

            #predicao do classificador gerado
            y_result = classifier.predict(X_teste)

            #acuracia dos resultados
            acc = accuracy_score(y_teste, y_result)
            print(f"Accuracy; {acc}\n")

            if acc > best_acc:
                best_acc = acc
                best_params["batch_sizes"] = bs
                best_params["hidden_layer"] = hl
                best_params["learning_rates"] = lr
print("\n------------------------------------")
print("Resultados:")
print("------------------------------------")
print(f"Acuracidade: {best_acc}")
print(f"Hiperparametros:\n{best_params}")
