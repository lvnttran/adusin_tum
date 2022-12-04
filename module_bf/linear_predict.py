import pandas as pd
import matplotlib.pyplot as plt


sale_data = pd.read_csv('data_train/clean_data.csv')
X = sale_data.values[:, 7]
y = sale_data.values[:, 4]
plt.scatter(X, y, marker='o')
# plt.show()


def predict(new_radio, weight, bias):
    return weight * new_radio + bias


def cost_function(X, y, weight, bias):
    n = len(X)
    sum_error = 0
    for i in range(n):
        sum_error += (y[i] - (weight * X[i] + bias)) ** 2
    return sum_error / n


def update_weight(X, y, weight, bias, learning_rate):
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        weight_temp += -2 * X[i] * (y[i] - (X[i] * weight + bias))
        bias_temp += -2 * (y[i] - (X[i] * weight + bias))
    weight -= (weight_temp / n) * learning_rate
    bias -= (bias / n) * learning_rate

    return weight, bias


def train(X, y, weight, bias, learning_rate, iter):
    cos_his = []
    for i in range(iter):
        weight, bias = update_weight(X, y, weight, bias, learning_rate)
        cost = cost_function(X, y, weight, bias)
        cos_his.append(cost)

    return weight, bias, cos_his


weight, bias, cost = train(X, y, 0.03, 0.0014, 0.001, 10)

# print(weight)
# print(bias)
# print(cost)
# fuel_index = float(input("Enter fuel price: "))
# sale_predict = predict(fuel_index, weight, bias)
# print(f"Weekly sale is: {sale_predict}")

# solanlap = [i for i in range(10)]
# plt.plot(solanlap, cost)
# plt.show()
