from DeepLearning import *

#data = [[1, 17, 91,2, 226, 38,20000, 1002, 1200,10, 18, 200,19, 31, 201]]
#expected = [[1,2,3,4,7]]
data = []
expected = []
y = 0
x = 1
while x < 100:
    x_list = [x]
    data.append(x_list)
    x += 1
    if y is 0:
        placeholder = 1
        y_list = [y, placeholder]
        expected.append(y_list)
        y = 1
    else:
        placeholder = 0
        y_list = [y, placeholder]
        expected.append(y_list)
        y = 0
print(data)
print(expected)

#data, expected, maximum = tensor_scaler(data, expected)

test = DeepLearning('test', 1, 2, 100, .001)
test.add_fully_connected_layer(2)
test.add_fully_connected_layer(10)
test.add_fully_connected_layer(2)
test.train_neural_network(data, expected)
#test.load_trained_neural_network(data, "logs/trained_model/test.ckpt")