import tensorflow as tf


def tensor_scaler(tensor, expected_tensor):
    value_list = []
    data_list = []
    expected_list = []
    for table in tensor:
        for value in table:
            value_list.append(value)
    for table in expected_tensor:
        for value in table:
            value_list.append(value)
    list_max = max(value_list)
    for table in tensor:
        change_list = []
        for value in table:
            x = value/list_max
            change_list.append(x)
        data_list.append(change_list)
    for table in expected_tensor:
        change_list = []
        for value in table:
            x = value / list_max
            change_list.append(x)
        expected_list.append(change_list)
    return data_list, expected_list, list_max


class DeepLearning:
    def __init__(self, name: str, inputs: int, outputs: int, epoch: int = 100, lr: float = 0.001):
        """
        For creating a Neural Network with training data, and values to predict in order to train
        :param convoluted:
        :param fully_connected:
        :param epoch:
        :param lr:
        """
        # name of the model being trained
        self.name = name
        # how much data is being input
        self.inputs = inputs
        # how much data is expected afterwards
        self.outputs = outputs
        # learning rate
        self.lr = lr
        # epochs
        self.epoch = epoch
        self.training_data = []
        self.fc_layers = {}
        self.layer_int = 0
        with tf.variable_scope('input'):
            self.data = tf.placeholder(tf.float32, shape=(None, self.inputs))

    def add_training_data(self, train):
        self.training_data.append(train)

    def del_training_data(self, train):
        self.training_data.remove(train)

    def add_fully_connected_layer(self, nodes: int):
        """
        Pass in a sequential int as a key, and the number of nodes as the value
        :param biases:
        :param weight:
        :param layer_int:
        :param nodes:
        :param layer:
        :return:
        """
        with tf.variable_scope('layer' +str(self.layer_int)):
            if self.layer_int is 0:
                weights = tf.get_variable(name="weight"+str(self.layer_int), shape=[self.inputs, nodes], initializer=tf.contrib.layers.xavier_initializer())
            else:
                layer_before = self.layer_int - 1
                weights = tf.get_variable(name="weight" + str(self.layer_int), shape=[self.fc_layers['layer' + str(layer_before)]['nodes'], nodes], initializer=tf.contrib.layers.xavier_initializer())

            biases = tf.get_variable(name="biases"+str(self.layer_int), shape=[nodes], initializer=tf.zeros_initializer())
            if self.layer_int is 0:
                output = tf.nn.relu(tf.matmul(self.data, weights) + biases)
            else:
                layer_before = self.layer_int - 1
                output = tf.nn.relu(tf.matmul(self.fc_layers['layer'+str(layer_before)]['output'], weights) + biases)

            self.fc_layers['layer' + str(self.layer_int)] = {
                'nodes': nodes,
                'weight': weights,
                'biases': biases,
                'output': output
            }
            self.layer_int += 1

    def del_layer(self, layer: str):
        """
        Renove a dict reference from the layer dict
        :param layer:
        :param nodes:
        :return:
        """
        del self.fc_layers[layer]

    def output_layer(self):
        if not self.fc_layers:
            print('Error, no layers initialized.')
        else:
            with tf.variable_scope('output'):
                final_layer = self.layer_int - 1
                weights = tf.get_variable("final_weights", shape=[self.fc_layers['layer'+str(final_layer)]['nodes'], self.outputs], initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable(name="final_biases", shape=[self.outputs], initializer=tf.zeros_initializer())
                prediction = tf.matmul(self.fc_layers['layer'+str(final_layer)]['output'], weights) + biases
                with tf.variable_scope('cost'):
                    actual_value = tf.placeholder(tf.float32, shape=(None, 1))
                    cost = tf.reduce_mean(tf.squared_difference(prediction, actual_value))
                return cost, prediction, actual_value

    def train_neural_network(self, data_inputs, expected_outputs, tensor_max=None):
        cost, prediction, actual_value = DeepLearning.output_layer(self)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)
            saved_model = tf.train.Saver()
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                for epoch in range(self.epoch):
                    session.run(optimizer, feed_dict={self.data: data_inputs, actual_value: expected_outputs})
                    if epoch % 5 == 0:
                        training_cost = session.run(cost, feed_dict={self.data: data_inputs, actual_value: expected_outputs})
                        print(epoch, training_cost)
                print("Training is complete!")
                final_training_cost = session.run(cost, feed_dict={self.data: data_inputs, actual_value: expected_outputs})
                print("Final Training cost: {}".format(final_training_cost))
                save_path = saved_model.save(session, "logs/trained_model/"+self.name+'.ckpt')
                predictions = session.run(prediction, feed_dict={self.data: data_inputs})
                if tensor_max != None:
                    unscaled_max = predictions*tensor_max
                    print('predicted value was' + str(unscaled_max))

    def load_trained_neural_network(self, data_inputs: list, name: str, tensor_max=None):
        cost, prediction, actual_value = DeepLearning.output_layer(self)
        with tf.variable_scope('train'):
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(cost)
        saved_model = tf.train.Saver()
        with tf.Session() as session:
            saved_model.restore(session, name)
            predictions = session.run(prediction, feed_dict={self.data: data_inputs})
            if tensor_max != None:
                unscaled_max = predictions*tensor_max
                print('predicted value was' + str(unscaled_max))
