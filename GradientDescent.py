__author__ = 'Panther'
"""
This class calculates the gradient descent for the given training set and returns the weights of each attributes
"""
import math


class GradientDescent:

    """
    Constructor initializes the training data and loads the attribute details
    """
    def __init__(self, training_list, number_of_attributes, attribute_names, learning_rate, biased_weight = 1):
        self.training_list = training_list
        self.number_of_attributes = number_of_attributes
        self.attributes = []
        for each in range(self.number_of_attributes):
            self.attributes.append({'name': attribute_names[each], "weight":0})
        self.number_of_iterations = 100  # Set default number of iterations to 100
        self.learning_rate = learning_rate
        self.class_index = self.number_of_attributes

    """
    This function assigns the random weights to each of the attributes
    By default, sets the weight to zero
    """
    def set_weights(self, weight=0):
        for each in range(self.number_of_attributes):
            self.attributes[each].update({"weight": weight})

    """
    This function calculates the sigmoid of a given value
    """
    @staticmethod
    def sigmoid(value):
        value = float( 1 / (1 + math.exp(-value)))
        return value

    """
    This function takes iterates through all the examples and updates the weights accordingly
    """
    def update_weights(self, training_example):
        # for training_example in self.training_list:
        prediction = 0
        for each in range(self.number_of_attributes):  # Find the prediction training example using cur. weights
            prediction += float(training_example[each]) * self.attributes[each]["weight"]

        prediction = self.sigmoid(prediction)  # Find the sigmoid of the prediction
        error = float(training_example[self.class_index]) - prediction  # Find the error
        for each in range(self.number_of_attributes):
            # Weight update equation is : wt = wt+(learning_rate)*(Error)*(prediction)*(1-prediction)(input)
            temp_weight = self.attributes[each]["weight"]
            temp_weight += self.learning_rate * error * prediction * (1 - prediction) * int(training_example[each])
            # temp_weight = round(temp_weight, 3)
            self.attributes[each].update({"weight": temp_weight})

    """
    This function finds the weights for the attributes of the given training set
    """
    def find_weights(self, number_of_iterations=100):
        self.number_of_iterations = number_of_iterations
        self.number_of_iterations = 2000
        self.set_weights()  # Set weights to each attribute
        each = 0
        iteration = 0
        for iteration in range(self.number_of_iterations):
            if (iteration % (len(self.training_list))) == 0:
                each = 0
            else:
                each += 1
            self.update_weights(self.training_list[each])

    """
    This function finds the class value of the given data using the calculated weights
    """
    def find_class(self, data):
        prediction = 0
        for each in range(self.number_of_attributes):
            prediction += (float(data[each]) * self.attributes[each]["weight"])

        prediction = self.sigmoid(prediction)
        if prediction >= 0.5:
            prediction = 1
        else:
            prediction = 0

        if prediction == int(data[self.class_index]):
            return True
        else:
            return False

    """
    This function takes input of a test data and returns the accuracy
    """
    def find_accuracy(self, data, ignore_first_line=True):
        hits = 0
        for each in data:
            if self.find_class(each):
                hits += 1
        return float(hits/(float(len(data))) * 100)