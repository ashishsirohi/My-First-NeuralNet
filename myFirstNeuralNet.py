# My 3 layer Neural Network

from numpy import exp, array, random, dot

class NeuralNetwork():
	def __init__(self):
		# Seed the random number generator, so it generates the 
		# same numbers every time the program runs.
		# Good for debugginh
		random.seed(1)

		# We model a perceptron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1 and mean 0.
		self.syn_weights = 2 * random.random((3,1)) - 1

	# The Sigmoid function, We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
	def sigmoidFn(self, x):
		return 1 / (1 + exp(-x))

	# The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
	def sigmoidDeriv(self, x):
		return x * (1 - x)

	# We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
	def trainNetwork(self, input_set, output_set, num_iterations):
		for i in xrange(num_iterations):
			# Passing the training set to our neural net consisting of perceptrons
			predicted_output = self.perceptron(input_set)
			# Calculating error rate
			error = output_set - predicted_output
			# Adjusting the weights using backpropogation  
			new_weight = dot(input_set.T, error * self.sigmoidDeriv(predicted_output))
			# Updating weight
			self.syn_weights += new_weight

	# A perceptron taking input and producing output using sigmoid function
	def perceptron(self, input_set):
		return self.sigmoidFn(dot(input_set, self.syn_weights))

if __name__=="__main__":

	nn = NeuralNetwork()

	print "Initial synaptic weights generated randomly: ", nn.syn_weights

	# Input
	input_set = array([[0,0,1],
				[1,1,1],
				[1,0,1],
				[0,1,1]
				])

	# Output
	output_set = array([[0],
				[1],
				[1],
				[0]])

	# Train the neural network using a training set.
	nn.trainNetwork(input_set, output_set, 50000)

	print "New synaptic weights after training: ", nn.syn_weights

	# Testing the NN with new input
	print "Predicted Output: ", nn.perceptron(array([[0,0,1],
				[1,1,1],
				[1,0,1],
				[0,1,1]
				]))
