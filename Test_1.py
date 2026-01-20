### NEURAL NETWORK CLASS ###
import autodiff
import random
import math


class neuralnet:
    """Neural network class, powered by in-python automatic differentiation.

    Arguments:
        n_in (int): number of input nodes, corresponding to the number of features in a dataset.
        n_hn (int): number of hidden nodes. Increasing this gives the ANN more flexibility, but increases overfitting and runtime.
        hl_af (expression): hidden layer activation function.
        ol_af (expression): output layer activation function
                            Use case examples:
                            Regression: use linear activation
                            Classification: use Sigmoid
        lf (function): loss function (should just be MSE for now)
    """

    def __init__(self, n_in, n_hn, hl_af, ol_af, lf):
        self.n_in = n_in
        self.n_hn = n_hn
        self.hl_af = hl_af
        self.ol_af = ol_af
        self.lf = lf

        # inputs, weights, bias initialization
        # weights naming system: w_from_layer_from_node_to_layer_to_node
        # bias naming system: b_layer (single bias for hidden layer)
        self.weights = {}
        self.biases = {}
        self.inputs = {}

        # create input variables
        for i in range(n_in):
            # TODO [optional]: Change the inputs namking system if you want
            input_name = "x_input_" + str(i)
            self.inputs[input_name] = autodiff.expression(input_name)

        # initialize weights with small random values
        # Input to hidden layer weights
        for i in range(n_in):
            for h in range(n_hn):
                # TODO [optional]: Change the weights namking system if you want
                weight_name = "w_from_input_" + str(i) + "_to_hidden_" + str(h)
                self.weights[weight_name] = autodiff.expression(weight_name)

        # Hidden to output layer weights (single output node)
        for h in range(n_hn):
            # TODO [optional]: Change the weights namking system if you want
            weight_name = "w_from_hidden_" + str(h) + "_to_output"
            self.weights[weight_name] = autodiff.expression(weight_name)

        # initialize single bias for hidden layer
        bias_name = "b_hidden"
        self.biases[bias_name] = autodiff.expression(bias_name)

    def feedforward(self, X):
        """Feedforward function.

        Arguments:
            X (array-like): input data, shape (n_samples, n_features)

        Returns:
            output_expression: autodiff expression representing single output
        """

        # Input-hidden
        hidden_outputs = []
        for h in range(self.n_hn):
            # Calculate weighted sum for this hidden node (with single shared bias)
            weighted_sum = self.biases["b_hidden"]
            for i in range(self.n_in):
                # TODO [optional]: If you'd like, adjust the naming to align with your changes earlier
                weighted_sum = (
                    weighted_sum
                    + self.weights["w_from_input_" + str(i) + "_to_hidden_" + str(h)]
                    * self.inputs["x_input_" + str(i)]
                )

            # Apply activation function
            if self.hl_af == "sigmoid":
                activated = autodiff.sigmoid(weighted_sum)
            elif self.hl_af == "relu":
                activated = autodiff.relu(weighted_sum)
            elif self.hl_af == "tanh":
                activated = autodiff.hyperbolic_tan(weighted_sum)
            elif self.hl_af == "linear":
                activated = autodiff.linear(weighted_sum)
            else:
                activated = weighted_sum  # default to linear (easy)

            hidden_outputs.append(activated)

        # Hidden-output (single output node, no bias)
        weighted_sum = autodiff.constant(0)
        for h in range(self.n_hn):
            # TODO [optional]: If you'd like, adjust the naming to align with your changes earlier
            weighted_sum = (
                weighted_sum
                + self.weights["w_from_hidden_" + str(h) + "_to_output"]
                * hidden_outputs[h]
            )

        # Apply activation function
        if self.ol_af == "sigmoid":
            output_expression = autodiff.sigmoid(weighted_sum)
        elif self.ol_af == "relu":
            output_expression = autodiff.relu(weighted_sum)
        elif self.ol_af == "tanh":
            output_expression = autodiff.hyperbolic_tan(weighted_sum)
        elif self.ol_af == "linear":
            output_expression = autodiff.linear(weighted_sum)
        else:
            output_expression = weighted_sum  # default to linear

        return output_expression

    # TODO: Take a look at these next two functions and understand what they do.
    # You don't need to implement anything -- just know how the funcs work to use them later.
    # Compute_loss simply returns a final autodiff expression representing MSE(predictions, actual).
    # Initialize_weights seeds our weights and bias term with small random values.
    def compute_loss(self, y_true_value, output_expression, values_dict):
        """Compute loss using autodiff expressions.

        Arguments:
            y_true_value (float): true target value
            output_expression: autodiff expression for network output
            values_dict (dict): current values of all variables

        Returns:
            loss_expression: autodiff expression representing the loss
        """
        # only loss function so far -- feel free to implement more
        if self.lf == "mse":
            # Mean Squared Error for single output
            diff = output_expression - autodiff.constant(y_true_value)
            loss = diff * diff
            return loss

    def initialize_weights(self, seed=None):
        """Initialize weights and biases with small random values.

        Arguments:
            seed (int): random seed for reproducibility
        """
        if seed:
            random.seed(seed)

        values = {}

        # Initialize weights with small random values
        for weight_name in self.weights:
            values[weight_name] = random.uniform(-.1, .1)

        # Initialize biases with small random values
        # Initialize biases to zero for more stable training start
        for bias_name in self.biases:
            values[bias_name] = 0.0

        return values

    def backpropagate(self, y_true, output_expression, values_dict):
        """Backpropagation function using automatic differentiation.

        Arguments:
            y_true (float): true label
            output_expression: autodiff expression for network output
            values_dict (dict): current values of all variables (inputs, weights, biases)

        Returns:
            gradients (dict): gradients of loss with respect to weights and biases
        """

        # Compute loss expression
        loss_expr = self.compute_loss(y_true, output_expression, values_dict)

        # TODO: At each training step, compute the gradients with respect to each weight.
        # Implement this in the for loops below.
        gradients = {}

        # Gradients with respect to weights
        for weight_name in self.weights:
            gradients[weight_name] = loss_expr.diff(values_dict, weight_name)
            # HINT: you've already created the loss function -- you just need to differentiate.

        # Gradients with respect to biases
        for bias_name in self.biases:
            gradients[bias_name] = loss_expr.diff(values_dict, bias_name)

        # Return the created dictionary of gradient values.
        # We'll actually adjust the weights and bias term later.
        return gradients

    def train_step(self, X, y_true, values_dict, learning_rate=0.01):
        """Perform one training step (forward + backward + update).

        Arguments:
            X (list): input data, shape (n_features,)
            y_true (float): true target value
            values_dict (dict): current values of all variables
            learning_rate (float): learning rate for gradient descent

        Returns:
            loss_value (float): current loss value
            updated_values_dict (dict): updated parameter values
        """
        # TODO: In the TODOs marked below, complete a full training step of the neural network.
        # Set input values
        for i, val in enumerate(X):
            values_dict[f"x_input_{i}"] = val

        # Forward pass
        output_expression = (
            self.feedforward(X)
        )
        # HINT: we made a function to do this last meeting, which you can call here.

        # Compute loss
        loss_expr = self.compute_loss(y_true, output_expression, values_dict)
        loss_value = loss_expr.eval(values_dict)

        # Backward pass
        gradients = self.backpropagate(y_true, output_expression, values_dict)

        # Update parameters using gradient descent
        for param_name in gradients:
            values_dict[param_name] -= gradients[param_name] * learning_rate
            # HINT: we want to use the gradient values, multiplied by another value to adjust the rate of learning.

        return loss_value, values_dict

    def train(self, training_data, epochs=1000, learning_rate=0.1, verbose=True):
        """Train the neural network on provided data.

        Arguments:
            training_data (list): list of (input, target) tuples
                -- input is a list of floats, target is a float
            epochs (int): number of training epochs
            learning_rate (float): learning rate for gradient descent
            verbose (bool): whether to print training progress

        Returns:
            values_dict (dict): trained parameter values
            loss_history (list): history of loss values
        """
        # TODO: for each epoch in epochs, call the train_step funciton.
        # These two values -- values_dict and loss_history -- are optional to return.
        values_dict = self.initialize_weights()
        loss_history = []
        for epoch in range(epochs):
            for X, y_true in training_data:
                loss_value, values_dict = self.train_step(X, y_true, values_dict, learning_rate)
                loss_history.append(loss_value)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_value:.4f}")

        # TODO: repeatedly call train_step, which will update your weights and biases and train the network.
        # HINT: when you call train_step, you'll use values_dict (created in this function) as an input.
        # HINT: You might want to print the total loss each 100 or so epochs to understand training progress better.

        return values_dict, loss_history

    def predict(self, X, values_dict):
        """Make predictions using current weights and biases.

        Arguments:
            X (list): input data, shape (n_features,)
            values_dict (dict): current values of all variables

        Returns:
            prediction (float): predicted output value
        """
        # Set input values -- note, NOT your weights
        for i, val in enumerate(X):
            values_dict[f"x_input_{i}"] = val

        # Get output expression
        output_expression = self.feedforward(X)

        # Evaluate output expression
        prediction = output_expression.eval(values_dict)

        return prediction


# TODO: once you're done with the rest of the code, run this file to call main().
# If you've done everything properly, you'll be able to predict for this non-linear problem very accurately.
# (Nothing to code here)
def main():
    print("XOR Problem (Classification)")

    linear_use = neuralnet(n_in=1, n_hn=2, hl_af="linear", ol_af="linear", lf="mse")

    training_data = [
    ([0.0], 1.2),
    ([0.5], 2.3),
    ([1.0], 2.8),
    ([1.5], 4.1),
    ([2.0], 5.3),
    ([2.5], 5.7),
    ([3.0], 7.2),
    ]

    testing_data = [
    ([0.25], 1.4),
    ([0.75], 2.7),
    ([1.25], 3.3),
    ([1.75], 4.6),
    ([2.25], 5.7),
    ([2.75], 6.3),
    ]
    # Train the network
    values_dict, loss_history = linear_use.train(training_data, epochs=10000, learning_rate=0.0001)

    # Test predictions
    print("\nXOR Results:")
    for X, y_true in testing_data:
        prediction = linear_use.predict(X, values_dict)
        error = abs(prediction - y_true)
        print(
            f"Input: {X}, True: {y_true:.1f}, Predicted: {prediction:.4f}, Error: {error:.4f}"
        )


if __name__ == "__main__":
    main()