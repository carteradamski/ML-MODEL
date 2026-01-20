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

        if self.lf == "logcosh":
            diff = output_expression - autodiff.constant(y_true_value)
            e_pos = autodiff.exponent(autodiff.constant(math.e), diff)
            e_neg = autodiff.exponent(autodiff.constant(math.e), autodiff.constant(-1) * diff)
            avg = (e_pos + e_neg) / autodiff.constant(2)
            return autodiff.natlog(avg)

        if self.lf == "mae":
            diff = output_expression - autodiff.constant(y_true_value)
            sq = diff * diff
            return autodiff.exponent(sq, autodiff.constant(0.5))
        
        

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
            values[weight_name] = .00000000000000000000000001

        # Initialize biases with small random values
        for bias_name in self.biases:
            values[bias_name] = .00000000000000000000000001
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
    print("Housing Price Prediction")

    

    training_housing = [
    ([2, 1, 26], 385),
    ([2, 3, 26], 447),
    ([2, 3, 18], 378),
    ([3, 3, 22], 440),
    ([2, 3, 15], 321),
    ([3, 2, 28], 450),
    ([4, 2, 26], 472),
    ([4, 2, 23], 448),
    ([5, 3, 29], 574),
    ([5, 1, 29], 541),
    ([4, 2, 27], 478),
    ([5, 1, 21], 501),
    ([3, 3, 30], 480),
    ([5, 1, 28], 528),
    ([3, 1, 29], 432),
    ([2, 2, 20], 359),
    ([4, 2, 28], 498),
    ([3, 1, 22], 387),
    ([3, 3, 28], 472),
    ([2, 2, 27], 364),
    ([5, 2, 20], 541),
    ([4, 1, 26], 471),
    ([2, 2, 20], 358),
    ([4, 1, 18], 422),
    ([5, 3, 16], 531),
    ([5, 2, 28], 559),
    ([5, 3, 24], 561),
    ([3, 2, 17], 400),
    ([4, 2, 29], 488),
    ([4, 2, 16], 455),
    ([3, 1, 19], 368),
    ([3, 3, 24], 456),
    ([5, 2, 29], 581),
    ([2, 3, 28], 408),
    ([2, 2, 25], 373),
    ([4, 1, 20], 438),
    ([2, 2, 18], 343),
    ([2, 2, 12], 328),
    ([5, 3, 15], 527),
    ([3, 2, 16], 406),
    ([3, 1, 29], 440),
    ([4, 3, 28], 531),
    ([5, 2, 16], 521),
    ([5, 2, 19], 527),
    ([3, 2, 17], 403),
    ([3, 1, 22], 383),
    ([3, 3, 20], 424),
    ([4, 2, 25], 482),
    ([5, 2, 29], 581),
    ([2, 3, 21], 372),
    ([4, 2, 27], 481),
    ([3, 2, 19], 415),
    ([3, 2, 28], 434),
    ([4, 2, 16], 445),
    ([4, 1, 22], 441),
    ([2, 2, 20], 350),
    ([2, 2, 27], 378),
    ([5, 3, 25], 576),
    ([5, 1, 20], 513),
    ([3, 2, 16], 393),
    ([2, 1, 20], 321),
    ([2, 3, 29], 415),
    ([4, 2, 21], 470),
    ([2, 2, 26], 378),
    ([2, 1, 23], 341),
    ([3, 2, 30], 453),
    ([4, 1, 27], 463),
    ([5, 2, 15], 503),
    ([4, 2, 22], 457),
    ([4, 1, 29], 478),
    ([5, 2, 25], 562),
    ([3, 2, 25], 427),
    ([4, 2, 19], 460),
    ([4, 1, 15], 419),
    ([4, 3, 22], 513),
    ([3, 2, 20], 420),
    ([3, 3, 18], 435),
    ([2, 1, 29], 369),
    ([5, 3, 10], 516),
    ([2, 1, 16], 305),
    ([5, 2, 17], 517),
    ([2, 2, 15], 343),
    ([5, 2, 22], 542),
    ([5, 3, 20], 558),
    ([3, 1, 26], 405),
    ([3, 3, 20], 443),
    ([3, 2, 20], 415),
    ([4, 1, 25], 468),
    ([4, 2, 28], 492),
    ([4, 3, 19], 494),
    ([5, 3, 16], 542),
    ([2, 2, 30], 391),
    ([5, 2, 19], 532),
    ([4, 2, 12], 438),
    ([2, 3, 18], 359),
    ([4, 3, 25], 518),
    ([3, 2, 17], 394),
    ([4, 3, 18], 500),
    ([2, 3, 25], 392),
    ([3, 3, 13], 418),
    ]

    testing_housing = [
    ([3, 2, 19], 412),
    ([2, 2, 25], 368),
    ([5, 3, 19], 544),
    ([4, 1, 30], 488),
    ([5, 2, 20], 531),
    ([3, 2, 21], 411),
    ([5, 3, 15], 542),
    ([5, 2, 28], 569),
    ([2, 2, 27], 384),
    ([4, 3, 15], 491),
    ([4, 1, 27], 468),
    ([4, 1, 28], 487),
    ([5, 1, 25], 516),
    ([3, 3, 14], 418),
    ([2, 3, 16], 349),
    ([3, 1, 14], 369),
    ([2, 2, 24], 364),
    ([3, 2, 25], 432),
    ([5, 3, 17], 549),
    ([5, 2, 30], 580),
    ([3, 3, 28], 472),
    ([4, 3, 20], 508),
    ([4, 2, 25], 475),
    ([2, 2, 24], 370),
    ([3, 2, 27], 440),
    ([2, 1, 27], 363),
    ([4, 2, 10], 430),
    ([5, 2, 28], 573),
    ([4, 1, 24], 450),
    ([3, 3, 16], 431),
    ([3, 2, 19], 409),
    ([2, 2, 29], 395),
    ([3, 2, 29], 447),
    ([4, 2, 26], 483),
    ([2, 3, 19], 361),
    ([5, 1, 26], 520),
    ([2, 2, 18], 347),
    ([3, 1, 24], 402),
    ([4, 3, 22], 514),
    ([3, 2, 27], 438),
    ([5, 2, 18], 525),
    ([4, 2, 29], 496),
    ([3, 3, 25], 458),
    ([2, 1, 22], 338),
    ([5, 3, 24], 568),
    ([2, 2, 28], 388),
    ([3, 2, 17], 399),
    ([4, 1, 25], 463),
    ([5, 2, 27], 568),
    ([3, 3, 18], 431),
    ([4, 2, 22], 468),
    ([2, 3, 25], 395),
    ([5, 1, 28], 541),
    ([3, 2, 24], 428),
    ([4, 3, 17], 499),
    ([2, 1, 25], 350),
    ([5, 2, 22], 543),
    ([3, 3, 27], 463),
    ([4, 2, 18], 459),
    ([2, 2, 22], 358),
    ([5, 1, 25], 520),
    ([3, 2, 28], 442),
    ([4, 3, 24], 528),
    ([2, 1, 17], 319),
    ([5, 2, 25], 560),
    ([3, 3, 22], 443),
    ([4, 2, 27], 488),
    ([2, 2, 18], 349),
    ([5, 1, 22], 513),
    ([3, 2, 25], 434),
    ([4, 3, 28], 532),
    ([2, 1, 24], 344),
    ([5, 2, 17], 519),
    ([3, 3, 25], 455),
    ([4, 2, 22], 468),
    ([2, 2, 28], 387),
    ([5, 1, 18], 508),
    ([3, 2, 22], 414),
    ([4, 3, 25], 520),
    ([2, 1, 27], 358),
    ([5, 2, 24], 555),
    ([3, 3, 17], 428),
    ([4, 2, 25], 480),
    ([2, 2, 24], 369),
    ([5, 1, 27], 536),
    ([3, 2, 18], 409),
    ([4, 3, 22], 513),
    ([2, 1, 25], 350),
    ([5, 2, 28], 572),
    ([3, 3, 24], 448),
    ([4, 2, 17], 449),
    ([2, 2, 22], 358),
    ([5, 1, 25], 520),
    ([3, 2, 27], 438),
    ([4, 3, 18], 509),
    ([2, 1, 22], 333),
    ([5, 2, 25], 560),
    ([3, 3, 28], 467),
    ([4, 2, 24], 474),
    ]
    nn_house = neuralnet(n_in=3, n_hn=4, hl_af="linear", ol_af="linear", lf="logcosh")

    # Train the network
    values_dict, loss_history = nn_house.train(training_housing, epochs = 300
                                               , learning_rate= .0001)

    # Test predictions
    print("\nResults:")
    total_error = 0
    for X, y_true in testing_housing:
        prediction = nn_house.predict(X, values_dict)
        error = abs(prediction - y_true)
        total_error+=error
        print(
            f"Input: {X}, True: {y_true:.1f}, Predicted: {prediction:.4f}, Error: {error:.4f}"
        )
    print(total_error)


if __name__ == "__main__":
    main()