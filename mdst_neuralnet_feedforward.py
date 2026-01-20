import autodiff
import random
import math


class artificial_neural_net:
    """Neural network class, powered by in-python automatic differentiation.

    Arguments:
        n_in (int): number of input nodes, corresponding to the number of features in a dataset.
        n_hl (int): number of hidden nodes. Increasing this gives the ANN more flexibility, but increases overfitting and runtime.
        n_bt (int): number of bias terms.
        hl_af (expression): hidden layer activation function.
        ol_af (expression): output layer activation function.
                            Regression: use linear activation
                            Classification: Sigma RAHHH
        lf (function): loss function (should just be MSE for now)
    """

    def __init__(self, n_in, n_hl, n_bt, hl_af, ol_af, lf):
        self.n_in = n_in
        self.n_hl = n_hl
        self.n_bt = n_bt
        self.hl_af = hl_af
        self.ol_af = ol_af
        self.lf = lf

        # inputs, weights, bias initialization
        # weights naming system: w_layer_from_to
        # bias naming system: b_layer_node
        self.weights = {}
        self.biases = {}
        self.inputs = {}
        self.outputs = {}

        # create input variables
        for i in range(n_in):
            self.inputs[f"x_{i}"] = autodiff.expression(f"x_{i}")

        # TODO: For each weight and bias, create an autodiff expression and store it in the appropriate dictionary.
        # initialize weights with small random values
        # Input to hidden layer weights
        for i in range(n_in):
            for h in range(n_hl):
                weight_name = (
                    f"w_input_{i}_hidden_{h}"
                )
                self.weights[weight_name] = (
                    autodiff.expression(weight_name)
                )

        # Hidden to output layer weights
        for h in range(n_hl):
            weight_name = (
                f"w_hidden_{h}_output"
               
            )
            self.weights[weight_name] = (
                autodiff.expression(weight_name)
            )

        # initialize biases
        for h in range(n_hl):
            bias_name = f"b_hidden_{h}"
            self.biases[bias_name] = (
                autodiff.expression(bias_name)
            )

    def feedforward(self, X):
        """Feedforward function.

        Arguments:
            X (array-like): input data, shape (n_samples, n_features)

        Returns:
            output_expressions (list): list of autodiff expressions representing outputs
        """

        # Input-hidden
        hidden_outputs = []
        for h in range(self.n_hl):
            weighted_sum = self.biases[f"b_hidden_{h}"] 
            for i in range(self.n_in):
                weighted_sum += self.weights[f"w_input_{i}_hidden_{h}"] * self.inputs[f"x_{i}"]

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

        # Hidden-output, including bias terms
        output_expressions = []
        weighted_sum = (
            self.biases[f"b_hidden_{h}"]
        )

        for h in range(self.n_hl):
            weighted_sum = (
                weighted_sum
                + self.weights[f"w_hidden_{h}_output"] * hidden_outputs[h]
            )

        # Apply activation function
        if self.ol_af == "sigmoid":
            activated = autodiff.sigmoid(weighted_sum)
        elif self.ol_af == "relu":
            activated = autodiff.relu(weighted_sum)
        elif self.ol_af == "tanh":
            activated = autodiff.hyperbolic_tan(weighted_sum)
        elif self.ol_af == "linear":
            activated = autodiff.linear(weighted_sum)
        else:
            activated = weighted_sum  # default to linear

        output_expressions.append(activated)

        return output_expressions


