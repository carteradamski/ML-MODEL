# imports
import autodiff
import random
import pandas as pd

# linear regression using the autodiff library
class mdst_linear_reg():
    """Multivariable linear regression (MLR) for MDST project, "Building ML Models."
    
    Attributes: 
        mx: Slopes of multivariate linear regression
        yint: Y-intercept of multivariable linear regression
    
    Methods: 
        fit(X_fit, y_fit): Using gradient descent, tune slope attributes & y-intercept over n_epochs.
        predict(X_pred): Using pre-tuned slope, y-intercept, estimate theoretical y-values for input X_pred.
    
    """
    def __init__(self):
        """Initializer function. Create slope and y-intercept attributes, blank.
        
        Args: 
        
        Returns: 

        """
        self.mx = {}
        self.yint = 0
        
    def fit(self, X_fit, y_fit, n_epochs, learning_rate=0.01):
        """Fit MLR on training data defined by X_fit, y_fit over n_epochs.
        
        Args:
            X_fit (pd.dataframe): x values / features in training data. 
                Number of columns defines num. slopes in mx attribute, number of rows MUST be same as length of y_fit.
                
            y_fit (list-like): y values / target in training data. 
                Number of values MUST be same as number of rows in X_fit.
                
            n_epochs (int): number of training epochs, or number of times MLR iterates through the full x-y training set.
                
            learning_rate (float): step size for gradient descent updates.
        """
        # step 1: populate the mx dictionary with X_columns number of slopes
        for i in range(len(X_fit.columns)):
            self.mx['m' + str(i)] = random.uniform(-1, 1) # NOT expressions; these are numeric values.
        
        # Initialize y-intercept
        self.yint = random.uniform(-1, 1)
        
        # step 2: Set up autodiff expressions for parameters
        # Create expression variables for each slope and y-intercept
        slope_vars = {}
        for i in range(len(X_fit.columns)):
            slope_vars['m' + str(i)] = autodiff.expression('m' + str(i))
        yint_var = autodiff.expression('yint')
        
        # step 3: Gradient descent training
        for epoch in range(n_epochs):
            
            # Initialize gradients for this epoch
            # Gradients is a dictionary to store the gradients for each parameter (slopes and y-intercept)
            # At the conclusion of this epoch, these will be used to update the parameters.
            gradients = {}
            for key in self.mx.keys():
                gradients[key] = 0
            gradients['yint'] = 0
            
            # Process each training example
            for idx in range(len(X_fit)):
                
                # Prediction expression: single prediction for each row of the training data.
                prediction_expr = 0
                # TODO: complete prediction_expr construction using slope_vars, X_fit, and yint_var values.
                # Hint: Build the prediction expression using the formula of a line: y_pred = m0*x0 + m1*x1 + ... + yint
                for i in range(len(X_fit.columns)):
                    prediction_expr +=  slope_vars['m' + str(i)] * X_fit.iloc[idx, i]
                prediction_expr += yint_var

                # Create loss expression: (y_true - y_hat)^2
                y_true = y_fit[idx] 
                # Use manual squaring to avoid domain error with math.log(): (a-b)^2 = (a-b)*(a-b)
                diff_expr = y_true - prediction_expr
                loss_expr = diff_expr * diff_expr
                
                # Current parameter values for evaluation
                current_values = dict(self.mx)
                current_values['yint'] = self.yint
                
                # Compute gradients
                for key in self.mx.keys():
                    # TODO: compute gradient for each slope parameter using .diff() with the overall loss expression, loss_expr.
                    # Hint: call diff with (current_values, key) to select the correct parameter. Make sure to add this gradient to gradients[key].
                    gradients[key] += loss_expr.diff(current_values, key)

                
                # TODO: compute gradient for y-intercept using .diff() with loss_expr, with respect to 'yint'.
                gradients['yint'] += loss_expr.diff(current_values, 'yint')
                
            
            # Update parameters using gradient descent
            for key in self.mx.keys():
                self.mx[key] -= learning_rate * gradients[key]
                # TODO: complete parameter update step using learning_rate and gradients.
                # Hint: you have self.mx populated with slope values -- update each slope here using the computed gradient values.
            # TODO: Once this is complete, do so for the y-intercept as well.
            self.yint -= learning_rate * gradients['yint']

    def predict(self, X_pred):
        """Generate prediction / hypothetical y based on X input, using pre-trained slope and y-intercept.

        Args:
            X_pred (pd.dataframe): x values / features in prediction data.
            
        Returns:
            list: predicted y values for each row in X_pred.
        """   
        predictions = []
        
        for idx in range(len(X_pred)):
            # y = m0*x0 + m1*x1 + ... + yint
            prediction = self.yint
            
            # TODO: complete prediction calculation. 
            # Hint: you now have self.mx populated with slope values -- you don't need to use autodiff expressions here!  
            # (Do something with the variable "prediction"... the first step of computation is done for you)

                    
            predictions.append(prediction)
            for i in range(len(X_pred.columns)):
                prediction += self.mx['m' + str(i)] * X_pred.iloc[idx, i]
                predictions.append(prediction)
            
        
        return predictions

# Once you're done with the actual coding portions, run this code to demonstrate your model.
# Courtesy of Github Copilot.
def main():
    """Demonstration of the linear regression model using gradient descent with autodiff."""
    print("=== MDST Linear Regression with Autodiff Demo ===\n")
    
    # Create synthetic data for demonstration
    import numpy as np
    
    # Generate synthetic dataset: y = 2*x1 + 3*x2 + 1 + noise
    np.random.seed(42)  # for reproducible results
    n_samples = 100
    
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    noise = np.random.normal(0, 0.1, n_samples)
    y = 2 * x1 + 3 * x2 + 1 + noise
    
    # Create DataFrame
    X = pd.DataFrame({
        'feature1': x1,
        'feature2': x2
    })
    
    print("Created synthetic dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {X.columns.tolist()}")
    print(f"  True parameters: slope1=2, slope2=3, intercept=1")
    print()
    
    # Split data for training and testing
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()
    
    # Create and train model
    model = mdst_linear_reg()
    
    print("Training model with gradient descent...")
    model.fit(X_train, y_train, n_epochs=1000, learning_rate=0.01)
    print()
    
    # Make predictions on test set
    predictions = model.predict(X_test)
    
    # Calculate test error
    test_error = sum((y_test[i] - predictions[i])**2 for i in range(len(y_test))) / len(y_test)
    print(f"Test Mean Squared Error: {test_error:.6f}")
    
    # Show some predictions vs actual
    print("\nSample predictions vs actual:")
    print("Predicted | Actual")
    print("-" * 20)
    for i in range(min(10, len(predictions))):
        print(f"{predictions[i]:8.3f} | {y_test[i]:6.3f}")
    
    print(f"\nModel successfully trained using autodiff for gradient computation!")
if __name__ == "__main__":
     main()
