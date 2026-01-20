### AUTOMATIC DIFFERENTIATION -- FUNCTIONS ###
# Isaac Heitmann
# University of Michigan Data Science Team
# For MDST Fall '25 Project "Building ML Models"

### Description ###
# Contains classes -- each class represents a mathematical expression.
# Primary use is automatic differentiation within a neural network; however, it has a wider range of utilities.

### Dev notes
# Sep 6 2025: Created base class

import math

### Expressions ### 

class expression:
    """Base expression class. Represents a singular variable, like x.
    """
    def __init__(self, name):
        """Init function.
            Args: 
                Name: name of the variable, like "x."
        """
        self.name = name
        
    def eval(self, values):
        """Evaluation function, returns value of expression/variable given value parameters.

        Args:
            values (dict): a dicitonary of variable names:values.

        Returns:
            float representing value of variable at given evaluation. 
        """
        return values[self.name]

    def diff(self, values, diffto):
        """Differentiate variable / expression with the current values. 

        Args:
            values (dict): a dicitonary of variable names:values.
            diffto (string): variable with respect to which differentiation happens.
        
        Returns: 
            1 or 0 -- depends if this is the variable we're differentiating with respect to. 
        """
        return 1 if diffto == self.name else 0

    # the following are all overrides: (all other subsequent classes inherit)
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return addition(self, other)
    
    def __radd__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return addition(other, self)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return subtraction(self, other)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return subtraction(other, self)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return multiplication(self, other)
    
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return multiplication(other, self)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return division(self, other)
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return division(other, self)
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return exponent(self, other)
    
    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            other = constant(other)
        return exponent(other, self)

class constant(expression):
    """Constant expression class. Represents a constant value, like 2 or 3.14.
    """
    def __init__(self, value):
        """Init function.
            Args: 
                value: numeric value of the constant.
        """
        self.value = value
        
    def eval(self, values):
        """Evaluation function, returns the constant value.

        Args:
            values (dict): a dicitonary of variable names:values (not used for constants).

        Returns:
            float representing the constant value. 
        """
        return self.value

    def diff(self, values, diffto):
        """Differentiate constant with respect to any variable. 

        Args:
            values (dict): a dicitonary of variable names:values (not used).
            diffto (string): variable with respect to which differentiation happens (not used).
        
        Returns: 
            0 -- derivative of a constant is always 0.
        """
        return 0
    
class addition(expression):
    def __init__(self, first, second):
        """Addition of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second
        
    def eval(self, values):
        return self.first.eval(values) + self.second.eval(values)

    def diff(self, values, diffto):
        return self.first.diff(values, diffto) + self.second.diff(values, diffto)

class subtraction(expression):
    def __init__(self, first, second):
        """Subtraction of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second
        
    def eval(self, values):
        return self.first.eval(values) - self.second.eval(values)

    def diff(self, values, diffto):
        return self.first.diff(values, diffto) - self.second.diff(values, diffto)

class multiplication(expression):
    def __init__(self, first, second):
        """Multiplication of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second
        
    def eval(self, values):
        return self.first.eval(values) * self.second.eval(values)

    def diff(self, values, diffto):
        # product rule, u'(x) * v(x) + u(x) * v'(x)
        return self.first.diff(values, diffto) * self.second.eval(values) + self.first.eval(values) * self.second.diff(values, diffto)
    
class division(expression):
    def __init__(self, first, second):
        """Division of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second
        
    def eval(self, values):
        return self.first.eval(values) / self.second.eval(values)

    def diff(self, values, diffto):
        # quotient rule: (v(x)u'(x) - v'(x)u(x)) / (v(x) ^ 2)
        return (self.second.eval(values) * self.first.diff(values, diffto) - self.second.diff(values, diffto) * self.first.eval(values)) / (self.second.eval(values) ** 2)
    
class exponent(expression):
    def __init__(self, first, second):
        """Exponent of two expressions.

        Args:
            first (expression): an expression
            second (expression): another expression
        """
        self.first = first
        self.second = second
        
    def eval(self, values):
        return self.first.eval(values) ** self.second.eval(values)

    def diff(self, values, diffto):
        # power rule -- generalized
        return self.eval(values) * (natlog(self.first) * self.second).diff(values, diffto)
    
### Functions -- single input###
# These MUST account for the chain rule.
# They also aren't called via overrides, meaning that they should handle casts to constant if need be.
class natlog(expression):
    def __init__(self, first):
        """Natural log of an expression.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        return math.log(self.first.eval(values))

    def diff(self, values, diffto):
        # chain rule
        return 1 / self.first.eval(values) * self.first.diff(values, diffto)

#class sine
class sine(expression):
    def __init__(self, first):
        """sine of an expression.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        return math.sin(self.first.eval(values))

    def diff(self, values, diffto):
        # chain rule
        # debatable to use math cos here instead of native. Should just have same effect but faster.
        return math.cos(self.first.eval(values)) * self.first.diff(values, diffto)

#class cosine
class cosine(expression):
    def __init__(self, first):
        """cosine of an expression.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        return math.cos(self.first.eval(values))

    def diff(self, values, diffto):
        # chain rule
        return -1 * math.sin(self.first.eval(values)) * self.first.diff(values, diffto)

#class tangent
class tangent(expression):
    def __init__(self, first):
        """tangent of an expression.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        return math.tan(self.first.eval(values))

    def diff(self, values, diffto):
        # chain rule
        return 1 / math.cos(self.first.eval(values)) ** 2 * self.first.diff(values, diffto)

#class tanh (tanh)
class hyperbolic_tan(expression):
    def __init__(self, first):
        """tanh. activation function

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        return math.tanh(self.first.eval(values))

    def diff(self, values, diffto):
        return 1 / math.cosh(self.first.eval(values)) ** 2 * self.first.diff(values, diffto)

#class sigmoid
class sigmoid(expression):
    def __init__(self, first):
        """Sigmoid activation function. 

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        return 1 / (1 + math.e ** (-1 * self.first.eval(values)))

    def diff(self, values, diffto):
        # chain rule
        return math.e ** self.first.eval(values) / (math.e ** self.first.eval(values) + 1) ** 2 * self.first.diff(values, diffto)

#class relu
class relu(expression):
    def __init__(self, first):
        """Rectified linear unit activation function.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        return max(self.first.eval(values), 0)

    def diff(self, values, diffto):
        # TEST: change this to just 1/0. See if it breaks an intermediate test. 
        return self.first.diff(values, diffto) if self.eval(values) > 0 else 0

#class relu_leaky
class leaky_relu(expression):
    def __init__(self, first):
        """Leaky rectified linear unit activation function.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        evaluated = self.first.eval(values)
        return  evaluated if evaluated > 0 else 0.001 

    def diff(self, values, diffto):
        # TEST: change this to just 1/0. See if it breaks an intermediate test. 
        return self.first.diff(values, diffto) if self.eval(values) > 0 else 0.001 * self.first.diff(values, diffto)

#class linear_activation
class linear(expression):
    def __init__(self, first):
        """Linear activation function.

        Args:
            first (expression): an expression
        """
        self.first = constant(first) if isinstance(first, (int, float)) else first
        
    def eval(self, values):
        return self.first.eval(values)

    def diff(self, values, diffto):
        # chain rule
        return self.first.diff(values, diffto)

### TESTING ###
def basictests():
    ### Basic tests -- only expressions ###
    vals = {
        'x':10, 
        'y':5,
        'z':-5
    }
    
    # test expression basic
    print("Expression Basic Test: ")
    x = expression('x')
    y = expression('y')
    print("value of x is 10: ", x.eval(vals) == 10)
    print("value of y is 5: ", y.eval(vals) == 5)
    print("diff of x, d/dx is 1: ", x.diff(vals, 'x') == 1)
    print("diff of y, d/dx is 0: ", y.diff(vals, 'x') == 0)
    
    # test addition basic
    print("\nAddition Basic Test: ")
    myadd = x + y
    print('value of x + y is 15: ', myadd.eval(vals) == 15)
    print('diff of x + y, d/dx is 1: ', myadd.diff(vals, 'x') == 1)
    print('diff of x + y, d/dy is 1: ', myadd.diff(vals, 'y') == 1)
    
    # test multiplication basic 
    print("\nMultiplication Basic Test: ")
    mymul = x * y
    mymultwo = x * x
    # the following assess parenthesis
    mymulthree = (x + y) * (x + y)
    mymulfour = x + y * x + y
    print('value of x * y is 50: ', mymul.eval(vals) == 50)
    print('diff of x * y, d/dx is 5: ', mymul.diff(vals, 'x') == 5)
    print('diff of x * y, d/dy is 10: ', mymul.diff(vals, 'y') == 10)
    print('diff of x * x, d/dx is 20: ', mymultwo.diff(vals, 'x') == 20)
    print('diff of (x + y) * (x + y), d/dx is 30: ', mymulthree.diff(vals, 'x') == 30)
    print('diff of x + y * x + y, d/dx is 6: ', mymulfour.diff(vals, 'x') == 6)
    
    # test division
    print("\nDivision Basic Test: ")
    mydiv = x / y
    mydivtwo = x / x
    # the following assess parenthesis
    mydivthree = y / x
    mydivfour = x + y / x + y
    print('value of x / y is 2: ', mydiv.eval(vals) == 2)
    print('diff of x / y, d/dx is 0.2: ', mydiv.diff(vals, 'x') == 0.2)
    print('diff of x / y, d/dy is -0.4: ', mydiv.diff(vals, 'y') == -0.4)
    print('diff of x / x, d/dx is 0: ', mydivtwo.diff(vals, 'x') == 0)
    print('diff of y / x, d/dx is -0.05: ', mydivthree.diff(vals, 'x') == -0.05)
    print('diff of x + y / x + y, d/dx is 0.95: ', mydivfour.diff(vals, 'x') == 0.95)
    
    # test exponent
    print("\nExponent Basic Test: ")
    myexp = exponent(x, y)
    myexptwo = exponent(x, constant(2))
    print('value of x ^ y is 100,000: ', myexp.eval(vals) == 100000)
    print('diff of x ^ y, d/dx is 50,000: ', myexp.diff(vals, 'x') == 50000)
    print('diff of x ^ 2, d/dx is 20: ', myexptwo.diff(vals, 'x') == 20)
    

    
    # test constants
    print("\nConstant Test: ")
    c = constant(5)
    print('value of constant 5 is 5: ', c.eval(vals) == 5)
    print('diff of constant 5, d/dx is 0: ', c.diff(vals, 'x') == 0)
    
    # test mixed expressions with constants
    print("\nMixed Constant Test: ")
    mixed1 = x + 2  # x + constant
    mixed2 = 3 * x  # constant * x
    mixed3 = x / 4  # x / constant
    mixed4 = 2 - x  # constant - x
    print('value of x + 2 is 12: ', mixed1.eval(vals) == 12)
    print('diff of x + 2, d/dx is 1: ', mixed1.diff(vals, 'x') == 1)
    print('value of 3 * x is 30: ', mixed2.eval(vals) == 30)
    print('diff of 3 * x, d/dx is 3: ', mixed2.diff(vals, 'x') == 3)
    print('value of x / 4 is 2.5: ', mixed3.eval(vals) == 2.5)
    print('diff of x / 4, d/dx is 0.25 ', mixed3.diff(vals, 'x') == 0.25)
    print('value of 2 - x is -8: ', mixed4.eval(vals) == -8)
    print('diff of 2 - x, d/dx is -1: ', mixed4.diff(vals, 'x') -1)
    
    # test power operator
    print("\nPower Operator Test: ")
    power1 = x ** 2  # x^2
    power2 = 2 ** x  # 2^x
    print('value of x ** 2: ', power1.eval(vals) == 100)
    print('diff of x ** 2, d/dx is 20: ', power1.diff(vals, 'x') == 20)
    print('value of 2 ** x is 1024: ', power2.eval(vals) == 1024)
    print('diff of 2 ** x, d/dx is 709.782712893: ', round(power2.diff(vals, 'x'), 3) == 709.783)
    
def intermediatetests():
    pass
    
    ### Intermediate tests -- expressions and constants added complexity ###

def main():
    basictests()
    
if __name__ == "__main__":
     main()
