### WEEK 6 OF BUILDING ML MODELS MDST PROJECT ###
from autodiff import *  # "import *" lets us use all functions defined in autodiff.py... no need to call "autodiff.function"
import math

### INSTRUCTIONS ###
# Below are a series of expressions you need to create using the autodiff library.
# The first 4 (and exercise 8) are filled out for you as a tutorial.
# In the main() section of this file, there are some test cases -- if you pass them all, you've completed this exercise!


### Tutorial exercises ###
# 1. Create a single constant value.
def exercise_1():
    return constant(5)
    # with constants -- set the value initially. no need to set a name or define values later.


# 2. Create a variable named 'x'.
def exercise_2():
    return expression("x")
    # with variables -- set the name. Values are determined later (see next example).


# 3. Create a variable named 'x' and evaluate it at x = 10.
def exercise_3():
    x = expression("x")
    return x.eval({"x": 10})
    # evaluate() takes a dictionary of variable names and their corresponding values.


# 4. Create the expression: 3 * x + 2
def exercise_4():
    x = expression("x")
    return 3 * x + 2
    # You can use standard arithmetic operators (+, -, *, /, **) to build expressions.
    # This is another way to work with the constants -- you can use them directly in expressions (no need to define constants using the constants class).


# 5. Using the epxression created in exercise_4, differentiate it with respect to x
def exercise_5():
    return exercise_4().diff({"x": -1}, "x")
    # diff() takes a dictionary of variables names and their corresponding values, and the variable name which you differentiate with respect to


### The rest of these exercises are for you to complete! ###
# 6. Create the expression: (x^2 + 2x + 1)
def exercise_6():
    x = None  # TODO: replace None with your code for the variable 'x'.
    return None  # TODO: replace None with a completed expression.


# 7. Create the expression: x^3 + 3y^2 + 3x + 1. Evaluate it at x = 10, y = 5.
def exercise_7():
    x = None  # TODO: replace None with your code for the variable 'x'.
    y = None  # TODO: replace None with your code for the variable 'y'.
    return None  # TODO: replace None with a completed expression, evaluated at x = 10, y = 5.


# 8. Create the expression: sin(x)^2 + cos(x)^2. Evaluate it at any x you want.
def exercise_8():
    x = None  # TODO: replace None with your code for the variable 'x'.
    return None  # TODO: replace None with a completed expression, evaluated at any x you want.


# 9. Create the expression: x^2 + 3x + 2. Compute its derivative with respect to x, and evaluate the derivative at x = 5.
def exercise_9():
    x = None  # TODO: replace None with your code for the variable 'x'.
    expr = None  # TODO: replace None with the expression x^2 + 3x + 2.
    return None  # TODO: replace None with the derivative of 'expr' with respect to 'x', evaluated at x = 5.


# 10. Create the expression: y^x, evaluate its derivative with respect to x at x = 2, y = 3.
def exercise_10():
    x = None  # TODO: replace None with your code for the variable 'x'.
    y = None  # TODO: replace None with your code for the variable 'y'.
    return None  # TODO: replace None with a completed expression, differentiated with respect to x, and evaluated at x = 2, y = 3.


# 11. Create the expression: ln(x^2 + 1). Evaluate its derivative with respect to x at x = 1.
def exercise_11():
    x = None  # TODO: replace None with your code for the variable 'x'.
    return None  # TODO: replace None with a completed expression, differentiated with respect to x, and evaluated at x = 1.


### MAIN FUNCTION FOR TESTS ###
if __name__ == "__main__":
    # Test cases to verify your solutions -- if you pass all tests, you've completed the exercise!
    # Note -- if you get the error "NoneType object has no attribute eval" -- you probably forgot to replace a "None" with your code!
    assert exercise_1().eval({}) == 5, f"{exercise_1.__name__} failed"
    assert exercise_2().eval({"x": 7}) == 7, f"{exercise_2.__name__} failed"
    assert exercise_3() == 10, f"{exercise_3.__name__} failed"
    assert exercise_4().eval({"x": 3}) == 11, f"{exercise_4.__name__} failed"
    assert exercise_5() == 3, f"{exercise_5.__name__} failed"
    assert exercise_6().eval({"x": 4}) == 25, f"{exercise_6.__name__} failed"
    assert exercise_7() == 1106, f"{exercise_7.__name__} failed"
    assert math.isclose(exercise_8(), 1.0), f"{exercise_8.__name__} failed"
    assert exercise_9() == 23, f"{exercise_9.__name__} failed"
    assert math.isclose(exercise_10(), 9.88751059801), f"{exercise_10.__name__} failed"
    assert math.isclose(exercise_11(), 1.0), f"{exercise_11.__name__} failed"
    print("All tests passed!")
