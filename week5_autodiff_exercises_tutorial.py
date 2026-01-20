from autodiff import expression, multiplication, addition, constant


### Tutorial exercises ###
# 1. Create a single constant value.
def exercise_1():
    c = constant(5)
    return c


# 2. Create a variable named 'x'.
def exercise_2():
    x = expression('x')
    return x


# 3. Create a variable named 'x' and evaluate it at x = 10.
def exercise_3():
    x = expression('x')
    x_val = x.evaluate({ 'x': 10 })
    return x_val


# 4. Create the expression: 3 * x + 2
def exercise_4():
    x = expression("x")
    return 3*x + 2


# 5. Using the epxression created in exercise_4, differentiate it with respect to x, and with respect to y
def exercise_5():
    return exercise_4().diff({"x": -1}, "x")
