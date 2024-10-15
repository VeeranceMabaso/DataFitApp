import numpy as np
import random
import math
from sklearn.metrics import r2_score
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


operators = ['+', '-', '*']

# List of available functions and terminals (variables/constant values)
#functions = [lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y, lambda x, y: x / y if y != 0 else 1]
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

# List of functions (replace lambda functions)
functions = [add, subtract, multiply]

terminals = ['X0', 'X1', 'random_constant']

# Helper Functions
def random_operator():
    return random.choice(functions)

def random_terminal():
    return random.choice(terminals)

def create_random_tree(depth=3):
    """Create a random tree of depth 'depth'."""
    if depth == 0:
        return random_terminal()  # Terminal node
    else:
        operator = random_operator()  # Internal node with an operator
        left = create_random_tree(depth - 1)  # Left subtree
        right = create_random_tree(depth - 1)  # Right subtree
        return (operator, left, right)  # Return a tree with an operator and two subtrees

def evaluate_tree(tree, X):
    """Evaluate a tree for input X."""
    if isinstance(tree, tuple):
        # Operator node
        operator, left, right = tree
        left_val = evaluate_tree(left, X)
        right_val = evaluate_tree(right, X)
        return operator(left_val, right_val)
    elif tree == 'X0':
        return X[0]
    elif tree == 'X1':
        return X[1]
    elif tree == 'random_constant':
        return random.uniform(-1, 1)

def calculate_fitness(tree, X, y):
    """Calculate the R-squared score of a tree based on training data."""
    predictions = []
    for i in range(len(X)):
        predictions.append(evaluate_tree(tree, X[i]))
    
    r_squared = r2_score(y, predictions)
    return r_squared

def mutate_tree(tree, depth=3):
    """Randomly mutate a tree by replacing a subtree or node."""
    if random.random() < 0.5:  # Mutate an internal node
        return create_random_tree(depth)
    else:  # Randomly replace a subtree
        if isinstance(tree, tuple):
            operator, left, right = tree
            if random.random() < 0.5:
                left = mutate_tree(left, depth - 1)
            else:
                right = mutate_tree(right, depth - 1)
            return (operator, left, right)
        else:
            return create_random_tree(depth)

def crossover_trees(tree1, tree2):
    """Perform crossover (subtree exchange) between two trees."""
    if isinstance(tree1, tuple) and isinstance(tree2, tuple):
        if random.random() < 0.5:
            return (tree1[0], tree1[1], tree2[2])
        else:
            return (tree2[0], tree1[1], tree2[2])
    else:
        return tree2  # Return tree2 if tree1 is terminal node

def select_parents(population, fitness):
    """Select two individuals based on their fitness using tournament selection."""
    tournament_size = 3
    parents = []
    for _ in range(2):
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness
        parents.append(tournament[0][0])  # Choose the best individual
    return parents

def evolve_population(population, fitness, mutation_rate=0.1, crossover_rate=0.7):
    """Evolve the population by selection, crossover, and mutation."""
    new_population = []
    for _ in range(len(population)):
        parent1, parent2 = select_parents(population, fitness)
        if random.random() < crossover_rate:
            child = crossover_trees(parent1, parent2)
        else:
            child = parent1  # No crossover, keep one of the parents
        
        if random.random() < mutation_rate:
            child = mutate_tree(child)
        
        new_population.append(child)
    
    return new_population

def genetic_programming(X, y, generations=50, population_size=500, tree_depth=3, mutation_rate=0.1, crossover_rate=0.7):
    """Main GP algorithm."""
    # Initialize random population
    population = [create_random_tree(depth=tree_depth) for _ in range(population_size)]
    best_tree = None
    best_fitness = -float('inf')

    for generation in range(generations):
        # Evaluate the fitness of the population
        fitness = [calculate_fitness(tree, X, y) for tree in population]
        
        # Find the best individual
        max_fitness = max(fitness)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_tree = population[fitness.index(max_fitness)]

        # Evolve the population for the next generation
        population = evolve_population(population, fitness, mutation_rate, crossover_rate)
        print(f"Generation {generation+1}/{generations} - Best Fitness: {best_fitness}")

    return best_tree

def tree_to_math_expr(tree):
    """Convert the tree into a mathematical expression in a string format."""
    if isinstance(tree, tuple):
        operator, left, right = tree
        left_expr = tree_to_math_expr(left)
        right_expr = tree_to_math_expr(right)
        
        # Map the function to the corresponding string operator
        if operator == functions[0]:  # Addition
            expr = f"({left_expr} + {right_expr})"
        elif operator == functions[1]:  # Subtraction
            expr = f"({left_expr} - {right_expr})"
        elif operator == functions[2]:  # Multiplication
            expr = f"({left_expr} * {right_expr})"
        elif operator == functions[3]:  # Division
            expr = f"({left_expr} / {right_expr})"
        return expr
    elif tree == 'X0':
        return 'X0'
    elif tree == 'X1':
        return 'X1'
    elif tree == 'random_constant':
        return f"{random.uniform(-1, 1):.4f}"
    
def plot_3d(X, y_truth, y_pred, title="Model Predictions"):
    """Plot 3D surfaces of ground truth and model predictions."""
    x0 = X[:, 0]
    x1 = X[:, 1]

    # Reshape the truth and prediction to match the X grid
    x0_range = np.linspace(-1, 1, 50)
    x1_range = np.linspace(-1, 1, 50)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Interpolate y_truth and y_pred over the grid
    grid_truth = griddata((x0, x1), y_truth, (X0, X1), method='cubic')
    grid_pred = griddata((x0, x1), y_pred, (X0, X1), method='cubic')
    
    fig = plt.figure(figsize=(12, 10))
    
    # Ground truth surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_xticks(np.arange(-1, 1.1, 0.5))
    ax1.set_yticks(np.arange(-1, 1.1, 0.5))
    ax1.set_title("Ground Truth")
    surf_truth = ax1.plot_surface(X0, X1, grid_truth, color='blue', alpha=0.5)
    ax1.scatter(x0, x1, y_truth, color='black', label="Data points")
    
    # Model prediction surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_xticks(np.arange(-1, 1.1, 0.5))
    ax2.set_yticks(np.arange(-1, 1.1, 0.5))
    ax2.set_title(title)
    surf_pred = ax2.plot_surface(X0, X1, grid_pred, color='red', alpha=0.5)
    ax2.scatter(x0, x1, y_pred, color='green', label="Predicted points")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Test with y_truth = X0**2 - X1**2 + X1 - 1
    X = np.random.uniform(-1, 1, (100, 2))  # Generating random data for X
    y = X[:, 0]**2 - X[:, 1]**2 + X[:, 1] - 1  # The target function
    
    best_tree = genetic_programming(X, y, generations=10)

    y_pred = np.array([evaluate_tree(best_tree, X[i]) for i in range(X.shape[0])])

    plot_3d(X, y, y_pred, title="Best Model Predictions")
    
    # # Convert the best tree into a mathematical expression
    # math_expr = tree_to_math_expr(best_tree)
    # print(f"Best Tree: {best_tree}")
    # print(f"Is tuple? {isinstance(best_tree, tuple)}")
    # print(f"Mathematical Expression of the best tree: {math_expr}")
    
    # # Calculate the fitness of the best tree
    # fitness_score = calculate_fitness(best_tree, X, y)
    # print(f"Fitness (R-squared score) of the best tree: {fitness_score}")
