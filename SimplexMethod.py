"""
SIMPLEX METHOD
Based on Chapter 4 of Introduction to Operations Research (10th Edition)
by Hilier and Lieberman

Features Missing:
Decision Variables with Negative Bound
Unbounded Decision Variables
"""

import numpy as np 
# This are the options for printing
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Define error for input
class InputError(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

# This performs divisions for minimum ratio test 
def _ratios(a,b):
    r = []
    if len(a) == len(b):
        for i in np.arange(len(a)):
            if b[i] <= 0:
                r.append(np.Inf)
            else:
                r.append(a[i]/b[i])
    return np.array(r)

def simplex_method(Z_input, LHS_input, RHS_input, comparisons_input, goal, M = 100000000):

    # Check the goal
    if goal not in ['max','min']:
        raise InputError(goal, "Goal {goal} is not valid value. Use 'max' to maximize or 'min' to minimize.".format(goal=goal))
    # Check whether the inputs are numpy arrays
    for input_array in [Z_input, LHS_input, RHS_input]:
        if not isinstance(input_array, np.ndarray):
            raise InputError(input_array, "Array {input_array} is not in np.ndarray format.".format(input_array=input_array))
    # Try to convert type to float
        try:
            input_array.astype(float)
        except:
            raise InputError(input_array, "Array {input_array} is not in numeric format.".format(input_array=input_array))
    if len(Z_input) != LHS_input.shape[1]:
        raise InputError((len(Z_input), LHS_input.shape[1]), "Number of decision variables is not the same in objective function and left hand side arrays.")
    if len(comparisons_input) != LHS_input.shape[0]:
        raise InputError((len(comparisons_input), LHS_input.shape[0]), "Number of constraints is not the same as number of left hand sides.")
    if len(RHS_input) != LHS_input.shape[0]:
        raise InputError((len(RHS_input), LHS_input.shape[0]), "Number of right hand sides is not the same as number of left hand sides.")
    # Current implementation assumes that all decision variables are greater or equal than zero.
    num_variables = Z_input.shape[0] 
    num_constraints = RHS_input.shape[0]
    LHS = []
    # Adds one to the start of the objective
    Z = np.insert(-1 * Z_input, 0, 1)
    # Changes objective function in case of minimization problem
    if goal == 'min':
       Z = -1 * Z
    # Convert comparisons
    comparison_conversion = {'>=': '<=', '<=': '>='}
    comparisons = []
    RHS_input_index = 0
    for RHS_input_record in RHS_input: 
        # This flips the constraint if right side is negative
        if RHS_input_record < 0:
            LHS_input[RHS_input_index, ] = -1 * LHS_input[RHS_input_index, ]
            RHS_input[RHS_input_index] = -1 * RHS_input[RHS_input_index]
            comparisons.append(comparison_conversion[comparisons_input[RHS_input_index]])
        else:
            comparisons.append(comparisons_input[RHS_input_index])
        RHS_input_index += 1
    # Notify the users that Big M method will be used
    if '=' in comparisons or '>=' in comparisons:
        print('The Big M Method is being used. The value of M is {M}. \nChange the value of M in the input of this function if necessary.'.format(M=M))
    # Check whether any surplus variables need to be included
    surplus_constraints = sum([c in ['>='] for c in comparisons])
    # Identify constrains with artificial variables that needs to be substracted from the objective function
    artificial_rows = []    
    # This counts surplus variables
    surplus_index = 0 
    for constrain_index in np.arange(num_constraints):
        # Create a slack variables for all constrain types
        slack_temp = np.zeros(num_constraints)
        slack_temp[constrain_index] = 1
        # Appends slack variables to a constraint
        left_temp = np.insert(np.insert(LHS_input[constrain_index,:],LHS_input[constrain_index,:].shape[0], slack_temp), 0, 0)
        if comparisons[constrain_index] == '<=':
            # Surplus variables not used here
            surplus_temp = np.zeros(surplus_constraints)
            left_temp = np.insert(left_temp, left_temp.shape[0], surplus_temp)
            LHS.append(left_temp) 
            # Appends slack variables to the objective function
            Z = np.append(Z, 0)
        elif comparisons[constrain_index] == '>=':
            # Appends surplus variables to constrains
            surplus_temp = np.zeros(surplus_constraints)
            surplus_temp[surplus_index] = -1
            left_temp = np.insert(left_temp, left_temp.shape[0], surplus_temp)
            LHS.append(left_temp) 
            # Appends slack variable to the objective function with M penalty
            Z = np.append(Z, M)
            # Move by one surplus variable
            surplus_index += 0
            # Store which constraint contains artificial variable
            artificial_rows.append(constrain_index)
        elif comparisons[constrain_index] == '=':
            # Appends zeros for surplus variables
            surplus_temp = np.zeros(surplus_constraints)
            left_temp = np.insert(left_temp, left_temp.shape[0], surplus_temp)
            LHS.append(left_temp) 
            # Appends penalization to the objective function. 
            # Not sure how to set it up the best
            Z = np.append(Z, M)
            # Store which constraint contains artificial variable
            artificial_rows.append(constrain_index)
    # Add surplus variables to the objective function
    Z = np.append(Z, np.zeros(surplus_constraints))
    # Format input into right shape to support calculations
    LHS = np.array(LHS)
    #print('LHS', LHS)
    Z = np.reshape(Z, (1, len(Z)))
    #print('Z', Z)
    RHS = np.insert(RHS_input, 0, 0)
    #print('RHS', RHS)
    # Create a tableau from equations
    tableau = np.append(np.append(Z, LHS, axis=0), np.reshape(RHS, (len(RHS),  1)), axis=1)
    tableau = tableau.astype('float')
    print('Starting Tableau')
    print(tableau)
    # Convert the objective function to the correct format
    if len(artificial_rows) != 0:
        for row in artificial_rows:
            tableau[0,] = tableau[0,] - (tableau[row,] * M)
    # Perform the simplex algorithm itirations
    j = 0
    while  np.array([x < 0 for x in tableau[0,-(tableau.shape[1] - 1):-1]]).any():
        j += 1
        print('Run ', j)
        print('Not Optimal')
        # Find the pivot column by searching for the minimum 
        pivot_column = np.argmin(tableau[0,:-1])
        #print('Pivot Column ', pivot_column)
        # Check whether Z is not unbounded
        if np.array( [x == 0 for x in tableau[1:, pivot_column]]).all():
            print('Z is unbounded!')
            break
        # Perform minumum ratio test
        # In case of ties, the first index is returned. 
        pivot_row = np.argmin(_ratios(tableau[1:, (tableau.shape[1] - 1)], tableau[1:, pivot_column])) + 1
        #print( 'Pivot Row ', pivot_row)
        # Solve new for new solution
        pivot_number = tableau[pivot_row, pivot_column]
        #print('Pivot Number ', pivot_number)
        new_pivot_row = tableau[pivot_row,:]/pivot_number
        #print('Pivot Row Data', new_pivot_row)
        for i in np.arange(tableau.shape[0]):
            if i == pivot_row:
                tableau[i,] = new_pivot_row
            else:
                #print('Coeff ', i,tableau[i, pivot_column])
                tableau[i,] = tableau[i,] - (new_pivot_row * tableau[i, pivot_column])
        print('Tableau', j)
        print(tableau)
        # Process final tableau to obtain results
    results = {}
    if goal == 'max':
        results['Goal'] = 'Maximize'
        results['Z_Optimal'] = tableau[0, -1]
        results['Final_Tab'] = tableau
    elif goal == 'min':
        results['Goal'] = 'Minimize'
        results['Z_Optimal'] = -1 * tableau[0, -1]
        results['Final_Tab'] = tableau
    return results
