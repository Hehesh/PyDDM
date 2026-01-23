import sys
import pandas as pd
import numpy as np
from pyddm.preprocessing import fixations

def test_fixations_dfToList(df):
    return fixations.derasterize_fixations(df)

def test_fixations_listToDf(sequence):
    return fixations.rasterize_fixations(sequence)

def create_test_cases():
    # Create df with one trial's info
    # Create list by hand
    # Compare the two 
    # test_dict = {key for key in [1, 2, 3]}

    # Case 1
    test_1 = {'df': pd.DataFrame({
                    'fix_start': [0, 4],
                    'fix_end': [2, 7],
                    'fix_location': [1, 2]
                }),
              'list': np.array([1, 1, 1, 0, 2, 2, 2, 2])
    }

    # Case 2
    test_2 = {'df': pd.DataFrame({
                    'fix_start': [0],
                    'fix_end': [4],
                    'fix_location': [1]
                }),
              'list': np.array([1, 1, 1, 1, 1])
    }

    # Case 3
    test_3 = {'df': pd.DataFrame({
                    'fix_start': [0, 4, 8],
                    'fix_end': [1, 5, 9],
                    'fix_location': [1, 2, 1]
                }),
              'list': np.array([1, 1, 0, 0, 2, 2, 0, 0, 1, 1])
    }
    
    return [test_1, test_2, test_3]

if __name__ == '__main__':
    tests = create_test_cases()
    # Check for each test
    for test in tests:
        check_dfToList = np.array_equal(test_fixations_dfToList(test['df']), test['list'])
        check_listToDf = test_fixations_listToDf(test['list']).equals(test['df'])
        if not (check_dfToList and check_listToDf):
            print('Error: fixation processing methods not correct')
            sys.exit(1)

    print('Success: fixation processing methods correct')

    