#%% Import Packages

import numpy as np
import pandas as pd
from tqdm import tqdm #Progress Bar
import sys 
import time
import cProfile, pstats



#%% functions

####################################################################################

def timelog(phase, name, globs, print_ = False):
    
    '''
    Keep track of time, saving the results to a list within dictionary
    (New values are only added while previous values are kept)
    
    Parameters
    ----------
    phase : str
        'start' or 'end'.
    name : str
        name of the key.
    print_ : TYPE, optional
        print the elapsed time. works only if 'phase' = 'end'. The default is False.

    Returns
    -------
    None.
    '''
    
    global start, end, timelog_dict
    
    if 'timelog_dict' not in globs:
        globs['timelog_dict'] = {}
    
    if phase == "start":
        globs['start'] = time.time()
    elif phase == "end":
        elapsedTime  = (time.time()-globs['start'])
        if name in globs['timelog_dict']:
            globs['timelog_dict'][name].append(elapsedTime)
        else:
            globs['timelog_dict'][name] = [elapsedTime]
        if print_:
            print(f'The elapsed time is {elapsedTime} seconds', flush=True)

# Usage:
#timelog("start", "step1", globals())
#timelog("end", "step1", globals())

####################################################################################

def stop():
    
    '''
    Stoping the execution at a desired XXX

    Raises
    ------
    Exception
        raises a conceived error.

    Returns
    -------
    None.
    '''
    
    raise Exception("intended error")

####################################################################################

#problem - it doesnt effect original file globals

def setGlobal(variable, new_variable, globs):
    
    '''
    Create a global variable with a copy of local variable's value'
    Meant for debugging, easy to track specific variables located inside functions

    Parameters
    ----------
    variable : variable (any type)
        local variable.
    new_variable : str
        the name of the new global variable.
    globs : globals()

    Returns
    -------
    None.

    '''
    try:
        globs[new_variable] = variable.copy()
    except AttributeError:
        globs[new_variable] = variable
    
# Usage:
#setGlobal(B, "B_global", globals())

####################################################################################

def profiler(phase, dump = False, print_stats = False):
    
    '''
    profiling to track resources usage
    
    Parameters
    ----------
    phase : str
        'start' or 'end'.
    dump : boolean, optional
        save the log to a file named 'stats.prof'. The default is False.
    print_stats : boolean, optional
        print the log. The default is False.

    Returns
    -------
    None.
    '''
    
    if phase == 'start':
        profiler = cProfile.Profile()
        profiler.enable()
    elif phase == 'end':
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        
        if print_stats:
            stats.print_stats()
    
        if dump:
            stats.dump_stats('stats.prof')

# Usage:
#profiler(phase = 'start')
#profiler(phase = 'end', dump = False, print_stats = False)

####################################################################################
  
def monitorObjects(globs):
    
    '''
    Document globals objects sizes, keep the record in a list
    Meant to catch growing objects

    Returns
    -------
    None.
    '''
    
    if 'objects' not in globs:
        globs['objects']={}
    
    for name,obj in globs.items():
        value = sys.getsizeof(obj)/ 1024
        if name in globs['objects']:
            globs['objects'][name].append(value)
        else:
            globs['objects'][name] = [value]

# Usage:
#monitorObjects()

####################################################################################

def compare_matrices(df1, df2): # compare two networks, for debugging | messy
    
    matrices_test = False
    matrices_T = False
    matrices_sort = False
    matrices_diff = False
    shapes = None
    sums = None
    
    arr1 = df1.to_numpy()
    arr2 = df2.to_numpy()

    shape1, shape2 = arr1.shape, arr2.shape
    shapes = (shape1, shape2)
    sums = (int((arr1>0).sum()), int((arr2>0).sum()))
    
    if np.array_equal(arr1, arr2):
        matrices_test = True
        return matrices_test, matrices_T, matrices_sort, matrices_diff, shapes, sums
    
    # try with transposing
    if shape1 != shape2:
        if shape1 == shape2[::-1]:
            arr2 = arr2.T
            matrices_T = True
            
            if np.array_equal(arr1, arr2):
                matrices_test = True
                return matrices_test, matrices_T, matrices_sort, matrices_diff, shapes, sums
        
        else:
            matrices_diff = True
            return matrices_test, matrices_T, matrices_sort, matrices_diff, shapes, sums
    
    # try with sorting
    df1.sort_index(axis=0, inplace=True)
    df1.sort_index(axis=1, inplace=True)
    df2.sort_index(axis=0, inplace=True)
    df2.sort_index(axis=1, inplace=True)
    
    arr1 = df1.to_numpy()
    arr2 = df2.to_numpy()
    
    if matrices_T:
        arr2 = arr2.T
        
    if np.array_equal(arr1, arr2):
        matrices_test = True
        matrices_sort = True
        return matrices_test, matrices_T, matrices_sort, matrices_diff, shapes, sums
    
    return matrices_test, matrices_T, matrices_sort, matrices_diff, shapes, sums

# Manual:
# net = 'A_HP_004'
# df1 = networks[net]['Network_Layers'][1]['Adjacency_Matrix'].copy()
# df2 = networks[ecomplab_duplicates[net][0]]['Network_Layers'][ecomplab_duplicates[net][1]]['Adjacency_Matrix'].copy()

####################################################################################