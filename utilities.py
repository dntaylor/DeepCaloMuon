import os
import errno

def python_mkdir(d):
    '''A function to make a unix directory as well as subdirectories'''
    try:
        os.makedirs(d)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(d):
            pass
        else: raise
