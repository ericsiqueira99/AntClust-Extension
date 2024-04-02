from pathlib import Path
import sys
path = Path(__file__)
sys.path.append(str(path.parent.parent))

from Ant import Ant
import time as time
from goto import goto, comefrom, label
if __name__ == '__main__':

    label .someLabel
    print('some code')

    goto .someLabel

    ant = Ant([1, 2], 0)
    num_iter = 2000000
    n = num_iter

    start_nolock = time.time()
    while n:
        n = n - 1
        value = ant.label
    time_nolock = time.time() - start_nolock

    n = num_iter
    start_lock = time.time()
    while n:
        n = n - 1
        value = ant.get_label()
    time_lock = time.time() - start_lock

    print(time_nolock)
    print(time_lock)
