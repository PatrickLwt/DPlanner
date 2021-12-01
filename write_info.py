import numpy as np 

a = np.array([[100.0], [50.]])
info = np.repeat(a, 23, axis=1)
np.savetxt('/home/liweiting/adult/data/info/budget.info', info, fmt='%f')