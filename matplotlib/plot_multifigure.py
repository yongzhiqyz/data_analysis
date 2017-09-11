import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(1000).reshape([1000,1])*1
x = np.sort(x, axis = 0)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_abs = np.abs(x)

plt.figure()
plt.plot(x, y_sin, label = "sin(x)")
plt.title('get the sin value')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()



plt.figure()
plt.plot(x, y_cos, label = "cos(x)")
plt.title('get the cos value')
plt.xlabel('X')
plt.ylabel('Y')

plt.figure()
plt.plot(x, y_abs, label = "abs(x)")
plt.title('get the abs value')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()




