'''
from: https://matplotlib.org/gallery/subplots_axes_and_figures/axhspan_demo.html
#sphx-glr-gallery-subplots-axes-and-figures-axhspan-demo-py
'''
import numpy as np
import matplotlib.pyplot as plt
t = np.arange(-1, 2, .01)
s = np.sin(2 * np.pi * t)
plt.plot(t, s)
plt.axhline(linewidth=8, color='#d62728')
plt.axhline(y=1)
plt.axvline(x=1)
plt.axvline(x=0, ymin=0.75, linewidth=8, color='#1f77b4')
plt.axhline(y=.5, xmin=0.25, xmax=0.75)
plt.axhspan(0.25, 0.75, facecolor='0.5', alpha=0.5)
plt.axvspan(1.25, 1.55, facecolor='#2ca02c', alpha=0.5)
plt.axis([-1, 2, -1, 2])
plt.show()
