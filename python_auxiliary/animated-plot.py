#%%

# https://matplotlib.org/examples/animation/dynamic_image.html
# https://stackoverflow.com/questions/17212722/matplotlib-imshow-how-to-animate
# http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
# https://stackoverflow.com/questions/17853680/animation-using-matplotlib-with-subplots-and-artistanimation
# https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

fig = plt.figure()
ax = plt.axes()
# ax = plt.axes(xlim=(-5, 5), ylim=(-5, 5))
# ax1=fig.add_subplot(1,2,1)
# ax2=fig.add_subplot(1,2,2)

ims=[]
for time in range(10):
    im = ax.imshow(np.random.rand(10,10))
    line, = ax.plot(np.random.uniform(0,10,10)[:time])
    ims.append([im, line])

#run animation
ani = animation.ArtistAnimation(fig,ims, interval=50,blit=False)
plt.show()
# ani.save('tmp/tmp.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# HTML(ani.to_html5_video())

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r.-')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()
ani.save('tmp/tmp.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# ani.save('tmp/tmp.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# HTML(ani.to_html5_video())
