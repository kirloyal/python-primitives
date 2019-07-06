#%%

# https://matplotlib.org/examples/animation/dynamic_image.html
# https://stackoverflow.com/questions/17212722/matplotlib-imshow-how-to-animate
# http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
# https://stackoverflow.com/questions/17853680/animation-using-matplotlib-with-subplots-and-artistanimation
# https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
# https://brushingupscience.com/2016/06/21/matplotlib-animations-the-easy-way/

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
    line, = ax.plot(np.random.uniform(0,9,10)[:time], c=np.random.uniform(0,1,3))
    scat = ax.scatter(np.random.uniform(0,9,100), np.random.uniform(0,9,100), c=np.random.rand(100,4), s=np.random.uniform(0,9,100))
    ims.append([im, line, scat])

#run animation
ani = animation.ArtistAnimation(fig, ims, interval=50,blit=False)
plt.show()
# ani.save('tmp/tmp.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# HTML(ani.to_html5_video())

#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
# im = ax.imshow(np.random.rand(10,10))
im = ax.imshow(np.zeros((10,10)), vmin=0, vmax=1)
# https://matplotlib.org/api/image_api.html
ln, = plt.plot([], [])
# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_color
scat = plt.scatter([],[])
# https://matplotlib.org/3.1.0/api/collections_api.html#matplotlib.collections.PathCollection

def init():
    im.set_data(np.random.rand(10,10))
    ln.set_data([], [])
    return im, ln, scat

def update(frame):
    im.set_data(np.random.rand(10,10))
    ln.set_data(np.arange(frame), np.random.uniform(0,9,10)[:frame])
    ln.set_color(np.random.rand(3))
    scat.set_offsets(np.random.uniform(0,9,(100,2)))
    scat.set_facecolors(np.random.rand(100,4))
    scat.set_sizes(np.random.uniform(0,9,100))
    return im, ln, scat

ani = FuncAnimation(fig, update, frames=10, interval=50, init_func=init, blit=True)
        
plt.show()
# ani.save('tmp/tmp.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# HTML(ani.to_html5_video())

