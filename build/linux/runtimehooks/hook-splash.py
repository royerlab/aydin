import matplotlib
import matplotlib.pyplot as plt

from aydin.gui.resources.json_resource_loader import absPath

matplotlib.rcParams['toolbar'] = 'None'


from PIL import Image


img = Image.open(absPath("biohub_logo.png"))
fig = plt.figure()
man = plt.get_current_fig_manager()
man.canvas.set_window_title("Aydin Studio - image denoising, but chill...")
imgplot = plt.imshow(img)

plt.axis('off')


plt.show(block=False)
plt.pause(2)
plt.close()
