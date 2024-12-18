import  numpy               as np

import  matplotlib.pyplot   as plt
import  matplotlib.colors   as col

import  glob
import  os

import  re
import  contextlib
from    PIL                 import Image

class Lattice():

    def __init__(self, n_size):
        'Initialize the lattice'

        # Size of lattice
        self.n_size = n_size

        # Initialize the lattice
        self.lattice = 2 * np.random.randint(2, size = (n_size,) * 2) - 1

        # Keep track of the original lattice
        self.lattice_org = self.lattice
        self.n_size_org = n_size

        # Initial size
        self.n_size_init = 4

        return
    
    def show_lattice(self):
        'Display current state of lattice'

        # Setup plots
        fig, ax = plt.subplots()
        fig.tight_layout()

        # Define color legend
        color_dict = {0: '#2D4263', 1: '#C84B31'}
        label_dict = {0: 'off', 1: 'on'}
        imax = max(label_dict)
        imin = min(label_dict)
        cmap = col.ListedColormap(color_dict.values())

        # Display lattice
        mat = ax.matshow(self.lattice, cmap=cmap, interpolation="nearest", vmin=imin, vmax=imax)
        
        # Display colorbar
        cbar = fig.colorbar(mat, ticks = [0,1])
        cbar.ax.set_yticklabels(['-1', '+1'])
    
        # Show grid (small lines)
        ax.vlines(x=[k-0.5 for k in range(self.n_size)], color="black", ymin=-0.5, ymax=self.n_size - 0.5, alpha=0.15, linestyles="-", linewidth=0.5)
        ax.hlines(y=[k-0.5 for k in range(self.n_size)], color="black", xmin=-0.5, xmax=self.n_size - 0.5, alpha=0.15, linestyles="-", linewidth=0.5)
        # Show grid (big lines)
        lines = [k-0.5 for k in range(0, self.n_size, self.n_size // self.n_size_init)]
        ax.vlines(x=lines, color="black", ymin=-0.5, ymax=self.n_size - 0.5, alpha=0.5)
        ax.hlines(y=lines, color="black", xmin=-0.5, xmax=self.n_size - 0.5, alpha=0.5)
        # Disable axes
        ax.axis('off')
        # Write size of lattice
        ax.text(0,-0.055,s="$n = $" + str(self.n_size), transform = ax.transAxes)

        # Show final plot
        plt.savefig("./img/" + str(self.n_size) + "_0.png", dpi=300)
        plt.close()
        # plt.show()

        return
    
    def montecarlo(self, n_steps, temperature, skip):
        'Perform Monte-Carlo simulation steps on lattice'

        # Magnetization and energy
        self.magnetization = np.zeros(n_steps)
        self.energy = np.zeros((n_steps, self.n_size, self.n_size))

        # Initial magnetization and energy
        self.magnetization[0] = np.sum(self.lattice)
        self.energy[0] = -np.sum(self.lattice*np.roll(self.lattice, 1, axis=0) + self.lattice*np.roll(self.lattice, 1, axis=1))

        # Define temperature
        beta = 1/temperature

        # Setup plots
        fig, ax = plt.subplots()
        fig.tight_layout()

        # Define color legend
        color_dict = {0: '#2D4263', 1: '#C84B31'}
        label_dict = {0: 'off', 1: 'on'}
        imax = max(label_dict)
        imin = min(label_dict)
        cmap = col.ListedColormap(color_dict.values())
    
        # Display colorbar
        img = plt.imshow(2 * np.random.randint(2, size = (self.n_size,) * 2) - 1, cmap=cmap, interpolation="nearest", vmin=imin, vmax=imax)
        img.set_visible(False)
        cbar = fig.colorbar(img, ticks = [0,1])
        cbar.ax.set_yticklabels(['-1', '+1'])

        # Show grid (small lines)
        ax.vlines(x=[k-0.5 for k in range(self.n_size)], color="black", ymin=-0.5, ymax=self.n_size - 0.5, alpha=0.15, linestyles="-", linewidth=0.5)
        ax.hlines(y=[k-0.5 for k in range(self.n_size)], color="black", xmin=-0.5, xmax=self.n_size - 0.5, alpha=0.15, linestyles="-", linewidth=0.5)
        # Show grid (big lines)
        lines = [k-0.5 for k in range(0, self.n_size, self.n_size // self.n_size_init)]
        ax.vlines(x=lines, color="black", ymin=-0.5, ymax=self.n_size - 0.5, alpha=0.5)
        ax.hlines(y=lines, color="black", xmin=-0.5, xmax=self.n_size - 0.5, alpha=0.5)
        # Write size of lattice
        ax.text(0,-0.055,s="$n = $" + str(self.n_size), transform = ax.transAxes)
        # Disable axes
        ax.axis("off")

        # Perform Monte-Carlo steps
        for i in range(1, n_steps):

            # Select a random spin
            x = np.random.randint(self.n_size)
            y = np.random.randint(self.n_size)

            # Flip the spin
            self.lattice[x,y] *= -1

            # Check if transformation is compatible
            if np.sign(np.sum(self.lattice[x//2*2:x//2*2+2, y//2*2:y//2*2+2])) == np.sign(self.lattice_org[x//2, y//2]):
                # Compatible
                self.lattice[x,y] *= -1

            elif np.sum(self.lattice[x//2*2:x//2*2+2, y//2*2:y//2*2+2]) == 0:
                # Randomly choose if compatible or not
                self.lattice[x,y] *= -1
                if np.random.choice([True, False]):
                    pass
                else:
                    # Update magnetization and energy
                    self.magnetization[i] = self.magnetization[i-1]
                    self.energy[i] = self.energy[i-1]
                    continue

            elif np.sign(np.sum(self.lattice[x//2*2:x//2*2+2, y//2*2:y//2*2+2])) != np.sign(self.lattice_org[x//2,y//2]):
                self.lattice[x,y] *= -1
                # Update magnetization and energy
                self.magnetization[i] = self.magnetization[i-1]
                self.energy[i] = self.energy[i-1]
                continue

            # Calculate the change in energy
            energy = 2 * self.lattice[x, y] * (self.lattice[(x+1) % self.n_size, y] 
                       + self.lattice[(x-1) % self.n_size, y] 
                       + self.lattice[x, (y+1) % self.n_size] 
                       + self.lattice[x, (y-1) % self.n_size])

            # Flip spin (or not)
            if energy < 0 or np.random.rand() < np.exp(-beta*energy):
                self.lattice[x,y] *= -1
                # Update magnetization and energy
                self.magnetization[i] = self.magnetization[i-1] + 2*self.lattice[x,y]
                self.energy[i] = self.energy[i-1] + energy
            else:
                # Update magnetization and energy
                self.magnetization[i] = self.magnetization[i-1]
                self.energy[i] = self.energy[i-1]

            # Skip steps
            if i % skip == 0:
                # Plot matrix
                mat = ax.matshow(self.lattice, cmap=cmap, interpolation="nearest", vmin=imin, vmax=imax)
                # Save image
                plt.savefig("./img/" + str(self.n_size) + "_" + str(i) + ".png", dpi=300)

        return
    
    def renorm(self):
        'Renormalize lattice to higher dimension'

        # New size
        n_size_new = self.n_size * 2
        # Initialize new lattice
        lattice_new = 2 * np.random.randint(2, size = (n_size_new,) * 2) - 1

        # Ensure new lattice has same values as old
        for i in range(self.n_size):
            for j in range(self.n_size):
                # This can be simplified I think
                lattice_new[2*i, 2*j] = self.lattice[i,j]
                lattice_new[2*i+1, 2*j] = self.lattice[i,j]
                lattice_new[2*i, 2*j+1] = self.lattice[i,j]
                lattice_new[2*i+1, 2*j+1] = self.lattice[i,j]

        # Save current lattice as previous
        self.lattice_org = self.lattice
        self.n_size_org = self.n_size

        # Set lattice to the new, renormalized lattice
        self.lattice = lattice_new
        self.n_size = n_size_new

        return
    
# Remove old files from folder
files = glob.glob('./img/*')
for f in files:
    os.remove(f)

# Simulation parameters
mc_steps = 10000
temperature = 2.269
skip = 1000
size_init = 2 # Initial size will be 2**size_init 
size_fin = 7 # Final size will be 2**size_fin
    
lattice = Lattice(n_size = 2**size_init)
lattice.show_lattice()

for _ in range(1, size_fin - 1):

    lattice.renorm()
    lattice.show_lattice()
    lattice.montecarlo(mc_steps, temperature, skip)

# Generate gif from images

# Filepaths
fp_in = "./img/"
fp_out = "./IMCRG.gif"

fnames = np.sort(os.listdir(fp_in))

# Sorting function that extracts x and y numbers
def extract_numbers(filename):
    match = re.match(r"(\d+)_(\d+)\.png", filename)
    if match:
        x, y = int(match[1]), int(match[2])
        return (x, y)
    return (float('inf'), float('inf'))  # In case of a non-matching format

# Sort using numpy's `sorted` function
fnames = np.array(sorted(fnames, key=extract_numbers))
print(fnames)
files = []
for i, fname in enumerate(fnames):
    files.append( str("./img/" + fname) )

# Use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # Lazily load images
    imgs = (stack.enter_context(Image.open(f))
            for f in files)

    # Extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)