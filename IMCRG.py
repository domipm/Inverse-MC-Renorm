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

        # Keep track of the "previous" lattice (to check compatibility)
        self.lattice_pre = self.lattice
        self.n_size_pre = n_size

        # Initial, original lattice size (visualization purposes)
        self.n_size_init = self.n_size

        # Energy and magnetization history
        self.energy_hist = []
        self.mag_hist = []

        # Current energy (compute for initial step)
        self.energy, self.mag = self.comp_em()

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
        ax.text(0, -0.055, s="$n = $" + str(self.n_size), transform = ax.transAxes)

        # Show final plot
        plt.savefig("./img/" + str(self.n_size) + "_0.png", dpi=300)
        plt.close()

        return
    
    def montecarlo(self, n_steps, temperature, skip = 1000):
        'Perform Monte-Carlo simulation steps on lattice'

        # Define temperature
        beta = 1.0 / temperature

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

            for _ in range(self.n_size ** 2):

                # Select a random spin
                x = np.random.randint(self.n_size)
                y = np.random.randint(self.n_size)

                # Propose spin flip
                self.lattice[x,y] *= -1            
                # Is this compatible?
                compatible = True

                # Check for compatibility with previous lattice

                # Compute sum of spins from new lattice
                block_sum = np.sum( self.lattice[x//2*2:x//2*2+2, y//2*2:y//2*2+2] )

                # Case: sum of spins same sign as previous block
                if np.sign( block_sum ) == np.sign(self.lattice_pre[x//2, y//2]):
                    # It is compatible
                    pass
                # Case: sum of spins equals zero
                elif np.sum( block_sum == 0 ):
                    # Randomly accept spin flip
                    compatible = np.random.choice([True, False])
                # Case: sum of spins different sign as previous block
                else:
                    # Not compatible
                    compatible = False

                # Revert proposed spin flip back
                self.lattice[x,y] *= -1

                # Compute the change in energy for single spin flip
                delta_energy = 2*self.lattice[x, y]*(
                    self.lattice[(x+1)%self.n_size, y] 
                    + self.lattice[(x-1)%self.n_size, y] 
                    + self.lattice[x, (y+1)%self.n_size] 
                    + self.lattice[x, (y-1)%self.n_size]
                    )

                # Consider proposed spin flip, if compatible, flip according to probability
                if compatible == True:
                    # Flip spin (accept proposal) if minimizes energy or according to acceptance probability
                    #if delta_energy < 0 or 
                    if np.random.rand() < np.min( (1.0, np.exp( - beta * delta_energy )) ):
                        self.lattice[x,y] *= -1
                    # Otherwise, don't flip the spin
                    
            # Skip steps in visualization
            if i % skip == 0:
                # Plot matrix
                _ = ax.matshow(self.lattice, cmap=cmap, interpolation="nearest", vmin=imin, vmax=imax)
                # Save image
                plt.savefig("./img/" + str(self.n_size) + "_" + str(i) + ".png", dpi=300)

            # At each Monte-Carlo step, compute energy and magnetization
            self.energy, self.mag = self.comp_em()
            
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
                lattice_new[2*i, 2*j] = self.lattice[i,j]
                lattice_new[2*i+1, 2*j] = self.lattice[i,j]
                lattice_new[2*i, 2*j+1] = self.lattice[i,j]
                lattice_new[2*i+1, 2*j+1] = self.lattice[i,j]

        # Save current lattice as previous
        self.lattice_pre = self.lattice
        self.n_size_pre = self.n_size
        # Set current lattice to the new renormalized lattice
        self.lattice = lattice_new
        self.n_size = n_size_new

        return
    
    def comp_em(self, J = 1):
        "Compute energy and magnetization of current state and append to history"

        # Spin-interaction energy
        energy_spin = 0
        for i in range(self.n_size):
            for j in range(self.n_size):
                # Sum over all nearest neighbors (periodic boundaries)
                energy_spin += self.lattice[i,j] * (
                    self.lattice[(i+1)%self.n_size, j] +
                    + self.lattice[(i-1)%self.n_size, j]
                    + self.lattice[i, (j+1)%self.n_size] 
                    + self.lattice[i, (j-1)%self.n_size]
                )

        # Avoid double-counting and include interaction strength
        energy = - energy_spin * J / 2

        # Calculate magnetization per spin
        mag = ( 1.0 / (self.n_size**2) ) * np.sum( self.lattice )

        # Append to corresponding arrays
        self.energy_hist.append(energy)
        self.mag_hist.append(mag)

        return energy, mag

# Function to generate *.gif from sequence of images *.png
def make_gif(filepath_in = "./img/", filepath_out = "./IMCRG.gif", duration = 200, loop = 0):

    # Information
    print("Generating animation")

    # Filepaths
    fp_in = filepath_in
    fp_out = filepath_out

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

    return



'''SIMULATION'''



# Where to save images
img_path = "./img/"
# Where to save data
data_path = "./out/"
# Where to save animation
anim_path = "./out/"
# Remove old images from folder (if there are any)
try:
    files = glob.glob(img_path + "*")
    for f in files:
        os.remove(f)
except:
    pass

# Simulation parameters
temperature = 2.269     # Approx. critical temperature
size_init = 2           # Initial size will be 2**size_init 
size_fin = 9            # Final size will be 2**size_fin
    
# Generate initial lattice
lattice = Lattice(n_size = 2**size_init)
# Plot initial lattice
lattice.show_lattice()

# At each lattice size, perform 10^L Monte-Carlo iterations
for n_iter in range(1, size_fin - 1):

    # NaN signals inverse renormalization transformation applied
    lattice.energy_hist.append(np.nan)
    lattice.mag_hist.append(np.nan)
    # Apply inverse renormalization transformation
    lattice.renorm()

    # Iteration information
    print("Iteration ", n_iter, " Size ", lattice.n_size)

    # Save corresponding image (first after renorm)
    lattice.show_lattice()
    # Perform Monte-Carlo steps
    lattice.montecarlo(n_steps = 1000, temperature = temperature, skip = 10**10)



'''SAVE RESULTS'''



# Generate the animation from images (default parameters)
make_gif(filepath_out = anim_path + "lattice.gif")

# Save data to file
np.save(file = data_path + "data", arr = [lattice.energy_hist, lattice.mag_hist])