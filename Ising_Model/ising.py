import  random
import  numpy                   as  np
import  matplotlib.pyplot       as  plt
import  matplotlib.colors       as  col
import  matplotlib.animation    as  animation

# Ignore overfow errors (exponential goes to zero fast)
np.seterr(over='ignore')

# Filenames
fname_out = "ising_out.txt"
fname_ani = "ising_ani.gif"

# Dimension of the square lattice
dim = 16
# Number of Monte-Carlo iterations to perform
iter = 75
# Temperature of system
T = 0.01

# Open output file
file = open(fname_out, mode="w")
    
# Create spin matrix for system
s = np.empty(shape=(dim,dim), dtype=int)

# Random initial configuration
for i in range(0, dim):
    for j in range(0, dim):
        # Set each spin to random value
        s[i][j] = random.choice([-1,+1])

    # Periodic conditions
    # s[dim-1][dim-1] = s[0][0]
    # s[i][dim-1] = s[i][0]
    # s[dim-1][j] = s[0][j]

# Perform Monte-Carlo iterations
for k in range(iter):
    # Loop over each Monte-Carlo iteration
    for l in range(dim*dim):

        # Chooose random point
        i = np.random.randint(low=0, high=dim)
        j = np.random.randint(low=0, high=dim)

        # Calculate Energy (Hamiltonian)
        E = 2*s[i][j] * ( s[ ( dim + (i+1) % dim ) % dim ][j] 
            + s[ ( dim + (i-1) % dim) % dim ][j] 
            + s[i][ ( dim + (j+1) % dim) % dim ] 
            + s[i][ ( dim + (j-1) % dim ) % dim ] )

        # Probability of changing the spin value
        p = min(1.0, np.exp(-E/T))

        # Randomly change spin based on probability
        if (random.random() < p):
            # Change the value
            s[i][j] *= -1
            
            # Periodic conditions
            # for i in range(dim):
            #     for j in range(dim):
            #         s[dim-1][dim-1] = s[0][0]
            #         s[i][dim-1] = s[i][0]
            #         s[dim-1][j] = s[0][j]

    # Write system state to file
    for i in range(dim):
        for j in range(dim):
            file.write(str(s[i][j]) + " ")
        file.write("\n")

# Close output file
file.close()

# Load output file to create plot
data = np.loadtxt(fname_out)

# Define colors and colormar
color_dict = {0: '#2D4263', 1: '#C84B31'}
label_dict = {0: 'off', 1: 'on'}
imax = max(label_dict)
imin = min(label_dict)
cmap = col.ListedColormap(color_dict.values())

# Create pixel grid
grid = np.zeros((dim,dim))

# Update function for animation (show current state of system)
def update(n):
    global grid
    newGrid = grid.copy()
    for i in range(dim):
        for j in range(dim):
            newGrid[i][j] = data[i+n*dim][j]
    mat.set_data(grid)
    grid = newGrid
    return [mat]

# How many iterations to skip for visualization
skip = 1

# Create subplots
fig, ax = plt.subplots()

# Show grid and create animation
mat = ax.matshow(grid, cmap=cmap, interpolation='nearest', vmin=imin, vmax=imax)
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, iter, skip), interval=1, repeat=True)
# Reduce margins
fig.tight_layout(rect=[0, 0, 1, 0.95])
# Set title and colorbar
plt.title("Ising Model (T = {})".format(T))
cbar = fig.colorbar(mat, ticks=[0,1])
cbar.ax.set_yticklabels(['-1', '+1'])
# plt.show()
ani.save(fname_ani, writer=animation.PillowWriter(fps=30))