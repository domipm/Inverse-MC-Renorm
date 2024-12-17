import numpy as np
import matplotlib.pyplot as plt 
import time

import matplotlib.animation as animation

n= 16	



def inverse_renornalization(lattice):
    
    n, n = lattice.shape 
    renorm_lattice = np.ones((n*2, n*2))

    for i in range(n):
        for j in range(n):
            renorm_lattice[2*i, 2*j] = lattice[i, j]
            renorm_lattice[2*i+1, 2*j] = lattice[i, j]
            renorm_lattice[2*i, 2*j+1] = lattice[i, j]
            renorm_lattice[2*i+1, 2*j+1] = lattice[i, j]

    return renorm_lattice

original_lattice =  2*np.random.randint(2, size=(n//2, n//2))-1


def ising_model(n, T, nsteps):

    # Initialize the lattice
    lattice = inverse_renornalization(original_lattice)

    # Initialize variables to store the magnetization and energy
    magnetization = np.zeros(nsteps)
    energy = np.zeros(nsteps)
    trajectory = np.zeros((nsteps, n, n))

    # Calculate the initial magnetization and energy
    magnetization[0] = np.sum(lattice)
    energy[0] = -np.sum(lattice*np.roll(lattice, 1, axis=0) + lattice*np.roll(lattice, 1, axis=1))
    trajectory[0] = lattice

    # Define the temperature
    beta = 1/T

    # Perform the Monte Carlo steps
    for i in range(1, nsteps):
        # Select a random spin
        x = np.random.randint(n)
        y = np.random.randint(n)

        if np.sign(np.sum(lattice[x//2:x//2+2,y//2:y//2+2])) == np.sign(original_lattice[x//2,y//2]):
            #do the simulation
            pass


        elif np.sign(np.sum(lattice[x//2:x//2+2,y//2:y//2+2])) == 0:
            #do 50/50 and do the simulation
            if np.random.choice([True,False]):
                pass
            else:
                magnetization[i] = magnetization[i-1]
                energy[i] = energy[i-1]
                trajectory[i] = trajectory[i-1]
                continue
            

        elif np.sign(np.sum(lattice[x//2:x//2+2,y//2:y//2+2])) != np.sign(original_lattice[x//2,y//2]):
                # dont flip

            magnetization[i] = magnetization[i-1]
            energy[i] = energy[i-1]
            trajectory[i] = trajectory[i-1]
            continue
            

        # Calculate the change in energy
        dE = 2*lattice[x, y]*(lattice[(x+1)%n, y] + lattice[(x-1)%n, y] + lattice[x, (y+1)%n] + lattice[x, (y-1)%n])

        # Decide whether to flip the spin
        if dE < 0 or np.random.rand() < np.exp(-beta*dE):
            lattice[x, y] *= -1
            magnetization[i] = magnetization[i-1] + 2*lattice[x, y]
            energy[i] = energy[i-1] + dE
            trajectory[i] = lattice 
        else:
            magnetization[i] = magnetization[i-1]
            energy[i] = energy[i-1]
            trajectory[i] = trajectory[i-1]

    return lattice, magnetization, energy, trajectory


def main(animate = False):
    # Define the parameters of the simulation
    n = 16
    T = 2.5
    nsteps = 10000

    # Perform the simulation
    start = time.time()
    lattice, magnetization, energy, trajectory = ising_model(n, T, nsteps)
    end = time.time()
    print('Time taken:', end-start)

    # Plot the results
    plt.figure()
    plt.imshow(lattice, cmap='binary')
    plt.title('Lattice configuration')
    plt.axis('off')

    plt.figure()
    plt.plot(magnetization)
    plt.title('Magnetization')
    plt.xlabel('Monte Carlo steps')
    plt.ylabel('Magnetization')

    plt.figure()
    plt.plot(energy)
    plt.title('Energy')
    plt.xlabel('Monte Carlo steps')
    plt.ylabel('Energy')

    plt.show()


    if animate == True:

        # animate the trajectory
        fig = plt.figure()
        ims = []
        for i in range(nsteps):
            im = plt.imshow(trajectory[i], animated=True, cmap='binary')
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=1, blit=True, repeat_delay=1000)
        plt.show()




if __name__ == '__main__':
    main(animate = True)




if __name__ == '__main__':
    main(animate = False)
