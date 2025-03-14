import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Load data from simulations
data = np.load("./out/data.npy")
# Extract energy and magnetization
energy = data[0]
mag = data[1]

# Find iterations where renormalization occurs
renorm_index = np.where(np.isnan(mag))[0]
# Lattice sizes (initial size 2**2)
sizes = [2**2]
for k in range(1, len(renorm_index) + 1):
    sizes.append(sizes[k-1]*2)
sizes = sizes[1:]


'''MAGNETIZATION PLOT'''


# Plot evolution of magnetization
plt.plot( range(len(mag)) , mag, label=r"Magnetization $m(t)$", color = "tab:blue" )

# Plot lines where renormalization occurs
plt.vlines(x = renorm_index, ymin = -1, ymax = +1, alpha = 0.5, color = "black", linestyle = "--", label="IRG")

# Generate legend
plt.legend()
# Title and axes
plt.title(r"Magnetization Evolution $m(t)$")
plt.xlabel(r"Time $t$ (Monte-Carlo Steps)")
plt.ylabel(r"Magnetization $m(t)$")
# Save plot
plt.savefig("./out/evol_mag.pdf", bbox_inches="tight")
# Show plot
# plt.show()


'''ENERGY PLOTS'''


# Close any old graphs
plt.close()

# Plot evolution of magnetization
plt.plot( range(len(energy)) , energy, label=r"Energy $E(t)$", color = "tab:orange" )

# Plot lines where renormalization occurs
plt.vlines(x = renorm_index, ymin = np.nanmin(energy), ymax = np.nanmax(energy), alpha = 0.5, color = "black", linestyle = "--", label="IRG")

# Generate legend
plt.legend()
# Title and axes
plt.title(r"Energy Evolution $E(t)$")
plt.xlabel(r"Time $t$ (Monte-Carlo Steps)")
plt.ylabel(r"Energy $E(t)$")
# Save plot
plt.savefig("./out/evol_nrg.pdf", bbox_inches="tight")
# Show plot
# plt.show()
plt.close()


'''CRITICAL EXPONENT BETA'''


# Compute mean of abs value of magnetization
m_mean = []

# Extract time series for each constant 
for k in range(len(renorm_index) - 1):
    # Compute the mean and append
    m_mean.append(  np.nanmean(  np.abs( mag[ renorm_index[k] : renorm_index[k+1] ] ) )  )
# Append remaining steps
m_mean.append( np.nanmean( np.abs( mag[ renorm_index[-1] :  ] ) ) )

plt.title(r"Magnetization Scaling at $T \approx T_c$")
plt.xlabel(r"$\log(L)$")
plt.ylabel(r"$\log(m)$")
plt.plot(sizes, np.array(m_mean), alpha = 0.5, color="tab:red")
plt.plot(sizes, np.array(m_mean), linestyle = "", marker = ".", color="tab:red")

plt.yscale("log")
plt.xscale("log")

# Linear spacing in size range
x = np.linspace(np.min(sizes), np.max(sizes), 100)

# Linear regression of logarithmic values
coefficients = np.polyfit(np.log(sizes), np.log(np.array(m_mean)), 1)
polynomial = np.poly1d(coefficients)
y_fit = polynomial(np.log(x))

# Print critical exponent
print("\nBeta = ", coefficients[1], "\n")

# Function to plot
y = np.exp( y_fit )
# Plot regression result
plt.plot(x, y, linestyle="-", color="black", alpha=0.5, label="Fit: $\beta = {}$".format(round( coefficients[1], 3) ))
plt.savefig("./out/scale_mag.pdf", bbox_inches="tight")
# plt.show()
plt.close()


'''SUSCEPTIBILITY'''


# Compute mean of abs value of magnetization
m_mean_sq = []
# Extract time series for each constant 
for k in range(len(renorm_index) - 1):
    # Compute the mean and append
    m_mean_sq.append( 
        np.nanmean( np.abs( np.array( mag[ renorm_index[k+1]+1 ] ) ** 2 ) )
    )
# Append remaining steps
m_mean_sq.append( 
    np.nanmean( np.abs( np.array( mag[ renorm_index[-1] :  ] ) ** 2 ) ) 
)

# Linear regression of logarithmic values
coefficients = np.polyfit(np.log(sizes), np.log(np.array(m_mean_sq)), 1)
polynomial = np.poly1d(coefficients)
y_fit = polynomial(np.log(x))
# Print critical exponent
print("\nBeta = ", coefficients[1], "\n")

# Function to plot
y = np.exp( y_fit )
# Plot regression result
plt.plot(x, y, linestyle="-", color="black", alpha=0.5, label="Fit: $\beta = {}$".format(round( coefficients[1], 3) ))

plt.plot(sizes, m_mean_sq)
plt.yscale("log")
plt.xscale("log")
# plt.show()
plt.close()


'''TIME AUTOCORRELATION'''

# Compute mean of abs value of magnetization
mags = []

# Append last steps
mags.append( 
    np.array( mag[ 1 + renorm_index[-1] :  ] )
)

M_fluct = np.array(mags[0]) - m_mean[-1]

'''DIRECT METHOD'''

# Autocorrelation function
C_tau = np.zeros(len(M_fluct)//2)

# Compute correlation function
for tau in range(0, len(M_fluct)//2):
    # Sum over times
    for t in range(0, len(M_fluct) - tau - 1):
        # Compute sum
        C_tau[tau] += M_fluct[t] * M_fluct[t + tau]

# Multiply by field H and divide by k_B*T
C_dir = C_tau / C_tau[0] # * 2.269

tau = integrate.trapezoid(y  = C_dir)
print(tau)

plt.plot(range(len(M_fluct)//2), C_dir)
plt.show()