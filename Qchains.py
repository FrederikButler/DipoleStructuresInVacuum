import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from matplotlib.colors import hsv_to_rgb
import cmath

def ArrayGenerator(num_points, showStructure = False):
    chain_3d = np.zeros((num_points, 3))  # shape (50, 3), all zeros to start
    chain_3d[:, 0] = np.arange(num_points)  # Set x-coordinates

    if showStructure == True: 
        x, y, z = chain_3d[:, 0], chain_3d[:, 1], chain_3d[:, 2]

        #interactive scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8)
        )])

        # Customize layout
        fig.update_layout(
            title="Geometry of the atoms",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()
    return chain_3d

def RingGenerater(n_points, showStructure = False):
    if n_points < 3:
        raise ValueError("Polygon must have at least 3 points.")

    # Generate angles evenly spaced around a circle
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    
    # Initial unscaled polygon points (unit circle)
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.stack((x, y), axis=-1)  # shape (n_points, 2)

    # Compute distance between two neighboring points
    side_length = np.linalg.norm(points[0] - points[1])

    # Scale so that side length is 1
    scale = 1.0 / side_length
    points *= scale

    # Embed in 3D by adding z=0
    points_3d = np.hstack((points, np.zeros((n_points, 1))))

    if showStructure == True: 
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

        #interactive scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8)
        )])

        # Customize layout
        fig.update_layout(
            title="Geometry of the atoms",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()

    return points_3d

def StarGenerator(n_half_points, showStructure=False):
    if n_half_points < 3:
        raise ValueError("Polygon must have at least 3 points.")

    # Step 1: Regular polygon on unit circle
    angles = np.linspace(0, 2 * np.pi, n_half_points, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.stack((x, y), axis=-1)
    
    # Scale so edge length = 1
    side_length = np.linalg.norm(points[0] - points[1])
    scale = 1.0 / side_length
    points *= scale
    points_3d = np.hstack((points, np.zeros((n_half_points, 1))))

    # Step 2: Add triangle peaks
    triangle_peaks = []
    for i in range(n_half_points):
        p1 = points_3d[i]
        p2 = points_3d[(i + 1) % n_half_points]  # wrap around

        # Midpoint of edge
        mid = 0.5 * (p1 + p2)

        # Direction of edge
        edge_vec = p2 - p1
        edge_vec /= np.linalg.norm(edge_vec)

        # Perpendicular direction (normal to the polygon plane)
        normal_vec = np.array([0, 0, 1])  # z-direction
        outward_dir = np.cross(edge_vec, normal_vec)
        outward_dir /= np.linalg.norm(outward_dir)

        # Height of isosceles triangle from base to tip:
        # For base length b and legs L:
        # height h = sqrt(L^2 - (b/2)^2)
        b = np.linalg.norm(p2 - p1)
        L = 1.0  # desired leg length
        h = np.sqrt(L**2 - (b / 2)**2)

        # Triangle peak position
        peak = mid + h * outward_dir
        triangle_peaks.append(peak)

    triangle_peaks = np.array(triangle_peaks)

    all_points = np.vstack((points_3d, triangle_peaks))


    if showStructure:
        fig = go.Figure()

        # Original base points (orange)
      # Close the ring loop
        ring_closed = np.vstack([points_3d, points_3d[0:1]])

        fig.add_trace(go.Scatter3d(
            x=ring_closed[:, 0], y=ring_closed[:, 1], z=ring_closed[:, 2],
            mode='markers+lines',
            marker=dict(size=5, color='orange'),
            line=dict(color='orange'),
            name='Ring vertices'
        ))

        # Close the triangle peak loop
        peaks_closed = np.vstack([triangle_peaks, triangle_peaks[0:1]])

        fig.add_trace(go.Scatter3d(
            x=peaks_closed[:, 0], y=peaks_closed[:, 1], z=peaks_closed[:, 2],
            mode='markers+lines',
            marker=dict(size=5, color='green'),
            line=dict(color='green'),
            name='Triangle peaks'
        ))

        fig.update_layout(
            title="Star Polygon Structure",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show()

    return all_points

def StructurePlotter(points_3d, title = "Geometry of the atoms" ):
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=5, opacity=0.8)
    )])

    # Customize layout
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

def FindEigenstates(points, dipole_vec_hat, distance):
    '''
    Solve the problem of db/dt = -1/2Gamma b - 1/2 Gamma F b. Solves it unitless. That is 
    finds eigvalues/vectors of A = -1/2 (I + F)

    Args:
    points(np.array): array of 3d points
    dipole_vec_hat(np.array): single unit vector specifying the direction of the dipole moment
    distance: distance between every atoms. Note d = r_ij * 2 pi / lambda_e 

    Out: 
    Eigvalues of A 
    Eigvectors of A. They are in the columns!
    '''
    #distance_matrix = distance_matrix(points, points) eventuelt for hurtigere 
    r_ij_vector = points[:, np.newaxis] - points
    r_ij = np.linalg.norm(r_ij_vector, axis=2, keepdims=True)
    r_ij_hat = np.divide(r_ij_vector, r_ij, out=np.zeros_like(r_ij_vector, dtype=float), where=(r_ij != 0)) 

    #Calculates k_e * r_ij. So k_e__r_ij[0] contains k_e times all distances between r_i'th atom and all others
    k_e__r_ij_vector = distance * 2 * np.pi * r_ij_vector 
    k_e__r_ij = np.linalg.norm(k_e__r_ij_vector, axis=2, keepdims=True)

    F = np.zeros(shape = (len(k_e__r_ij), len(k_e__r_ij)), dtype=complex)

    for l in range(len(k_e__r_ij)): 
        dot_product = np.dot(r_ij_hat[l, :, :] , dipole_vec_hat)

        denom1 = (k_e__r_ij[l].flatten())
        denom2 = (k_e__r_ij[l].flatten())**2
        denom3 = (k_e__r_ij[l].flatten())**3
        safe_denom1 = np.where(denom1 == 0, np.nan, denom1)
        safe_denom2 = np.where(denom2 == 0, np.nan, denom2)
        safe_denom3 = np.where(denom3 == 0, np.nan, denom3)

        f_ji =  3/2 * (1 - (dot_product)**2) * np.sin(k_e__r_ij[l].flatten())/safe_denom1 \
            + 3/2 * (1 - 3 * (dot_product)**2) * (np.cos(k_e__r_ij[l].flatten())/safe_denom2 - np.sin(k_e__r_ij[l].flatten())/ safe_denom3) 
        g_ji = -3/2 * (1 - (dot_product)**2) * np.cos(k_e__r_ij[l].flatten())/safe_denom1 \
            + 3/2 * (1 - 3 * (dot_product)**2) * (np.sin(k_e__r_ij[l].flatten())/safe_denom2 + np.cos(k_e__r_ij[l].flatten())/ safe_denom3) 
        
        f_ji = np.nan_to_num(f_ji)
        g_ji = np.nan_to_num(g_ji)

        #Fill F
        F[:, l] = f_ji + 1j * g_ji 
    
    I = np.eye(len(k_e__r_ij), dtype = complex)
    A = - 1/2 * (I + F)

    A_eigenvalues, A_eigenvectors = np.linalg.eig(A)
    return A_eigenvalues, A_eigenvectors

def EigenstatesHedgehog(ring_points, unit_polarizations, distance):


    ### Find eigenvalues and eigenstates of this system 
    #distance_matrix = distance_matrix(points, points) eventuelt for hurtigere 
    r_ij_vector = ring_points[:, np.newaxis] - ring_points
    r_ij = np.linalg.norm(r_ij_vector, axis=2, keepdims=True)
    r_ij_hat = np.divide(r_ij_vector, r_ij, out=np.zeros_like(r_ij_vector, dtype=float), where=(r_ij != 0)) 

    #Calculates k_e * r_ij. So k_e__r_ij[0] contains k_e times all distances between r_i'th atom and all others
    k_e__r_ij_vector = distance * 2 * np.pi * r_ij_vector 
    k_e__r_ij = np.linalg.norm(k_e__r_ij_vector, axis=2, keepdims=True)

    F = np.zeros(shape = (len(k_e__r_ij), len(k_e__r_ij)), dtype=complex)

    for l in range(len(k_e__r_ij)): 
        dipole_vec_hat = unit_polarizations[l]

        dot_product = np.dot(r_ij_hat[l, :, :] , dipole_vec_hat)

        denom1 = (k_e__r_ij[l].flatten())
        denom2 = (k_e__r_ij[l].flatten())**2
        denom3 = (k_e__r_ij[l].flatten())**3
        safe_denom1 = np.where(denom1 == 0, np.nan, denom1)
        safe_denom2 = np.where(denom2 == 0, np.nan, denom2)
        safe_denom3 = np.where(denom3 == 0, np.nan, denom3)

        f_ji =  3/2 * (1 - (dot_product)**2) * np.sin(k_e__r_ij[l].flatten())/safe_denom1 \
            + 3/2 * (1 - 3 * (dot_product)**2) * (np.cos(k_e__r_ij[l].flatten())/safe_denom2 - np.sin(k_e__r_ij[l].flatten())/ safe_denom3) 
        g_ji = -3/2 * (1 - (dot_product)**2) * np.cos(k_e__r_ij[l].flatten())/safe_denom1 \
            + 3/2 * (1 - 3 * (dot_product)**2) * (np.sin(k_e__r_ij[l].flatten())/safe_denom2 + np.cos(k_e__r_ij[l].flatten())/ safe_denom3) 
        
        f_ji = np.nan_to_num(f_ji)
        g_ji = np.nan_to_num(g_ji)

        #Fill F
        F[:, l] = f_ji + 1j * g_ji 

    I = np.eye(len(k_e__r_ij), dtype = complex)
    A = - 1/2 * (I + F)

    A_eigenvalues, A_eigenvectors = np.linalg.eig(A)
    return A_eigenvalues, A_eigenvectors

def PlotEigenstateEvolution(eigenvalues, eigenvectors, time, title = 'Time evolution starting in eigenstate', legend=True):
    '''
    Creates a plot of the time evolution of the eigenstates. So it shows the probability of being in
    the eigenstate at time t, given you started in that eigenstate

    Args:
    time(np.linspace): The time interval interested in. In units of Gamma = spontanious decay rate of atom in the excited state
    '''
    
    bt = np.zeros(shape = (len(eigenvalues), len(time)), dtype=complex)

    for i in range(len(eigenvalues)):
        bt[i] = np.exp((eigenvalues[i] + np.conjugate(eigenvalues[i])) * time) 
        plt.plot(time, bt[i].real, label = f"$\\chi$$_{i+1}$")

    plt.plot(time, np.exp(-time), c='red', linestyle = '--', label = r"$e^{-t \ \Gamma_0}$") #for better distinction of sub and super radiant states
    if legend:
        plt.legend()
    plt.title(title)
    plt.xlabel(r"$t \ \Gamma_0$")
    plt.ylabel(r"$|\chi(t)|^2$")
    plt.show()

def RandInitialStateEvolution(eigenvalues, eigenvectors, init_state, time, return_init_lambdabase=False): 
    '''
    Given the eigenstates, eigenvalues of a chain and a normalized initial state, and a time (as linspace in gammas) 
    this function converts the initial state to the basis of the eigenstates. In this basis we now the time evolution. 
    So we get the time evolution for the probability to be on each site in the lambda base. We convert back to the i-base. 

    So the function returns a matrix with (#sites, P(t)). So for each site the probability that a site is excited over the interval
    '''
    A_eigenvectors_inv = np.linalg.inv(eigenvectors)        #so that A_eigvectors_inv * eigenvectors = I 

    tester = np.round(A_eigenvectors_inv @ eigenvectors, 14)
    is_identity = np.array_equal(tester, np.eye(eigenvectors.shape[0]))
    if is_identity != True:
        print('OBS: Something wrong in the inverse')
    
    #Get time evolution of the sites
    init_state_lambdabase = A_eigenvectors_inv @ init_state
    coeff_evolution_lambdabase = init_state_lambdabase[:, np.newaxis] * np.exp(eigenvalues[:, np.newaxis] * time)
    coeff_evolution_ibase = eigenvectors @ coeff_evolution_lambdabase 
    prob_exc_sites_t = (coeff_evolution_ibase * coeff_evolution_ibase.conj()).real          #is already real but due numerical inaccuracies we get like e-17. So just to get rid of these  

    if return_init_lambdabase: 
        return init_state_lambdabase, prob_exc_sites_t
    else:
        return prob_exc_sites_t

def InitialStateEvolutionHeatMap(prob_exc_sites_t, time):

    '''
    Plots the different sites on the x-axis and the time on the y-axis. Each state pixel gets coloured by the value of  the 
    probability that the given site is excited at that time. Note the y axis needs some extra thought. time = np.linspace(0, t_final, res). 
    So the second argument gives how many Gammas/decay rates you are shown whereas the third argument res simply gives you the resolution. 
    Thus if t_final = res then each pixel you move up corresponds to one time of Gamma passing. 
    '''

def SiteProbDistribution(eig_values, eig_vectors, state_num, figsize, hue=False): 
    #sort eigenvalues and eigenvectors. Lowest index being most sub-radiant and highest subradiant 
    sorted_indices = np.lexsort((eig_values.real, np.abs(eig_values.real)))
    sorted_eig_values = eig_values[sorted_indices]
    sorted_eig_vectors = eig_vectors[:, sorted_indices]

    ##The state we plot
    state_vec = sorted_eig_vectors[:, state_num] 

    #Setup the plot
    sites = np.arange(1, len(eig_vectors) + 1, 1) 
    site_prob = (state_vec * state_vec.conj()).real #only take real to get rid of the numerical imprecision

    #The phase 
    phases = []
    for i in range(len(eig_vectors)):
        r, theta = cmath.polar(state_vec[i])
        phases.append(theta)

    phases = np.asarray(phases)
    normalized_phases = (phases + np.pi) / (2 * np.pi)  # normalize to [0, 1]

    #create colorbar
    if hue: 
        hsv_cmap = plt.cm.hsv
        colors = hsv_cmap(normalized_phases)
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
        sm = cm.ScalarMappable(cmap=hsv_cmap, norm=norm)
    else:
        cyclic_two_color = LinearSegmentedColormap.from_list("cyclic_two", ["blue", "red", "blue"])
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
        colors = cyclic_two_color(normalized_phases)
        sm = cm.ScalarMappable(cmap=cyclic_two_color, norm=norm)

    #Plotting
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(sites, site_prob, color=colors, label = r'$\mid c_j \mid^2$')
        
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    cbar.set_label("Phase", fontsize=12, labelpad=10)

    plt.xlabel('Site')
    plt.ylabel(r'$\mid c_j \mid^2$')
    plt.title(rf'$\xi = {state_num + 1}$')
    #plt.legend()
    plt.show()

def SiteAmplitudeDistribution(eig_values, eig_vectors, state_num, hue=False, figsize = (7,5)): 
    #sort eigenvalues and eigenvectors. Lowest index being most sub-radiant and highest subradiant 
    sorted_indices = np.lexsort((eig_values.real, np.abs(eig_values.real)))
    sorted_eig_values = eig_values[sorted_indices]
    sorted_eig_vectors = eig_vectors[:, sorted_indices]

    ##The state we plot
    state_vec = sorted_eig_vectors[:, state_num] 

    #Setup the plot
    sites = np.arange(1, len(eig_vectors) + 1, 1) 
    site_coeff  = state_vec.real #only take real to get rid of the numerical imprecision

    #The phase 
    phases = []
    for i in range(len(eig_vectors)):
        r, theta = cmath.polar(state_vec[i])
        phases.append(theta)

    phases = np.asarray(phases)
    normalized_phases = (phases + np.pi) / (2 * np.pi)  # normalize to [0, 1]
    
    if hue: 
        hsv_cmap = plt.cm.hsv
        colors = hsv_cmap(normalized_phases)
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
        sm = cm.ScalarMappable(cmap=hsv_cmap, norm=norm)
    else:
        cyclic_two_color = LinearSegmentedColormap.from_list("cyclic_two", ["blue", "red", "blue"])
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
        colors = cyclic_two_color(normalized_phases)
        sm = cm.ScalarMappable(cmap=cyclic_two_color, norm=norm)


    #Plotting
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(sites, site_coeff, color=colors, label = r'$ c_j $')

    # Create and add colorbar
    
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    cbar.set_label("Phase", fontsize=12, labelpad=10)

    plt.xlabel('Site')
    plt.ylabel(r'$Re(c_j)$')
    plt.title(rf'$\xi = {state_num + 1}$')
    #plt.legend()
    plt.show()

def SiteProbAndAmplitudeDistribution(eig_values, eig_vectors, state_num, titleprecision, hue=True, sharey = True, interrupted_array = False, is_ring = False, is_hedgehog = False, is_star=False, return_sorted_eigval = False): 
    #sort eigenvalues and eigenvectors. Lowest index being most sub-radiant and highest subradiant 
    sorted_indices = np.lexsort((eig_values.real, np.abs(eig_values.real)))
    sorted_eig_values = eig_values[sorted_indices]
    sorted_eig_vectors = eig_vectors[:, sorted_indices]

    ##The state we plot
    state_vec = sorted_eig_vectors[:, state_num] 
    state_vec_decay_rate = np.round(-2 * sorted_eig_values[state_num].real , titleprecision)

    #Setup the plot
    if interrupted_array: 
        sites = np.arange(0, len(eig_vectors) + 1, 1)  #one extra so we can plot central site with 0 amplitude
        missing_site = len(eig_vectors) // 2 #return the center index
        
        site_coeff_temp  = state_vec.real 
        site_prob_temp = (state_vec * state_vec.conj()).real 
        #Add center with value 0
        site_coeff = np.concatenate([site_coeff_temp[:missing_site], [0], site_coeff_temp[missing_site:]])
        site_prob = np.concatenate([site_prob_temp[:missing_site], [0], site_prob_temp[missing_site:]])
    else:     
        sites = np.arange(1, len(eig_vectors) + 1, 1) 
        site_coeff  = state_vec.real #only take real to get rid of the numerical imprecision
        site_prob = (state_vec * state_vec.conj()).real #only take real to get rid of the numerical imprecision


    #The phase 
    phases = []
    for i in range(len(eig_vectors)):
        r, theta = cmath.polar(state_vec[i])
        phases.append(theta)

    phases = np.asarray(phases)
    normalized_phases = (phases + np.pi) / (2 * np.pi)  # normalize to [0, 1]
    
    if hue: 
        hsv_cmap = plt.cm.hsv
        colors = hsv_cmap(normalized_phases)
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
        sm = cm.ScalarMappable(cmap=hsv_cmap, norm=norm)
    else:
        cyclic_two_color = LinearSegmentedColormap.from_list("cyclic_two", ["blue", "red", "blue"])
        norm = Normalize(vmin=-np.pi, vmax=np.pi)
        colors = cyclic_two_color(normalized_phases)
        sm = cm.ScalarMappable(cmap=cyclic_two_color, norm=norm)


    #Plotting
    fig, axs = plt.subplots(1, 2, figsize = (12,5), sharey=sharey)
    fig.subplots_adjust(wspace=0.3)
    bars_prob = axs[0].bar(sites, site_prob, color = colors) #label = r'$ |c_j|^2 $'
    bars_site = axs[1].bar(sites, site_coeff, color=colors) #label = r'$ c_j $'

    fontsize = 18
    # Create and add colorbar
    
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axs, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    cbar.set_label("Phase", fontsize=fontsize, labelpad=10)

    axs[0].set_xlabel('Site', fontsize = fontsize )
    axs[0].set_ylabel(r'$|c_j|^2$', fontsize = fontsize)
    axs[1].set_xlabel('Site', fontsize = fontsize)
    axs[1].set_ylabel(r'Re($c_j$)', fontsize = fontsize)
    
    if is_star:
        axs[0].axvline(x=sites[len(sites)//2] -0.5, color='grey', linestyle='--', linewidth=0.5, label = 'Seperation of inner and outer ring')
        axs[1].axvline(x=sites[len(sites)//2] -0.5, color='grey', linestyle='--', linewidth=0.5, label = 'Seperation of inner and outer ring')
        plt.legend()

    
    if interrupted_array: 
        fig.suptitle(rf'$\xi = {state_num + 1}$ for interrupted, $\Gamma / \Gamma_0 = {state_vec_decay_rate}$', x = 0.43, fontsize = fontsize + 2)
    elif is_ring: 
        fig.suptitle(rf'$\xi = {state_num + 1}$ for ring, $\Gamma / \Gamma_0 = {state_vec_decay_rate}$', x = 0.43, fontsize = fontsize + 2)
    elif is_hedgehog:
        fig.suptitle(rf'$\xi = {state_num + 1}$ for hedgehog, $\Gamma / \Gamma_0 = {state_vec_decay_rate}$', x = 0.43, fontsize = fontsize + 2)
    elif is_star: 
        fig.suptitle(rf'$\xi = {state_num + 1}$ for star, $\Gamma / \Gamma_0 = {state_vec_decay_rate}$', x = 0.43, fontsize = fontsize + 2)
    else:
        fig.suptitle(rf'$\xi = {state_num + 1}$ for array, $\Gamma / \Gamma_0 = {state_vec_decay_rate}$', x = 0.43, fontsize = fontsize + 2)
    #plt.legend()
    plt.show()

    if return_sorted_eigval:
        return state_vec, site_coeff, sorted_indices, sorted_eig_values, sorted_eig_vectors