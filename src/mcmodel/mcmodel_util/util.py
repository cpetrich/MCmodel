import numpy as np

def make_phase_function(pf, parameters=None, angle_steps=None):
    # default parameters turn eveything isotropic
    try: PARAMETER_1 = parameters[0]
    except: PARAMETER_1 = 0.
    try: PARAMETER_2 = parameters[1]
    except: PARAMETER_2 = 0.

    # Use odd number of steps to get the random number generator symmetric
    if angle_steps == None: phi_steps = 4097
    else: phi_steps = angle_steps    
        
    # The phase function determines the rotation of a vector
    #  incident to a scatterer.
    # An "elevation angle" (more like a polar angle)
    #  of 0 indicates forward scattering.
    # We use the range from -pi to +pi (instead of 0 to +pi)
    #  as the original version of this program was written
    #  for scattering in 2D. (legacy, legacy...)
    Phi = np.linspace(-np.pi, np.pi, phi_steps, endpoint=True)
    Phi_ctr = .5*(Phi[1:]+Phi[:-1])

    # default (error) values:
    p = None # will raise an error below
    pf_identifed = 'None'

    if pf == 'isotropic':
        pf_identified = 'isotropic'
        p=1.
        expected_g = 0.

    if 'Eddington' in pf:
        g_prime = PARAMETER_1
        if g_prime > 1./3.:
            print("Warning: g'=%g out of range in Eddington phase function" % g_prime)
        if 'delta' in pf:
            pf_identified = 'delta-Eddington'
            f = PARAMETER_2
        else:            
            pf_identified = 'Eddington'
            f = 0. # no forward peak

        # Eddington:
        p = (1-f)*(1+3.*g_prime*np.cos(Phi_ctr))
        # delta-Eddington
        zero_phi = np.min(np.abs(Phi_ctr))
        zero_idx = np.nonzero(np.abs(np.abs(Phi_ctr)-zero_phi) < 1e-12)[0]        
        # two equivalent formulations of the delta function:
        # 1.
        scale = 2/len(zero_idx) * (phi_steps-1)**2/(np.pi*(max(Phi)-min(Phi)))
        # 2.
        dPhi = Phi[max(zero_idx)+1]-Phi[min(zero_idx)]
        scale = 2*(phi_steps-1)/(np.pi*dPhi)
        
        p[zero_idx] += 2.*f *scale

        p[p<0] = 0. # no negative probabilities

        expected_g = f+(1-f)*g_prime
    
    if ('Henyey' in pf) and ('Greenstein' in pf):
        # (can also be inverted analytically)
        g_HG = PARAMETER_1
        if 'modified' in pf:
            beta = PARAMETER_2
            pf_identified = 'modified-Henyey-Greenstein'
        else:
            beta = 0. # no isotropic component
            pf_identified = 'Henyey-Greenstein'

        # Henyey-Greenstein
        p = (1-g_HG**2) / (1+g_HG**2-2*g_HG*np.cos(Phi_ctr))**1.5
        # modified Henyey-Greenstein
        p = beta + (1-beta)*p

        expected_g = (1-beta)*g_HG
       
    # Phi==0 is forward scattering (i.e. no rotation)
    # Phi==+-pi is back-scattering
    # Phi=+-pi/2 is normal scattering

    # the azimuth rotation is uniformly distributed
    #   --> scale probability with |sin(Phi)|
    # (either one of the following works)
    if True:
        p *= np.abs(np.sin(Phi_ctr))
    else:
        p *= .5*(np.abs(np.sin(Phi[1:]))+np.abs(np.sin(Phi[:-1])))
    # get normalized cumulative density function (CDF)
    #  at Phi
    cum_p = np.cumsum(p) / np.sum(p)
    # CDF at the first angle is 0 by definition
    cum_p = np.append([0], cum_p)

    if True:
        # check mean cosine:
        # probability that the random number generator
        # selects ...
        probe_p = np.diff(cum_p)
        # ... a particular polar angle
        probe_Phi = .5*(Phi[1:]+Phi[:-1])
        # Summed up:
        mean_cos = np.sum( np.cos(probe_Phi) * probe_p )
    else: mean_cos = None
    
    cum_p[-1] = 1. if abs(cum_p[-1]-1.) < 1e-12 else cum_p[-1]
    
    
    phase_function_definition = {'pf':pf_identified, # name of phase function (for reference only)
                                 'g-param':PARAMETER_1, # parameter g (for reference only)
                                 'f/beta-param':PARAMETER_2, # parameter f or beta (for reference only)
                                 'phi_steps':phi_steps, # number of entries in lookup table (for reference only)
                                 'expected_g':expected_g, # expected effective mean cosine (for reference only)
                                 'measured_g':mean_cos, # calculated effective mean cosine (for debugging only)
                                 'lookup_phi': Phi.copy(), # angle array of lookup table (important: passed on to scattering code)
                                 'lookup_cdf': cum_p.copy()} # probability array of lookup table (important: passed on to scattering code)
    # return parameters for future reference
    return phase_function_definition
