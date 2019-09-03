# MCmodel
A 3D Monte Carlo scattering code to keep track of individual parcels in a plane-parallel homogenous slab

The theory behind the model, many of its parameters, and model validation have been
described by Petrich et al. (2012), https://doi.org/10.1016/j.coldregions.2011.12.004

The Monte Carlo code itself is implemented in C for speed and integrated
with a Python interface for convenience.

License: Apache 2.0.

## Installation

### Requirements

* Python 2.7 and 3.6 are known to work on Linux and Windows.
* C-compiler. On Linux, gcc is usually installed.
  On Windows, install the free MS Visual Studio C compiler in
  a version compatible with the target Python version.
* To build wheels and install with pip: pip, setuptools, wheel

### Use without installation

Run `python setup.py build` and copy the contents of the `built/lib.[...]`
directory into the same directory as your own modeling code.

### Installation with pip

The module is currently not on PyPi, so a wheel has to be built
locally and installed with pip. Build the module and create a wheel file
with `python setup.py bdist_wheel`, `cd dist`,
`pip install MCmodel-3.[...].whl`. To uninstall the package later run
`pip uninstall mcmodel`.

### Test

To test the result of the build process, a rather incomplete test suite
can be run with `python setup.py test`.

## Limitations

The code is currently not reentrant since it uses global state for the random
number generator, the phase function, and buffers that hold detailed information
about the most recent particle track. To take advantage of multitasking capabilities
of recent processors, multiple invocations of this model can be run safely in
separate processes. In this case it would be prudent to ensure that each invocation
of the model is seeded uniquely, e.g. through
```
import os
import mcmodel
mcmmodel.set_seed(int.from_bytes(os.urandom(4),'little'))
```

The special case - in a scattering model - of "no scattering" (`w0 == 0`) is
implemented only for `k == 0` (i.e. no absorption).


## Usage

The usual mode of operation is:

0. seed the random number generator
1. define a scattering phase function
2. call the scattering code with slab optical properties, and position and direction of one parcel of light
3. store entrance and exit positions and angles of the parcel and information about plane crossings inside the medium
4. if desired, store complete track of the light parcel
5. continue with (2) a couple of thousand or million times.
6. read the simulation results from disk, and determine parcel intensity due to absorption and Fresnel reflection
7. analyze result as desired

An example of seeding the random number generator of the Monte Carlo model is
```
import os
import mcmodel
mcmmodel.set_seed(int.from_bytes(os.urandom(4),'little'))
```

A helper function exists to generate common scattering phase functions, e.g.
```
import mcmodel_util
pf = mcmodel_util.make_phase_function('Henyey-Greenstein', (0.98, 0))
```

The phase function is transfered to the model code like
```
mcmodel.define_phase_function(pf['lookup_cdf'],pf['lookup_phi'])
```

A random number generator is usually used to generate an incoming distribution
(that random number generator is separate from the one used by the Monte Carlo
model and should also be seeded).
```
import random
random.seed()
```

A sample from the angular distribution of a Lambert emitter
(i.e., overcast over snow-covered ground) is
```
import math

elevation_angle = math.arcsin(random.uniform(-1., 0))
azimuth = random.uniform(-math.pi, math.pi)
```
Note that the Monte Carlo code works in **elevation** angles. A normally
incident beam at the upper surface of the slab has an elevation angle
of -pi/2.

The domain parameters and incident angle are passed on to the Monte Carlo
code in a dictionary, e.g.
```
in = {'angle': elevation_angle,
      'azimuth': azimuth,
      'thickness': 1, # slab thickness
      'k': 1}         # densiy of microscopic interactions used by the model
```
with additional optional input parameters described below. Many of the output
data are obtained through updates to a dictionary that is passed to the scattering
simulation. To calculate the track of a single parcel call
```
out = {}
mcmodel.simulate(in, out)
```
where exit position and angles are
```
exit_position = (out['x'], out['y'], out['z'])
exit_elevation_angle = out['angle']
exit_azimuth = out['azimuth']
path_length = out['path_length']
```

Attenuation of the parcel inside the medium is accounted for by decreasing its
amplitude. The precise method of this depends on the definition and units of
thickness and microscopic extinction coefficient passed into the model above.
However, the intensity of the emerging parcel will be of a form similar to
```
exit_intensity = math.exp(-kappa * path_length)
```
where kappa is the absorption coefficient.

Intensity reduction due to Fresnel reflection (other than total reflection)
would be accounted for based on entrance and exit angles and relative refractive
indices. To do this in hindsight is not trivial and requires the simulations
to be run with a relative refractive index of 1 and appropriate scaling of
the input angular distribution to account for non-trivial refractive indices.
This is outside the scope of this overview.
*[TODO: in a future version, add an option to have Monte Carlo model take care
of this by selecting a path stochastically.]*

### Parameters

The model domain is a horizontal slab of thickness `thickness`, extending from
`z=0` at the upper surface to `z=-thickness` at the bottom. Parcles are injected
at depth `z=z0` (`-thickness <= z0 <= 0`) and `x=0` and `y=0`. Injection at
`z0 = -thickness` and `z0 = 0` are on the outside of the slab at the lower and
upper interface, respectively (i.e. still subject to refraction).

If an anisotropic scattering coefficient is used, i.e. `sigma_anisotropy != 0`,
then the vertical and horizontal scattering coefficients are
`sigma_v = k * w0` and `sigma_h = k * w0 * (sigma_anisotropy+1)`, respectively.


#### Input Dictionary of `mcmodel.simulate()`

Name | Default | Meaning
-----|---------|--------
`angle` | *none* | elevation angle of incident parcel [-pi; pi]
`azimuth` | *none* | azimuth angle of incident parcel [-pi; pi)
`z0` | `0` | vertial coordinate of parcel injection
`thickness`| *none* | slab thickness
`k` | *none* | microscopic extinction coefficient
`w0` | `1` | single scattering albedo
`n_surface` | `1` | refractive index `n_slab / n(z>0)`
`n_bottom` | `1` | refractive index `n_slab / n(z<-thickness)`
`sigma_anisotropy` | `0` | anisotropy of scattering coefficient (cf. Trodahl et al.), `sigma_anisotropy = sigma_h/sigma_v - 1`
`do_record_track` | `False` | log coordinates of scattering the parcel. Retrieve with `mcmodel.get_last_particle_track()`
`do_record_plane_crossings` | `False` | log position and angle of parcel every time it crosses specified planes
`record_planes` | `[]` | list of `z` coordinates to record parcel crossings at

#### Output Dictionary of `mcmodel.simulate()`

Name | Meaning
-----|--------
`angle_in` | copy of `angle` in input dictionary
`azimuth_in` | copy of `azimuth` in input dictionary
`x` | lateral exit coordinate, same unit as `thickness` in input
`y` | lateral exit coordinate, same unit as `thickness` in input
`z` | vertical exit coordinate, same unit as `thickness` in input
`path_length` | total path length of parcel in the medium, same unit as `thickness` in input
`exit_direction` | `+1`: exit at `z=0`, `-1`: exit at `z=-thickness`
`min_z` | lowest `z` coordinate along the track
`max_R` | largest value of `sqrt( x^2 + y^2 )` along the track
`n_collisions` | number of scattering events along the path (NB: not total reflection)
`n_missed_collisions` | number of absorption events along the path, all of which have been ignored (count is slightly too high if `w0>0`)
`n_refractions` | number of refractions experienced along the path
`n_total_reflections` | number of total reflections along the path
`n_plane_crossings` | number of plane crossing events along the path
`plane_crossings` | 2D `numpy` array containing 6 entries (columns) per crossing event (see below)

The six column entries for each plane crossing are:
1. elevation angle
2. azimuth angle
3. x-coordinate at the point of crossing
4. y-coordinate at the point of crossing
5. z-coordinate at the point of crossing
6. particle path length so far
