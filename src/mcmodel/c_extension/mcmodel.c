/*
Copyright (c) 2011-2019, Chris Petrich

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/* This is file 'mcmodel.c' of the Monte Carlo Scattering Code. */

/* Notes:
    * Compile c module with
      python setup.py build
    * This code uses global state and is not thread safe.
    * If run as a batch job, explicitly set the random number generator seed
      to unique values.    
    * Even if the medium is defined as absorptive, the light parcel is actually
      never stochastically extinguished. Hence, Intensity reduction (or
      stochastic parcel extinction) has to be performed in post-processing.
    * Treatment of Fresnel reflection at the interfaces is limited to total 
      reflection vs refraction. Intensity reduction due to partial reflection,
      or stochastic selection of the reflected vs refracted path needs to be 
      performed in post-processing.
    * Elevation angles of the path are used rather than zenith angles.
    * Note that vertical coordinate is called 'z' in the user interface and
      'y' in the scattering code.
    * In the unlikely event that the C code runs out of memory (i.e., malloc()
      or realloc() fail), the program will eventually segfault rather than
      emit an error.
*/

/* Version History

    3.0 (3 Sept 2019)     
     * Breaking change:
       + Renamed parameter 'depth' to 'thickness' to be consistent with Petrich et al. (2012).
       + Renamed many other names in input and output dictionary.     
     * Allow z0 starting position inside the scattering medium, i.e. -thickness <= z0 <= 0. Previously, any value <0 was assumed to be outside the
       scattering domain since there was no perceived need for in-medium injection. As a result, parcel injection at the bottom interface 
       (i.e., outside the medium) has to be exactly, to "double" precision, at -thickness to be detected as such.
     * Add special case treatment for w0 == 0. Still, the model won't terminate if elevation angle is ever exactly 0.
     * Implented special case w0=0 and k=0.
     * Reject case of w0=0 with k>0 as this is not implemented.
     * Note that the count of missed collisions is an overestimate on the final leg intersected by a boundary.
     * Add function for unit testing that deallocates all global memory.
     * Verify that parameter z0 is -thickness <= z0 <= 0.
     * Exported function to set seed of random number generator.
     * Consider that PyDict_Contains() could return -1 (although not here: we check only for string keys) and test for 1 explicitly.     
     * Changed setup.py to use setuptools if installed in order to be able to build binary wheels.     
     * Properly test return value of PyDict_Contains().
     * Allocated more global string objects to avoid repeated creation and deletion of temporary string objects. I.e., eliminated 
       calls to PyDict_GetItemString().
     * Eliminated dead code and obsolete comments from source.     
     * Tidied up revision history and introduced Semantic Versioning.
     
    2.0 (2 Jan 2019)
     * Version contains breaking changes:
       + Renamed scattering parameters in input dictionary.
       + Renamed output dictionary keys 'travel_distance' -> 'path_length', 'max(R)' -> 'max_R', 'min(z)' -> 'min_z'.
       + If particle does not enter the medium due to total reflection, vertical direction of travel is now reversed before returning results.
     * Added __version__ string.
     * Added parameter 'z0' to input_dict to specify the vertical plane the light parcel enters. 
       The domain is now always from -depth to 0 and -depth < 0 is assumed in the code.
     * Refactored anisotropy treatement of scattering to allow for finite absorption (which is not anisotropic).
     * Included check for total reflection at the start of the simulation (also) at the bottom interface.
     * Remove depreciated Numpy API to avoid compile-time warnings: NPY_IN_ARRAY -> NPY_ARRAY_IN_ARRAY, 
       explicit typecasts between (PyArrayObject*) and (PyObject*).
       As a result, required version of Numpy is now >=1.7.
     * Input 'record_plane' can now be a list as alternative to Numpy array.
     * Output array 'plane_crossings' is now 2D (was: 1D).
     * Corrected count of number of plane crossings by changing division by 5 to division by 6.
     * Made parameter 'n_surface' optional.
     
    1.2 (8 Nov 2017)
     * Made C-code compatible with Python 3 (specifically, Python 3.6), https://docs.python.org/3/howto/cporting.html
     * Fixed memory leak by allocating dictionary keys with PyString_FromString() globally.
     * Extended error messages to include dictionary parameters.
     * Changed most integers to 'long long' to suppress warnings.
     * Changed PyArray_XDECREF_ERR(out) to Py_XDECREF(out) since 'out' is not an array.       
     
    1.1 (25 Jan 2013)
     * Added optional parameter n_bottom for refraction and total reflection at the bottom interface.

    1.0 (21 Oct 2011)
     * Initial Release
     * Code used by Petrich et al. (2012), Sensitivity of the light field under sea ice 
       to spatially inhomogeneous optical properties and incident light assessed with 
       three-dimensional Monte Carlo radiative transfer simulations, Cold Regions Science
       and Technology, 73, 1-11, https://doi.org/10.1016/j.coldregions.2011.12.004     
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define MODULE_VERSION_STRING "3.0"

#include <Python.h>
/* arrayobject.h is located in Python25/Lib/site-packages/numpy/core/include/numpy/ */
/* #include "arrayobject.h" */
#include <arrayobject.h>

/* include trig functions and square root */
#include <math.h>

/* seed for random number generator */
#include <time.h> 
/* random number generator */
#include "dSFMT.h"

#define TRUE 1
#define FALSE 0

#define PI 3.1415926535897932384626433832795

#ifndef min
/* MSVC defines these for us */
#define min(a,b) ((a)>(b)?(b):(a))
#define max(a,b) ((a)<(b)?(b):(a))
#endif
#define sign(x) (( (x) > 0 ) - ( (x) < 0 ))

#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong(x) PyLong_FromLong(x)
#define PyInt_AsLong(x) PyLong_AsLong(x)
#define PyString_FromString(x) PyUnicode_FromString(x)
#endif

/* global variable to hold the state of the random number generator */
dsfmt_t dsfmt;

/* global lookup table of cumulative scattering function of isotropic scatterers */
double *lookup_cum_p = NULL; /* NB: lookup table has to include a value for _both_ 0 and 1 */
double *lookup_Phi = NULL;
long long lookup_length = 0;

long long do_record_planes = FALSE;
long long RECORD_N_PLANES = 0;
/* Arbitrarily limit the number of crossing planes to be able to allocate */
/* fixed amount of memory rather than call malloc() repeatedly.           */
/* Realistically, we won't need more than 10 to 20.                       */
#define MAX_RECORD_N_PLANES 1024
double record_plane[MAX_RECORD_N_PLANES];

/* 6 doubles per record: angle,azimuth, x,y,z, and path length */
/* memory will be allocated as needed but never freed          */
double *record_plane_crossings;
long record_index=0; /* next index to fill */
long record_length=0; /* memory allocated */

/* particle track log */
/* 3 doubles per entry: x, y, z                       */
/* Memory will be allocated as needed and never freed */
long do_track_particles = FALSE;
double *track_log_buffer = NULL;
long track_log_index=0; /* next index to fill */
long track_log_buffer_length=0; /* memory allocated */


int track_log_reset(void)
{
	/* reset track log index to zero, */
	/* effective 'freeing' all memory */
	track_log_index = 0;
	return 0;
}

int track_log_append(double x, double y, double z)
{
	int N = 3;
	/* for speed reasons, never de-allocate memory */
	
	/* allocate memory for buffer */
	if (track_log_index > track_log_buffer_length-N) {
		if (track_log_buffer_length>0) {
			/* this should rarely happen */
			track_log_buffer = (double*)realloc(track_log_buffer, sizeof(double) * (N * 4096 + track_log_buffer_length) );
            if (!track_log_buffer) return 1; /* error */
			/* if there's a problem with realloc() record_plane_crossings==NULL we'll seg-fault below */
			track_log_buffer_length += N*4096;
		} else {
			track_log_buffer = (double*)malloc(sizeof(double)* N * 4096);
            if (!track_log_buffer) return 1; /* error */
			track_log_buffer_length = N*4096;
		}						
	}

	/* enter new coordinates to log */
	track_log_buffer[ track_log_index+0 ] = x;
	track_log_buffer[ track_log_index+1 ] = y;
	track_log_buffer[ track_log_index+2 ] = z;
	
	track_log_index += 3;

	return 0;
}


int init_random_number_generator(uint32_t seed)
{	
	/* call this function in the main procedure */
	dsfmt_init_gen_rand(&dsfmt, seed);	
	return 0;
}

double get_random_number(void)
{
	/* returns number [0;1) */
	return dsfmt_genrand_close_open(&dsfmt);
}

double expovariate(double lambd)
{
	/* lambd is inverse scale */
	/* invert CDF, 1-exp(-lambda x) */
	return log( 1 - get_random_number() ) / (-lambd);
}

double get_distance_to_collision(double prob)
{
	/* average is 1/prob */
	/* random.expovariate(prob) */
    return expovariate(prob);
}

void clear(void) {
    /* free memory and random number generator state to be able to start with a clean slate in unit tests */
    free(track_log_buffer); /* safe to do on a NULL pointer */    
    track_log_buffer = NULL;
    track_log_buffer_length = 0;
    track_log_index = 0;
    do_track_particles = FALSE;
    
    free(record_plane_crossings);
    record_plane_crossings = NULL;
    record_length = 0;
    record_index = 0;

    free(lookup_cum_p);
    lookup_cum_p = NULL;    
    free(lookup_Phi);
    lookup_Phi = NULL;
    lookup_length = 0;    
    
    long i;
    for (i = 0; i < MAX_RECORD_N_PLANES; i++)
        record_plane[i] = 9999999;    
    RECORD_N_PLANES = 0;
    do_record_planes = FALSE;
    
    uint32_t seed = (uint32_t)time(NULL);
	init_random_number_generator( seed );
}

int refract(double *angle, double n, int is_inside, int *total)
{
    /* Apply Snell's law */
	/* angle is measured with respect to the horizontal */
	double angle_in = *angle;
	if (is_inside) n = 1/n;
	while (angle_in >= PI) angle_in -= 2*PI;
	while (angle_in < -PI) angle_in += 2*PI;
	
	double arg = cos(angle_in) /n;
	
	if ((arg > 1.0) || (arg < -1.0)) {
		/* total reflection */
		/* reverse vertical component, keep azimuth */
		double angle_out = -angle_in;
		
		*total = TRUE;
		*angle = angle_out;
	} else {	
		double angle_out = acos( arg );
		while (angle_out >= PI) angle_out -= 2*PI;
		while (angle_out < -PI) angle_out += 2*PI;
		
		if (angle_in < 0)
			angle_out *= -1;
			
		*total = FALSE;
		*angle = angle_out;
	}
	return 0;
}

double get_random_angle(void)
{			
	double a = get_random_number();

	long long idx=lookup_length / 2; /* odd number divided by two */
	long long step = max(1, lookup_length / 4);

    /* find index in lookup table through successive approximation */
	while ((idx < lookup_length) && (idx>0)) {
		if ((lookup_cum_p[idx-1] < a) && (lookup_cum_p[idx] >= a)) break; /* found index */
		
		if (lookup_cum_p[idx] >= a) idx -= step;
		else idx += step;
				
		step = max(1, step / 2);
		
		if ((idx>=lookup_length) && (step > 1)) {
			step = 1;
			idx = lookup_length-1;
		}
		
		if ((idx<=0) && (step > 1)) {
			step = 1;
			idx = 1;
		}
	}
	
	/* a is in array --> no interpolation */
	if (lookup_cum_p[idx] == a)
		return lookup_Phi[idx];
	
	/* a is larger than largest element in array --> interpolate as constant value */
	/*  better option would be to raise an error */
	if (lookup_cum_p[idx] < a)
		return lookup_Phi[idx];

	/* linearly interpolate angle between [idx] and [idx-1] */
	double m = (a - lookup_cum_p[idx-1]) / (lookup_cum_p[idx] - lookup_cum_p[idx-1]);
	double Phi = lookup_Phi[idx-1] + m*(lookup_Phi[idx]-lookup_Phi[idx-1]);
		
	return Phi;	
}

int get_normal_in_xy_plane(double azimuth, double *U)
{
	azimuth += PI/2;
	U[0] = cos(azimuth);
	U[1] = sin(azimuth);
	U[2] = 0;
	
	return 0;
}

int rotate_3D(double *X, double *U, double phi, double *X_prime)
{
	/* U is a unit vector */
	double c = cos(phi);
	double s = sin(phi);
	double c1 = 1-c;
	
	X_prime[0] = 	(c+c1*U[0]*U[0])      *X[0]+
					(U[0]*U[1]*c1-U[2]*s) *X[1]+
					(U[0]*U[2]*c1+U[1]*s) *X[2];
					
	X_prime[1] = 	(U[1]*U[0]*c1+U[2]*s) *X[0]+
					(c+c1*U[1]*U[1])      *X[1]+
					(U[1]*U[2]*c1-U[0]*s) *X[2];
					
	X_prime[2] = 	(U[2]*U[0]*c1-U[1]*s) *X[0]+
					(U[2]*U[1]*c1+U[0]*s) *X[1]+
					(c+c1*U[2]*U[2])      *X[2];
	
	return 0;
}

int rotate_vector(double elevation_in, double azimuth_in,
		double elevation, double azimuth,
		double *elevation_out, double *azimuth_out )
{
	double X[3];
	double u[3];
	double X_prime[3];
	double OUT[3];
	/* equations from Wikipedia */
    
    /* get unit vector of pre-scattered radiation: */
    /* note: elevation is above XY plane */
    double z = sin(elevation_in);
    double x = cos(elevation_in) * cos(azimuth_in);
    double y = cos(elevation_in) * sin(azimuth_in);
    
	X[0] = x; X[1] = y; X[2] = z;
    double R = sqrt(x*x+y*y+z*z);

    /* get axis to rotate X around: */
    /* u = get_normal_in_xy_plane(azimuth_in) */
	get_normal_in_xy_plane(azimuth_in, (double*)&u);
    
    /* rotate elevation around this axis */
	rotate_3D((double*)&X, (double*)&u, elevation, (double*)&X_prime);
    
    /* now rotate around original direction */
	X[0] /= R; X[1] /= R; X[2] /= R; /* make unit vector */
    rotate_3D((double*)&X_prime, (double*)&X, azimuth, (double*)&OUT);
	double x_out = OUT[0]; double y_out = OUT[1]; double z_out = OUT[2];

    /* recover elevation and azimuth */
    *elevation_out = asin(z_out); /* returns -pi/2 to pi/2 ONLY */        
    *azimuth_out = atan2(y_out, x_out);
	return 0;
}

int get_angle_after_scattering(double angle_in, double azimuth_in, double *angle_out, double *azimuth_out)
{
	/* isotropic scatterers */
	/* get angle with PDF according to scattering function: */
	double forward = get_random_angle();
	double azimuth = get_random_number() * 2*PI - PI;
	
	rotate_vector(angle_in,azimuth_in, forward, azimuth, angle_out, azimuth_out);
	return 0;
}

int get_distance_3D_from_intercept(double surface_z, 
		double x, double y, double z, 
		double elevation, 
		double azimuth, /* only needed for sloped interfaces */
		double *over, 
		double *xx, double *yy, double *zz )
		/* NOTE: different x,y,z notation than in rest of program! */
{
	if ((elevation == 0) && (surface_z == z)) {
		/* special case: grazing angle in boundary plane */
		*over = 0;
		*xx = x;
		*zz = z;
		*yy = y;
		return 0;
	}

	/* by how much did we overshoot the surface? */
	/* TODO: pass both start and end coordinates to this function */
	/*   so we can get rid of trigonometric calculations */
	/*   that slow things down. */
	double distance = (surface_z - z) / sin(elevation);
	*over = fabs(distance);
	*xx = x + cos(azimuth) * cos(elevation) * distance;
	*yy = y + sin(azimuth) * cos(elevation) * distance;
	*zz = surface_z;
	
	return 0;
}

int track_particle(double angle, double azimuth, double k, double depth, 
					double n_surface, double n_bottom, /* n_bottom NEW-130125 */
					double w0,
					double sigma_anisotropy,
                    double vertical_pos0, // new 2 Jan 2019. Default: 0
					double *return_stats, long *return_count)
{	
    /* n_.... = n_medium/n_outside  */
	/* n_surface = 1.33 is the refractive index at the surface, meaning medium is ice beneath air */
	/* n_bottom = 1.33 is the refractive index at the bottom, meaning medium is ice overlying air */
    		
	double x = 0; /* to the right */
	double y = vertical_pos0; /* upward */
	double z = 0; /* to the front */
	
	/* these are the points of intersection at the boundary: */
	double xx=x;
	double yy=y;
	double zz=z;
	
	double x_entrance = x;
	double z_entrance = z;
	
	/* scatter path characteristics. Note that vertical coordinate is 'y'. */
	double min_y = y; /* lowest point in path (makes sense only on injection at y=0) */
	double max_R = 0; /* furthest radial position from entrance location (x=0, z=0) */
	
	double current_angle=-1e10, current_azimuth=-1e10;
		
	double pos_surface_y0 = 0; /* entrance --> defaults to 0 */
	double pos_bottom_y0 = 0-depth; /* exit downward --> always at -depth */
	
	/* flags */
	int total; /* flag indicating total reflection */
	int move_inside_domain;
	long exit_direction_is_up;
	
	/* counters */
	long cnt_total_reflections = 0;
	long cnt_refractions = 0;
	long cnt_collisions = 0;
	long cnt_missed_collisions = 0; /* if w0<1, this counts up every time */
									/*   we would have an absorption event */
	double travel_distance = 0.0;
	
	
	if (do_track_particles) track_log_reset(); /* overwrite existing record */
	if (do_track_particles) track_log_append(x,z,y); /* log entrance position */
	
	int is_inside;
    if (y == pos_surface_y0) {
        is_inside = FALSE; /* we're above the surface */
        refract(&angle, n_surface, is_inside, &total);
    } else if (y == pos_bottom_y0) { /* explicit check as per 30 Aug 2019 */
        is_inside = FALSE; /* we're below the bottom */
        refract(&angle, n_bottom, is_inside, &total);
    } else if ((y < pos_surface_y0) && (y > pos_bottom_y0)) { /* new as per 30 Aug 2019 */
        /* starting inside domain */        
        total = FALSE; /* don't need to check for total refraction */        
        cnt_refractions -= 1; /* reduce by 1 since we'll increase it below (assuming particles entering at the interface) */
    } else {
        /* illegal parameters, should have been filtered before */
        return -1; /* new as per 30 Aug 2019 */
    }
	
	if (total) {		
		/* does not enter material */
		move_inside_domain = FALSE;        
		exit_direction_is_up = (y==pos_surface_y0); /* yes */
		cnt_total_reflections += 1;		
		current_angle = -angle; /* reflected */
		current_azimuth = azimuth;
	} else {
		move_inside_domain = TRUE; /* yes */
		cnt_refractions += 1;
	}
	
	int leaving_domain = FALSE;	
	int do_scatter_now = TRUE; /* default, unless we're coming from internal reflection */
	double l=0; /* travel distance */
	while (move_inside_domain) {
	    /* find location of anticipated next scattering event: */
        
		/* this is not being called after internal reflection */
		/* because remaining travel distance has already been determined */
		if (do_scatter_now) {
            /* anisotropy treatment to work in the presence of absorption */
            double current_k;
            double current_w0;
            if (sigma_anisotropy == 0) {
                current_k = k; // microscopic extinction coefficient independent of angle                
                current_w0 = w0;
            } else {
                /* back-calculate sigma and kappa */
                /* k = kappa + sigma */
                /* w0 = sigma/(sigma+kappa) = sigma / k */
                double sigma_v = k * w0;
                double kappa = k * (1-w0);
                /* sigma_anisotropy = sigma_h/sigma_v - 1  --> sigma_h = sigma_v * (sigma_anisotropy+1) */
                /*   (NB: Trodahl et al. define sigma_anisotropy = gamma/(1-gamma) ) */
                double sigma_h = sigma_v * (sigma_anisotropy+1);
                /* anisotropy definition of Trodahl et al.: */                
                double sigma = sigma_v + (sigma_h-sigma_v) * cos(angle);
                current_k = sigma + kappa;
                current_w0 = sigma / (sigma+kappa);
            }
            
			/* explicitly account for the fact that only */
			/* the fraction of w0 events actually lead to */
			/* scatter with change in direction */
			/* as the other ones are absorption */
			/* --> hence, we determine the distance to the */
			/*  true next scattering event */
			/* (and account for absorption later as continuous process) */
            
            if (current_w0 > 0) { /* infinite loop if w0 == 0 */
                l = 0; /* distance until next true scatter event */
                double not_scatter_prob = 1; /* if w0 > 1 then we'll keep scattering at the spot */
                while (not_scatter_prob >= current_w0) { /* always true the first time */
                    l += get_distance_to_collision(current_k);
                    
                    if (current_w0<1) not_scatter_prob = get_random_number(); /* this is always < 1, in [0; 1) */
                    else not_scatter_prob = 0; /* anything less than 1 will work */
                    
                    cnt_missed_collisions +=1;
                }
                cnt_missed_collisions -=1; /* since one is an actual collision, not a missed one */			
                /* cnt_missed_collisions may be too high if the path is cut by an interface */
            } else {
                l = 2*depth / sin(angle); /* we're out of luck if angle == 0 since the parcel will never leave */
                /* if (current_k > 0) */
                /*     then there may be missed collisions -- but we ignore those */
            }
		} /* conditional exists to be able to skip after internal reflection */
		
		do_scatter_now = TRUE; /* next time, scatter! */
		
		double x0=x, y0=y, z0=z; /* remember where we came from */
		
		y += l*sin(angle);
        x += l*cos(angle)*cos(azimuth);
        z += l*cos(angle)*sin(azimuth);
        
        travel_distance += l;

		current_angle = angle;
		current_azimuth = azimuth;
		
		if (do_record_planes) {
			/* check to see if we passed through a plane */			
			long long idx, idx_0, idx_max, idx_step, idx_sign;
			if (angle <= 0) {
				/* check array top-to-bottom */
				idx_0 = 0;
				idx_max = RECORD_N_PLANES;
				idx_step = 1;
				idx_sign = 1;
			} else {
				/* check array bottom-to-top */
				idx_0 = RECORD_N_PLANES-1;
				idx_max = 1; /* NB this is implicitly negated by idx_sign */
				idx_step = -1;
				idx_sign = -1;
			}
			
			/* fancy way of saying: for (idx = 0; idx< RECORD_N_PLANES; idx++) { */
			for (idx = idx_0; idx_sign*idx< idx_max; idx+=idx_step) {
				double m;
				if ((y-y0) != 0) m = (record_plane[idx]-y0) / (y-y0);
				else m = 1; /* movement in-plane: "crossing" at the final destination */
				if ((m<=1) && (m>0)) {
					/* crossing the plane (or ending on the plane) */
					
					/* allocate memory for buffer */
					if (record_index > record_length-6) {
						if (record_plane_crossings != NULL) {
							record_plane_crossings = (double*)realloc(record_plane_crossings, sizeof(double) * (6 * 4096 + record_length) );
							/* if record_plane_crossings==NULL we'll seg-fault below */
							record_length += 6*4096;
						} else {
							record_plane_crossings = (double*)malloc(sizeof(double)* 6 * 4096);
							record_length = 6*4096;
						}						
					}
					
					double xc = x0+m*(x-x0); /* azimuth = 0 */
					double yc = z0+m*(z-z0);
					double zc = record_plane[idx]; /* vertical */
					/* distance between calculated finish and intersection point */
					/* (Note confusing def of (y|z) and (zc|yc).) */
					
					double over = sqrt( (x-xc)*(x-xc)+(z-yc)*(z-yc)+(y-zc)*(y-zc) );
					/* -- same result but uses trigonometric functions -- */
					/*
					double over, xx,yy,zz;
					get_distance_3D_from_intercept( record_plane[idx], x,z,y, current_angle, current_azimuth, &over, &xx, &zz, &yy );
					*/					
					double L = travel_distance - over; /* correct for overshooting */
				
					record_plane_crossings[ record_index+0 ] = angle;
					record_plane_crossings[ record_index+1 ] = azimuth;
					record_plane_crossings[ record_index+2 ] = xc;
					record_plane_crossings[ record_index+3 ] = yc;
					record_plane_crossings[ record_index+4 ] = zc;
					record_plane_crossings[ record_index+5 ] = L;
					record_index += 6;
				}
			}			
		}
		
		/* will we intersect a boundary before we reach */
        /* the anticipated next scattering event? */
        int check_for_exit_at_boundary = TRUE;		
        while (check_for_exit_at_boundary) {
			/* run through this only once */
            check_for_exit_at_boundary = FALSE;
            /* unless we get total reflection, in which */
            /* case we'll test both boundaries again */
					
			/* upper surface, including test for total reflection */
			if (y > pos_surface_y0) {
				exit_direction_is_up = TRUE;
				double over; /* amount of overshooting */
				get_distance_3D_from_intercept( pos_surface_y0, x,z,y, current_angle, current_azimuth, &over, &xx, &zz, &yy );
				travel_distance -= over; /* correct for overshooting */
				
				/* now account for diffraction at the interface: */
				is_inside = TRUE;
				refract(&current_angle, n_surface, is_inside, &total);
				if (total == FALSE) {
					/* leaving the material */
					cnt_refractions += 1; /* exit is refraction at the bottom */
					leaving_domain = TRUE;
					check_for_exit_at_boundary = FALSE;
					break; /* leaving the material */
				}
				/* total reflection --> adjust position and continue */
				cnt_total_reflections += 1;
				if (TRUE) {
					/* keep movement */
					l = over;
					angle = current_angle;
					azimuth = current_azimuth;
					/* from new origin */
					y=yy;
					x=xx;
					z=zz;
					/* and don't calculate new scattering */
					do_scatter_now = FALSE;
				}					
			}
			/* bottom, also including test for total reflection */
			if (y < pos_bottom_y0) {			
				exit_direction_is_up = FALSE;			
				double over; /* amount of overshooting */
				get_distance_3D_from_intercept( pos_bottom_y0, x,z,y, current_angle, current_azimuth, &over, &xx, &zz, &yy );
				travel_distance -= over; /* correct for overshooting */
				
				/* inserted NEW-130125 */
				/* now account for diffraction at the interface: */
				is_inside = TRUE;
				refract(&current_angle, n_bottom, is_inside, &total);
				if (total == FALSE) {
					/* leaving the material */
					cnt_refractions += 1; /* exit is refraction at the bottom */
					leaving_domain = TRUE;
					check_for_exit_at_boundary = FALSE;
					break; /* leaving the material */
				}
				/* total reflection --> adjust position and continue */
				cnt_total_reflections += 1;
				if (TRUE) {
					/* keep movement */
					l = over;
					angle = current_angle;
					azimuth = current_azimuth;
					/* from new origin */
					y=yy;
					x=xx;
					z=zz;
					/* and don't calculate new scattering */
					do_scatter_now = FALSE;
				}	
			}
		}
		
		if (leaving_domain) break;
		
		if (do_track_particles) track_log_append(x,z,y); /* log new position */
				
		/* SCATTERING */
        /* we've reached the scatterer */
        /* get new scattering angle: */
		/* (this is conditional because it is being skipped after */
		/*  total reflection) */
		if (do_scatter_now) {
			get_angle_after_scattering(current_angle, current_azimuth, &angle, &azimuth);
			cnt_collisions += 1;
		} /* skip if total reflection */
		
		min_y = min(min_y, y);
		max_R = max(max_R, (x-x_entrance)*(x-x_entrance)+(z-z_entrance)*(z-z_entrance));
	}
	
	/* confine output angle */
	while (current_angle >= PI)
		current_angle -= 2*PI;
	while (current_angle < -PI)
		current_angle += 2*PI;
		
	if (do_track_particles) track_log_append(xx,zz,yy); /* log exit position */

	min_y = min(min_y, yy);
	max_R = max(max_R, (xx-x_entrance)*(xx-x_entrance)+(zz-z_entrance)*(zz-z_entrance));
	max_R = sqrt(max_R);

		
	/* output */
	return_stats[0]=current_angle;
	return_stats[1]=current_azimuth;
	return_stats[2]=xx;
	return_stats[3]=zz;
	return_stats[4]=yy;
	return_stats[5]=travel_distance;
	return_stats[6]=min_y;
	return_stats[7]=max_R;

	return_count[0]=exit_direction_is_up;
	return_count[1]=cnt_collisions;
	return_count[2]=cnt_refractions;
	return_count[3]=cnt_total_reflections;
	return_count[4]=cnt_missed_collisions;
	
	return 0;
}


#define QUOTE(s) # s   /* turn s into string "s" */

#define NDIM_CHECK(arr, expected_ndim) \
   if (PyArray_NDIM(arr) != expected_ndim) { \
   PyErr_Format(PyExc_ValueError, \
   "%s array is %d-dimensional, but expected to be %d-dimensional", \
   QUOTE(arr), PyArray_NDIM(arr), expected_ndim); \
   goto fail; \
   }

#define NDIM_CHECK_fail2(arr, expected_ndim) \
   if (PyArray_NDIM(arr) != expected_ndim) { \
   PyErr_Format(PyExc_ValueError, \
   "%s array is %d-dimensional, but expected to be %d-dimensional", \
   QUOTE(arr), PyArray_NDIM(arr), expected_ndim); \
   goto fail2; \
   }

#define NDIM_CHECK_fail_spec(arr, expected_ndim, fail_label) \
   if (PyArray_NDIM(arr) != expected_ndim) { \
   PyErr_Format(PyExc_ValueError, \
   "%s array is %d-dimensional, but expected to be %d-dimensional", \
   QUOTE(arr), PyArray_NDIM(arr), expected_ndim); \
   goto fail_label; \
   }
   
#define LEN_CHECK(arr, expected_len) \
   if (PyArray_DIMS(arr)[0] != (expected_len)) { \
   PyErr_Format(PyExc_ValueError, \
   "%s array is of length %lli, but expected to be of length %lli", \
   QUOTE(arr), (long long)PyArray_DIMS(arr)[0], (long long)(expected_len)); \
   goto fail; \
   }

#define DVAL(array,index) ( *((double*)PyArray_GETPTR1(array, index)) )
#define D2VAL(array,y,x) ( *((double*)PyArray_GETPTR2(array, y, x)) )

/* set dictionary value without increasing the reference count */
#define PYTHONDICT(out,key,pyfcn,value) {\
PyObject *tmp; \
tmp = pyfcn(value); \
PyDict_SetItem(out, key, tmp); \
Py_DECREF(tmp); }

#define PYTHONDICTstring(out,key,pyfcn,value) {\
PyObject *tmp; \
tmp = pyfcn(value); \
PyDict_SetItemString(out, key, tmp); \
Py_DECREF(tmp); }

/* create the following python objects only once */
PyObject *_pystr_angle;
PyObject *_pystr_azimuth;
PyObject *_pystr_prob; /* "k" */
PyObject *_pystr_depth; /* "thickness" */
PyObject *_pystr_n_surface;
PyObject *_pystr_n_bottom;
PyObject *_pystr_vertical_pos0; /* "z0" */
PyObject *_pystr_w0;
PyObject *_pystr_prob_anisotropy; /* "sigma_anisotropy" */
PyObject *_pystr_do_record_particle_track;
PyObject *_pystr_do_record_plane_crossing;
PyObject *_pystr_record_plane;

PyObject *_pystr_plane_crossings;
PyObject *_pystr_plane_crossings_count;
    
PyObject *_pystr_angle_in;
PyObject *_pystr_x;
PyObject *_pystr_y;
PyObject *_pystr_z;
PyObject *_pystr_path_length;
PyObject *_pystr_min_z;
PyObject *_pystr_max_R;
PyObject *_pystr_exit_direction;
PyObject *_pystr_collisions;
PyObject *_pystr_refractions;
PyObject *_pystr_total_reflections;
PyObject *_pystr_missed_collisions;

PyObject *_pystr_elevation_in;
PyObject *_pystr_azimuth_in;
PyObject *_pystr_polar_rot;
PyObject *_pystr_azimuth_rot;
PyObject *_pystr_elevation_out;
PyObject *_pystr_azimuth_out;

static PyObject * set_seed(PyObject *self, PyObject *args) {    /* new-190902 */
    unsigned long seed;
    if (!PyArg_ParseTuple(args, "k", &seed)) return NULL; // pass on error to Python
    init_random_number_generator( seed );
    return Py_BuildValue(""); // return None
}

static PyObject * simulate_one(PyObject *self, PyObject *args)
{
	/* this version is only 5 to 10% slower for many iterations */
	/* including overhead due to copying in and out of the dictionary */
	/* compared with a loop in C and storing stats in a NumPy array */

	
	/* track a single particle */
	PyObject *in =NULL, *out =NULL;
	
	/* O does not increase reference count */
	if (!PyArg_ParseTuple(args, "OO!", &in, &PyDict_Type, &out)) return NULL;
	
	if ( (PyDict_Check(in) == 0) || (PyDict_Check(out) == 0)) {
		PyErr_Format(PyExc_TypeError, \
			"Expect to receive two dictionaries");
		goto fail2;
	}
	
	int errors = 0;
	errors += 1 != PyDict_Contains(in, _pystr_angle );
	errors += 1 != PyDict_Contains(in, _pystr_azimuth );
	errors += 1 != PyDict_Contains(in, _pystr_prob );
	errors += 1 != PyDict_Contains(in, _pystr_depth );	
	if (errors > 0) {
		PyErr_Format(PyExc_ValueError, \
			"Input dictionary must contain items 'angle', 'azimuth', 'k', 'thickness'. It may also contain 'w0' (default: 1), 'n_surface' (default: 1), 'n_bottom' (default: 1), 'z0' (default: 0), 'sigma_anisotropy' (default: 0), 'do_record_particle_track' (default: False), and 'do_record_plane_crossings' (default: False) together with 'record_plane' (list or numpy array).");
		goto fail2;
	}
	
	if (lookup_length < 2) {
		PyErr_Format(PyExc_ValueError, 
			"Lookup table is of length %lli.", 
			lookup_length);
		goto fail2;
	}
	
	/* get input parameters from dictionary */	
	double angle = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_angle) );
	double azimuth = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_azimuth) );	
	double prob = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_prob) );
	double depth = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_depth) );
	    
    double n_surface = 1; /* optional as of 2 Jan 2019 */
    if (1 == PyDict_Contains(in, _pystr_n_surface )) {
		/* optional since 2 Jan 2019 */
		n_surface = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_n_surface) );
	}
    
    double n_bottom = 1; /* default setting NEW-130125 */
	if (1 == PyDict_Contains(in, _pystr_n_bottom )) {
		/* option NEW-130125 */
		n_bottom = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_n_bottom) );
	}

    double vertical_pos0 = 0; /* new 2 Jan 2019 */
    if (1 == PyDict_Contains(in, _pystr_vertical_pos0 )) {	
		vertical_pos0 = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_vertical_pos0) );
	}
    
    /* check that -depth <= vertical_pos0 <= 0 -- NEW 190902 */
	if ((vertical_pos0 < -depth) || (vertical_pos0 > 0)) {
		PyErr_Format(PyExc_ValueError, \
			"Initial position has to be %f=-thickness <= z0 <= 0, i.e. on either boundary (i.e. outside the medium) or in the domain (i.e. inside the medium), but z0=%f.",
            -depth, vertical_pos0);
		goto fail2;
    }
    
	double w0 = 1;
	if (1 == PyDict_Contains(in, _pystr_w0 )) {
		w0 = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_w0) );
	}
    
    if ((w0 == 0) && (prob > 0)) {
        PyErr_Format(PyExc_ValueError, \
            "Case of purely absorptive medium (w0=0, k>0) is currently not implemented. Use w0=0 with k=0 instead.");
        goto fail2;
    }
	
	double prob_anisotropy = 0;
	if (1 == PyDict_Contains(in, _pystr_prob_anisotropy )) {
		prob_anisotropy = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_prob_anisotropy) );
	}	
	
	if (1 == PyDict_Contains(in, _pystr_do_record_particle_track ) ) {	
		/* do as specified */
		do_track_particles = PyInt_AsLong( PyDict_GetItem(in, _pystr_do_record_particle_track) );
	} else {
		/* don't track unless specified explicitly */
		do_track_particles = FALSE;
	}
		
	if (1 == PyDict_Contains(in, _pystr_do_record_plane_crossing ) ) {


		do_record_planes = PyInt_AsLong( PyDict_GetItem(in, _pystr_do_record_plane_crossing) );
		record_index = 0; /* start (over)writing at the start */
		if (do_record_planes) {
			/* get plane crossing array */		
			PyObject *plane_array = PyDict_GetItem(in, _pystr_record_plane);
			PyArrayObject *py_record_plane = (PyArrayObject*) PyArray_FROM_OTF( plane_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
			NDIM_CHECK_fail2(py_record_plane, 1);
			/* and length */			
            RECORD_N_PLANES = PyArray_DIMS(py_record_plane)[0]; /* new as of 3 Jan 2019: use py_record_plane instead of plane_array */
			
			/* limit length to pre-allocated array */
			if (RECORD_N_PLANES > MAX_RECORD_N_PLANES) RECORD_N_PLANES = MAX_RECORD_N_PLANES;
			
			/* copy python array to double array */
			int i;
			for (i=0; i<RECORD_N_PLANES; i++) {
				record_plane[i] = DVAL(py_record_plane, i);
			}
			
			Py_DECREF(py_record_plane);
		} else {
			RECORD_N_PLANES = 0;
		}
	} else {
		do_record_planes = FALSE;
		RECORD_N_PLANES = 0;
	}
		
	
	double return_stats[8];
	long return_count[5];
	
	/* do all the work */
	track_particle(angle, azimuth, prob, depth, n_surface, n_bottom, w0, prob_anisotropy, vertical_pos0,
					(double*)&return_stats, (long*)&return_count); /* calling convention NEW-130125 */
	
	/* wrap return values */
	
	PYTHONDICT(out,_pystr_angle_in,PyFloat_FromDouble,angle)
	PYTHONDICT(out,_pystr_azimuth_in,PyFloat_FromDouble,azimuth)
	PYTHONDICT(out,_pystr_angle,PyFloat_FromDouble,return_stats[0])
	PYTHONDICT(out,_pystr_azimuth,PyFloat_FromDouble,return_stats[1])
	
	/* NB: x,y,z have different definition here than in most of this program */
	/*   in return values, z is vertical component */
	PYTHONDICT(out,_pystr_x,PyFloat_FromDouble,return_stats[2])
	PYTHONDICT(out,_pystr_y,PyFloat_FromDouble,return_stats[3])
	PYTHONDICT(out,_pystr_z,PyFloat_FromDouble,return_stats[4])
	PYTHONDICT(out,_pystr_path_length,PyFloat_FromDouble,return_stats[5])
	PYTHONDICT(out,_pystr_min_z,PyFloat_FromDouble,return_stats[6])
	PYTHONDICT(out,_pystr_max_R,PyFloat_FromDouble,return_stats[7])
	
	if (return_count[0] == TRUE)
		/* exit direction is up */
		PYTHONDICT(out,_pystr_exit_direction,PyInt_FromLong,(+1))
	else
		/* exit direction is down */		
		PYTHONDICT(out,_pystr_exit_direction,PyInt_FromLong,(-1))
		
	PYTHONDICT(out,_pystr_collisions,PyInt_FromLong,return_count[1])
	PYTHONDICT(out,_pystr_refractions,PyInt_FromLong,return_count[2])
	PYTHONDICT(out,_pystr_total_reflections,PyInt_FromLong,return_count[3])
	PYTHONDICT(out,_pystr_missed_collisions,PyInt_FromLong,return_count[4])

	if (do_record_planes) {
		/* add NumPy array to dictionary */
		
		npy_intp  shape[] = {record_index/6,6}; /* returns 2-dim array as of 3 Jan 2019 */
		PyObject *arr_obj = PyArray_SimpleNewFromData(2, (npy_intp*) &shape, 
												NPY_DOUBLE, record_plane_crossings);
		/* have Python never destroy the buffer */
		/* assuming Python uses malloc() and free() */
		/* ((PyArrayObject *)arr_obj)->flags |= NPY_OWNDATA; */
	
		PyDict_SetItem(out, _pystr_plane_crossings, arr_obj);		
		
		PYTHONDICT(out,_pystr_plane_crossings_count,PyInt_FromLong,(record_index/6)) /* corrected to divide by 6, 3 Jan 2019 */
	} else {
		/* indicate that any contents of the ['plane_crossings'] array is invalid */
		PYTHONDICT(out,_pystr_plane_crossings_count,PyInt_FromLong,(-1))
	}
	
	return Py_BuildValue("");

fail2:
	Py_XDECREF(in);
	Py_XDECREF(out);
	return NULL;
}

static PyObject * simulate_N_raw_out(PyObject *self, PyObject *args)
{	
	/* track a single particle */
	PyObject *in =NULL, *arg_out=NULL;
    PyArrayObject *out =NULL;
	
	long long iterations;
	/* O does not increase reference count */
	if (!PyArg_ParseTuple(args, "iOO!", &iterations, &in, &PyArray_Type, &arg_out)) return NULL;
	
	if ( (PyDict_Check(in) == 0) ) {
		PyErr_Format(PyExc_TypeError, \
			"Expect dictionary as second argument.");
		goto fail3;
	}
	out = (PyArrayObject*) PyArray_FROM_OTF(arg_out, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	long long dim_y = PyArray_DIMS(out)[0];
	long long dim_x = PyArray_DIMS(out)[1];
	
	long long errors = 0;
	errors += 1 != PyDict_Contains(in, _pystr_angle );
	errors += 1 != PyDict_Contains(in, _pystr_azimuth );
	errors += 1 != PyDict_Contains(in, _pystr_prob );
	errors += 1 != PyDict_Contains(in, _pystr_depth );	
	if (errors > 0) {
		PyErr_Format(PyExc_ValueError, \
			"Input dictionary must contain items 'angle', 'azimuth', 'k', 'thickness'.");
		goto fail3;
	}
		
	/* get input parameters from dictionary */	
	double angle = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_angle) );
	double azimuth = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_azimuth) );
	double prob = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_prob) );
	double depth = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_depth) );
    
    double n_surface = 1; /* optional as of 2 Jan 2019 */
    if (1 == PyDict_Contains(in, _pystr_n_surface )) {
		/* optional since 2 Jan 2019 */
		n_surface = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_n_surface) );
	}

    double n_bottom = 1; /* default setting NEW-130125 */
	if (1 == PyDict_Contains(in, _pystr_n_bottom )) {
		/* option NEW-130125 */
		n_bottom = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_n_bottom) );
	}
    
    double vertical_pos0 = 0; /* new 2 Jan 2019 */
    if (1 == PyDict_Contains(in, _pystr_vertical_pos0 )) {	
		vertical_pos0 = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_vertical_pos0) );
	}
    
    if ((vertical_pos0 < -depth) || (vertical_pos0 > 0)) {
        PyErr_Format(PyExc_ValueError, \
            "Initial position has to be %f=-thickness <= z0 <= 0, i.e. on either boundary (i.e. outside the medium) or in the domain (i.e. inside the medium), but z0=%f.",
            -depth, vertical_pos0);
        goto fail3;
    }
	
	double w0 = 1;
	if (1 == PyDict_Contains(in, _pystr_w0 )) {
		w0 = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_w0) );
	}

    if ((w0 == 0) && (prob > 0)) {
        PyErr_Format(PyExc_ValueError, \
            "Case of purely absorptive medium (w0=0, k>0) is currently not implemented. Use w0=0 with k=0 instead.");
        goto fail3;
    }

	double prob_anisotropy = 0;
	if (1 == PyDict_Contains(in, _pystr_prob_anisotropy )) {
		prob_anisotropy = PyFloat_AsDouble( PyDict_GetItem(in, _pystr_prob_anisotropy) );
	}
	
	#define N_stats 8
	#define N_count 5
	/* note: N_counts increased to 5 for cnt_missed_collisions on Sept 25, 2011 */
	double return_stats[N_stats];
	long long return_count[N_count];
	
	long long Nx = 2+N_stats + N_count;

	if ((dim_y != iterations) || (dim_x != Nx)) {
		PyErr_Format(PyExc_ValueError, \
		"Array is of size (%d,%d), but expected to be of size (%d,%d)", \
		(int)dim_y, (int)dim_x, (int)iterations, (int)Nx);
	}
	
	/* return raw dump of statistics */

	if (lookup_length >= 2) {
	
		long long run;
		for (run=0; run<iterations; run++) {
			/* do all the work */
			track_particle(angle, azimuth, prob, depth, n_surface, n_bottom, w0, prob_anisotropy, vertical_pos0,
							(double*)&return_stats, (long*)&return_count); /* calling convention NEW-130125 */
		
			D2VAL(out, run, 0) = angle;
			D2VAL(out, run, 1) = azimuth;
			long long i;
			for (i=0; i<N_stats; i++)
				D2VAL(out, run, i+2) = return_stats[i];
			for (i=0; i<N_count; i++)
				D2VAL(out, run, i+2+N_stats) = (double)return_count[i];
		}

	} else {
		PyErr_Format(PyExc_ValueError, 
			"Lookup table is of length %lli.", 
			lookup_length);
		goto fail3;
	}
	
	Py_XDECREF(out);
	
    return Py_BuildValue("");	

fail3:
	Py_XDECREF(in);	
	Py_XDECREF(out);
	return NULL;
}

static PyObject * define_phase_function(PyObject *self, PyObject *args)
{
	/* set global lookup table with pre-calculated cumulative distribution function */

	PyObject *arg_cum_p =NULL, *arg_Phi =NULL;    
    PyArrayObject *cum_p =NULL, *Phi =NULL;	
	
	if (!PyArg_ParseTuple(args, "OO", &arg_cum_p, &arg_Phi)) return NULL;

	cum_p = (PyArrayObject *) PyArray_FROM_OTF(arg_cum_p, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	Phi = (PyArrayObject *) PyArray_FROM_OTF(arg_Phi, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	
	NDIM_CHECK(cum_p, 1);
	NDIM_CHECK(Phi, 1);
	
    LEN_CHECK(Phi, PyArray_DIMS(cum_p)[0]);
	long long length = PyArray_DIMS(cum_p)[0];	
	
	if (lookup_cum_p == NULL)
		lookup_cum_p = (double*) malloc(sizeof(double) * length);
	else {
		double *ptr = (double*) realloc(lookup_cum_p, sizeof(double) * length);
		if (ptr == NULL) free(lookup_cum_p);
		lookup_cum_p = ptr;
	}
	
	if (lookup_Phi == NULL)
		lookup_Phi = (double*) malloc(sizeof(double) * length);
	else {
		double *ptr = (double*) realloc(lookup_Phi, sizeof(double) * length);
		if (ptr == NULL) free(lookup_Phi);
		lookup_Phi = ptr;
	}
		
	if ((lookup_cum_p == NULL) || (lookup_Phi == NULL)) {
		free(lookup_cum_p); lookup_cum_p = NULL;
		free(lookup_Phi); lookup_Phi = NULL;
		lookup_length = 0;
		PyErr_Format(PyExc_ValueError, \
			"Could not allocate arrays of length %lli.", \
			length);
		goto fail;
	}
	
	/* decode arrays */
	/* and check limits */
	double min_cum_p = 1e10;
	double max_cum_p = -1e10;
	
	long long idx;
	for (idx=0; idx < length; idx ++) {
		lookup_cum_p[idx] = DVAL(cum_p, idx);
		lookup_Phi[idx] = DVAL(Phi, idx);
		min_cum_p = min( min_cum_p, lookup_cum_p[idx] );
		max_cum_p = max( max_cum_p, lookup_cum_p[idx] );
	}
	lookup_length = length;
	
	/* do sanity check on cum_P */
	if (min_cum_p > 0.0) {
		PyErr_Format(PyExc_ValueError, 
			"CDF definition has to include 0.0 but starts at %f.", min_cum_p);
		
		free(lookup_cum_p); lookup_cum_p = NULL;
		free(lookup_Phi); lookup_Phi = NULL;
		lookup_length = 0;
		goto fail;
	}
	
	if (max_cum_p < 1.0) {
		PyErr_Format(PyExc_ValueError, 
			"CDF definition has to include 1.0 but ends at %f.", max_cum_p);
	
		free(lookup_cum_p); lookup_cum_p = NULL;
		free(lookup_Phi); lookup_Phi = NULL;
		lookup_length = 0;
		goto fail;
	}
	
	Py_DECREF(cum_p);
	Py_DECREF(Phi);
	
	Py_INCREF(Py_None);
	return Py_None;

fail:
	Py_XDECREF(cum_p);
	Py_XDECREF(Phi);
	return NULL;
}

static PyObject * return_particle_track(PyObject *self, PyObject *placeholder)
{        
    /* Returns a NumPy array containing the track */
    long long rows = track_log_index / 3;
    long long columns = 3;
    npy_intp extent[2] = {rows, columns};
    
    /* if (rows==0) return Py_BuildValue("");   */ /* no data --> return None */ /* new-190903: always return NumPy array, even if empty */
    
    /* create new 2D array */
    PyArrayObject *array = (PyArrayObject*) PyArray_SimpleNew(2, extent, NPY_DOUBLE);
    
    /* copy data from memory to Python array */
    long long row, column;
    for (row = 0; row < rows; row++) {
        for (column=0; column<columns; column++) {
            D2VAL(array, row, column) = track_log_buffer[row*columns+column];
        }
    }
                    
    return PyArray_Return(array);
}

static PyObject * python_get_random_angle(PyObject *self, PyObject *placeholder)
{
	return Py_BuildValue("f", get_random_angle() );
}


static PyObject * python_rotate_vector(PyObject* self, PyObject *args)
{
	PyObject *out =NULL;
	
	/* O does not increase reference count */
	if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &out)) return NULL;
	
	if ( (PyDict_Check(out) == 0)) {
		PyErr_Format(PyExc_TypeError, \
			"Expecting to receive dictionary");
		goto fail3;
	}
		
	/* in-vector in absolute reference frame */
	double angle_in = PyFloat_AsDouble( PyDict_GetItem(out, _pystr_elevation_in) );
	double azimuth_in = PyFloat_AsDouble( PyDict_GetItem(out, _pystr_azimuth_in) );
	
	/* relative to in-vector */
	double forward = PyFloat_AsDouble( PyDict_GetItem(out, _pystr_polar_rot) );
	double azimuth = PyFloat_AsDouble( PyDict_GetItem(out, _pystr_azimuth_rot) );
	
	double angle_out, azimuth_out;
	rotate_vector(angle_in,azimuth_in, forward, azimuth, &angle_out, &azimuth_out);

	/* out-vector in absolute reference frame */
	PYTHONDICT(out,_pystr_azimuth_out,PyFloat_FromDouble,azimuth_out)
	PYTHONDICT(out,_pystr_elevation_out,PyFloat_FromDouble,angle_out)

    return Py_BuildValue("ff", angle_out, azimuth_out); /* tuple */

fail3:
	Py_XDECREF(out);
	return NULL;
}

static PyObject * python_refract(PyObject *self, PyObject *args) {
        
    // get parameters from dictionary
    PyObject *io =NULL;
	
	/* O does not increase reference count */
	if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &io)) return NULL;
    double angle = PyFloat_AsDouble( PyDict_GetItemString(io, "angle") );
    double n = PyFloat_AsDouble( PyDict_GetItemString(io, "n") );
    int is_inside = (int)PyLong_AsLong( PyDict_GetItemString(io, "is_inside") );
    int total = 0; /* output flag, only */
    
    int ret = refract(&angle, n, is_inside, &total);
    // overwrite 'angle' (double) and 'total' (bool) entries in dictionary
    PYTHONDICTstring(io,"angle",PyFloat_FromDouble,angle)
    PYTHONDICTstring(io,"total",PyLong_FromLong,total)
    
    return Py_BuildValue("i", ret); /* always 0 */
}


static PyObject * python_clear(PyObject *self, PyObject *placeholder) {
    clear();
    return Py_BuildValue("");
}


/* The following Python2/Python3 module initialization is adapted from      */
/* Benjamin Peterson's Python HOWTO "Porting Extension Modules to Python 3" */
/* that is part of the Python Documentation.                                */

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyMethodDef Methods[] =
{
    /* user API */
    {"set_seed", set_seed, METH_VARARGS, "Set seed of random number generator: (seed)"},
	{"define_phase_function", define_phase_function, METH_VARARGS, "Defines phase function: (CDF_lookup, angle_lookup)"},
	{"simulate", simulate_one, METH_VARARGS, "Simulate scattering: (input_dictionary, output_dictionary)"},
	{"get_last_particle_track", return_particle_track, METH_NOARGS, "Return array with last particle track (if recorded)."},
    /* function that could be used operationally but may not be worth maintaining in the long run */
	{"_simulate_N_raw_out", simulate_N_raw_out, METH_VARARGS, "Simulate scattering: (count, input_dictionary, output_array)"},
    /* functions exported for code verification */
	{"_get_random_angle", python_get_random_angle, METH_NOARGS, "Returns a randomly generated scattering angle from CDF"},
	{"_rotate_vector", python_rotate_vector, METH_VARARGS, "Rotate vector as specified in dictionary (elevation_in, azimuth_in, polar_rot, azimuth_rot)."},
    {"_refract", python_refract, METH_VARARGS, "Calculate refraction at the interface"},
    /* functions exported for unit testing (i.e. reset global state) */
    {"_clear", python_clear, METH_NOARGS, "Free global memory and initialize random number generator"},
	{NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int gc_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int gc_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mcmodel",
        NULL, sizeof(struct module_state),
        Methods,
        NULL, gc_traverse, gc_clear, NULL
};

#define INITERROR return NULL
PyMODINIT_FUNC PyInit_mcmodel(void)

#else // Python 2:

#define INITERROR return
PyMODINIT_FUNC initmcmodel(void)

#endif
{
#if PY_MAJOR_VERSION >= 3
	PyObject *module = PyModule_Create(&moduledef);
#else // Python 2
	PyObject *module = Py_InitModule("mcmodel", Methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);
	
	st->error = PyErr_NewException("mcmodel.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

	import_array();
		
	/* do not use clock() in batch processes!!! */
	/* instead, use time() or something more sophisticated */
	uint32_t seed = (uint32_t)time(NULL);
	init_random_number_generator( seed );

    PyModule_AddStringConstant(module, "__version__", MODULE_VERSION_STRING);

    /* Global object creation to avoid repeated, temporary creation of string objects. These are never freed. */
	_pystr_angle = PyString_FromString( "angle" );
	_pystr_azimuth = PyString_FromString( "azimuth" );
	_pystr_prob = PyString_FromString( "k" );
	_pystr_depth = PyString_FromString( "thickness" );
	_pystr_n_surface = PyString_FromString( "n_surface" );
	_pystr_n_bottom = PyString_FromString( "n_bottom" );
    _pystr_vertical_pos0 = PyString_FromString( "z0" );
	_pystr_w0 = PyString_FromString( "w0" );
	_pystr_prob_anisotropy = PyString_FromString( "sigma_anisotropy" );
	_pystr_do_record_particle_track = PyString_FromString( "do_record_track" );
	_pystr_do_record_plane_crossing = PyString_FromString( "do_record_plane_crossings" );
    _pystr_record_plane = PyString_FromString( "record_planes" );
    
    _pystr_angle_in = PyString_FromString( "angle_in" );
    _pystr_x = PyString_FromString( "x" );
    _pystr_y = PyString_FromString( "y" );
    _pystr_z = PyString_FromString( "z" );
    _pystr_path_length = PyString_FromString( "path_length" );
    _pystr_min_z = PyString_FromString( "min_z" );
    _pystr_max_R = PyString_FromString( "max_R" );
    _pystr_exit_direction = PyString_FromString( "exit_direction" );
    _pystr_collisions = PyString_FromString( "n_collisions" );
    _pystr_refractions = PyString_FromString( "n_refractions" );
    _pystr_total_reflections = PyString_FromString( "n_total_reflections" );
    _pystr_missed_collisions = PyString_FromString( "n_missed_collisions" );
    _pystr_plane_crossings_count = PyString_FromString( "n_plane_crossings" );
    _pystr_plane_crossings = PyString_FromString( "plane_crossings" );
    

    _pystr_elevation_in = PyString_FromString( "elevation_in" );
    _pystr_azimuth_in = PyString_FromString( "azimuth_in" );
    _pystr_polar_rot = PyString_FromString( "polar_rot" );
    _pystr_azimuth_rot = PyString_FromString( "azimuth_rot" );
    _pystr_elevation_out = PyString_FromString( "elevation_out" );
    _pystr_azimuth_out = PyString_FromString( "azimuth_out" );
	
#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
