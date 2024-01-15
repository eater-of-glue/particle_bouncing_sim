# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:37:20 2024

@author: Roman
"""
import numpy as np

from scipy.sparse import tril as sparse_tril
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.integrate import solve_ivp


from scipy.interpolate import interp1d # (interpolate x vs y data)


import matplotlib.pyplot as plt
from matplotlib import cm

class State():
    kay = -1.0
    dist_power_term = -2.0
    _callcount_ode = 0
    
    def __init__(self,
                 bounding_wall,
                 particles):
        self.bw = bounding_wall
        self.pl = particles
        
        self._Setup_state_vec()
        
        self._Setup_b_id() # ? (key find slots)
        self._Setup_system_variables()
        self._Setup_forces_and_distances()
        #self._Compute_distances()
        #self._Compute_forces() # ? do here or ext?
        
        return
        
    def _Setup_state_vec(self):
        # 1. count particles & total dof
        # 2. calc len u, len dudt
        n = sum(self.pl.eq_count)
        self.__n_entries = n
        """ 
        for insertion, if all are 4 => can use `x%4`
        otherwise: entry_idx = sum( pvec.eq_count[ :x ])
        
        in fact can do this in the setup and therefore know spacing
        and index immediately for updating _u and _dudt
        
        but here is simpler as all counts are `4`
        """
        # 3. pre-allocate
        self._u = np.zeros(n)
        self._dudt = np.zeros(n) # this is RHS
        return
    #end `_Setup_state_vec()`
    
    def _Setup_b_id(self):
        idx = np.arange(0, self.__n_entries)
        
        self._b_u_x  = ( (idx%4) == 0 )
        self._b_u_y  = ( (idx%4) == 1 )
        self._b_u_xv = ( (idx%4) == 2 )
        self._b_u_yv = ( (idx%4) == 3 )
        # example:
        # idx :   [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]
        # idx%4 : [ 0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2,  3]
        # i.e., we reserve every [0,4,8,...] for x positions
        #                  every [1,5,9,...] for y positions, etc.
        
        # This effectively will act as our key for quickly updating _dudt
        # _u is also required as it passes into the ODE equation function
        return
    
    def _Setup_system_variables(self):
        self._u[self._b_u_x]  = self.pl.xi
        self._u[self._b_u_y]  = self.pl.yi
        
        self._u[self._b_u_xv] = self.pl.xvi
        self._dudt[self._b_u_x] = self.pl.xvi # add missing
        self._u[self._b_u_yv] = self.pl.yvi
        self._dudt[self._b_u_y] = self.pl.yvi # add missing
        
        
        return
    
    def _Setup_forces_and_distances(self):
        # build triangular lookup? sign table / sign equation
        
        # use nCr to find number of unique combinations
        # throw error if very high; * by 2 because x and y!
        n = len(self.pl.id_number)
        self._n_sparse = n
        
        #end if
        
        # just use for loops here as one off
        # afterwards index normally
        # sparse structure to 
        # req dx, dy -> distance or force => 3
        # but then sum down force
        #vec_temp = np.ones(n, dtype = self._u.dtype)
        
        # my guess: just take the L and create
        # (act: can create half array in np, then port?)
        
        
        # be cheap: use lists // wait no: need to be able to easily access rows
        # and col's
        
        # use row major order
        # https://en.wikipedia.org/wiki/Row-_and_column-major_order
        nr = n + 0 # include zero diagonal -> makes indexing easier to follow
        # i.e., A[i][j] corresponds to ith body on jth ...
        # whereas otherwise need to convert and gets confusing
        self._dx =              lil_matrix( (nr , nr) )#[ None ] * (n - 1)
        self._dy =              lil_matrix( (nr , nr) )#[ None ] * (n - 1)
        self._dist_terms =      lil_matrix( (nr , nr) )#[ None ] * (n - 1)
        self._force_terms =     lil_matrix( (nr , nr) )#[ None ] * (n - 1)
        self._charge_products = lil_matrix( (nr , nr) )
        self._nx =              lil_matrix( (nr , nr) )#[ None ] * (n - 1)
        self._ny =              lil_matrix( (nr , nr) )#[ None ] * (n - 1)
        
        self._fmag_x = np.zeros(nr)
        self._fmag_y = np.zeros(nr)
        
        for ix in range(0, self._n_sparse):
            self._charge_products[ix, ix+1:] = self.pl.q[ix] * self.pl.q[ix+1:]
        #end for
        
        """ Shape is
        0 1 2 3 ... n-1
        ------------------
        1   
        2 2
        3 3 3
        : : : :.
        N N N ......N
        
        i.e. dx[0] = array of 1:N indices
             dx[2] = array of 3:N indices
             this is a bootleg sparse structure
        """
        #for ix in range(0, n - 1):
        #    self._dx[ix] = np.zeros(n-ix - 1)
        #    self._dy[ix] = np.zeros(n-ix - 1)
        #    self._dist_terms[ix] = np.zeros(n-ix - 1)
        #    self._force_terms[ix] = np.zeros(n-ix - 1)
        #  
        return
    
    def _Compute_distances(self):
        # update dx
        for ix in range(0, self._n_sparse):
            # start 0, cols after 0 = xi[0] - xi[after 0]
            #    _dx[0, 0:]
            
            #" here could replace pl.xi with ._u[self._b_u_x][ix]
# =============================================================================
#             self._dx[ix, ix+1: ] = self.pl.xi[ix] - self.pl.xi[ix + 1:]
#             self._dy[ix, ix+1: ] = self.pl.yi[ix] - self.pl.yi[ix + 1:]
#             self._charge_products[ix, ix+1:] = self.pl.q[ix] * self.pl.q[ix+1:]
# =============================================================================
            self._dx[ix, ix+1: ] = self._u[self._b_u_x][ix] \
                                 - self._u[self._b_u_x][ix + 1:];
            self._dy[ix, ix+1: ] = self._u[self._b_u_y][ix] \
                                 - self._u[self._b_u_y][ix + 1:];
            # self charges do not update as not funcs of time (...yet!)
            
            # get distances
            #self._dist_terms[ix] = np.sqrt(self._dx[ix]**2 + self._dy[ix]**2)
            
            #self._force_terms[ix]= self.pl.q[ix] * self.pl.q[ix+1:] * (self._dist_terms[ix]**(-2.0))
        """ missing 
        nx vector and ny vector...
        
        `_dist_terms` should be renamed to `_dist_terms_squared` or something
        like that, as it is actually "s^2" not "s" : """
        
        #self._dist_terms = self._dx.power(2) + self._dy.power(2)
        self._dist_terms = self._dx.multiply( self._dx ) + self._dy.multiply( self._dy)
        #self._dist_terms = self._dist_terms.power(0.5)
        #self._dist_terms = self._dist_terms.power(0.5)
        
        #self._nx = self._dx.divide( self._dist_terms.power(0.5) )
        #self._ny = self._dy.divide( self._dist_terms.power(0.5) )
        #update
        temp = self._dist_terms.power(self.dist_power_term*0.5 - 0.5)
        self._nx = self._dx.multiply(temp)
        self._ny = self._dy.multiply(temp)
        return
    
    def _Compute_forces(self):
        # use 1 matrix and read backwards for -ve values (i.e. B->A)
        
        
        #q1q2_div_r2 = self.kay * self._charge_products.multiply( self._dist_terms )
        
        #self.force_terms = np.ravel(q1q2_div_r2.sum(axis=1) - q1q2_div_r2.sum(axis=0))
        
        #intermediary_a:
        self._force_terms = self.kay \
                       * self._charge_products#.multiply( \
        #                 self._dist_terms.power(self.dist_power_term) )
        
        #this bit is wrong
        # update nx := self._dx / dist =>  n-0.5 power (-0.5 because sqrt)
        # dx / (dist^2)^(0.5) * dist^(N)
        intermediary_x = self._force_terms.multiply(self._nx)
        intermediary_y = self._force_terms.multiply(self._ny)
        
        # for on particle i from sum of j's:
        # sum in `j` for A_ij, minus sum in `i` for A_ij,
        # as A_ij = -A_ij, this gets the math to be equivalent
        # i.e., doing A_ij - transpose(A_ij), then summing for particle `i`
        # (comment written after, please refer to `axis` values; they take
        # precidence here! Not the comment)
        self._fmag_x = np.ravel(intermediary_x.sum(axis=1)) - np.ravel(intermediary_x.sum(axis=0))
        self._fmag_y = np.ravel(intermediary_y.sum(axis=1)) - np.ravel(intermediary_y.sum(axis=0))
        return
    
    def _Compute_wall_forces(self):
        #Wrapper function to call method from bounding wall
        # CALL AFTER `_Compute_forces()`
        #dudt[6:9] += bounding_wall.Force_of_wall_arrayed_generalised(u[0:3],
        #                                    bounding_wall.wall_x0_reference,
        #                               bounding_wall.sign_modifier_lookup_x)
        #
        #dudt[9: ] += bounding_wall.Force_of_wall_arrayed_generalised(u[3:6],
        #                                    bounding_wall.wall_y0_reference,
        #                               bounding_wall.sign_modifier_lookup_y)
# =============================================================================
#         self._dudt[self._b_u_xv] += self.bw.Force_of_wall_arrayed_generalised(
#                                 self._u[self._b_u_x],
#                                 self.bw.wall_x0_reference,
#                                 self.bw.sign_modifier_lookup_x,
#                                 ) / self.pl.m
#         self._dudt[self._b_u_yv] += self.bw.Force_of_wall_arrayed_generalised(
#                                 self._u[self._b_u_y],
#                                 self.bw.wall_y0_reference,
#                                 self.bw.sign_modifier_lookup_y,
#                                 ) / self.pl.m
# =============================================================================
        
        self._fmag_x += self.bw.Force_of_wall_arrayed_generalised(
                                self._u[self._b_u_x],
                                self.bw.wall_x0_reference,
                                self.bw.sign_modifier_lookup_x,
                                )
        self._fmag_y += self.bw.Force_of_wall_arrayed_generalised(
                                self._u[self._b_u_y],
                                self.bw.wall_y0_reference,
                                self.bw.sign_modifier_lookup_y,
                                )
        # Apply element masses!
        return
    
    def _Ode_eq(self, t, u):
# =============================================================================
#         self.pl.xi = u[self._b_u_x] = u[self._b_u_x]
#         self.pl.yi = u[self._b_u_y]
#         
#         self.pl.xvi = u[self._b_u_xv]
#         self.pl.yvi = u[self._b_u_yv]
# =============================================================================
        #could move 4 lines below to `_Update_system_variables()` function
        self._u[self._b_u_x]  = u[self._b_u_x]
        self._u[self._b_u_y]  = u[self._b_u_y]
        self._u[self._b_u_xv] = u[self._b_u_xv]
        self._u[self._b_u_yv] = u[self._b_u_yv]
        
        # prevent u updates here // outdated, no need!
        #self._Setup_system_variables()
        # update distance and THEN force calcs
        self._Compute_distances()
        self._Compute_forces()
        self._Compute_wall_forces()
        
        # UPDATE POSITIONS STOOPID!!
        self._dudt[self._b_u_x] = u[self._b_u_xv]
        self._dudt[self._b_u_y] = u[self._b_u_yv]
        
        # update dudt side with forces
        self._dudt[self._b_u_xv] = self._fmag_x / self.pl.m
        self._dudt[self._b_u_yv] = self._fmag_y / self.pl.m
        #print(t)
        # wall forces tba
        
        # wall force end
        
        # div by masses
        ###self._dudt[self._b_u_xv] /= self.pl.m
        self._callcount_ode += 1
        return self._dudt
    
    def _Hurp_durp(self):
        # placeholder when originally started writing
        return
    #end `_Hurp_durp()`
    
#end classdef

class Particle():
    """Particle
     No 'call as', this class is self explanatory. This object
     models a single particles initial state and persistent
     properties such as mass.
     
     The identifier number is assigned by `Particle_Vec`, but
     I do not believe it ended up being used
    """
    def __init__(self,
                 x0 = 1.0,
                 y0 = 1.0,
                 xv0 = 0,
                 yv0 = 0.0,
                 mass = 1.0,
                 charge = 1.0,
                 id_number = None):
        # initial state
        self.x0 = x0
        self.y0 = y0
        self.xv0 = xv0
        self.yv0 = yv0
        
        # properties (here assume constant)
        self.m = mass
        self.q = charge
        
        # bean counter
        self.id_number = id_number
        
        # future proofingz
        self.eq_count = int(4)
        # (must be integer)
        
        # active coordinates
        self.xi = x0
        self.yi = y0
        self.xvi = xv0
        self.yvi = yv0
        return
#end classdef `Particle`


class Particle_Vec():
    """Vector of Particle type
    Provides methods to hide for looping lists
    (call as:)
        particle_vec = Particle_Vec(particle_list)
    
    
    Note some methods are not used as it made more sense
    to put it in the `State` object
    
    Effectively this just acts as something to transport
    the particles properties and initial conditions into
    the state object.
    
    This object also converts particle object properties
    into numpy arrays (otherwise, I forgot what this code
    looks like)
    """
    # could alternatively store vector directly
    # in particle and index that way?
    
    #speaking of can vectorise distance calc completely
    # in numpy
    # e.g. dx = vec.x[b_target]**2 - vec.x[b_others]**2
    "ok use this as actual working layer"
    " we lie to user when Particle is shown"
    " i.e., use once then migrate here"
    def __init__(self, particles_list):
        # make this cheaply immutable
        # -> store as private
        # -> use getter for x0;
        #actually dont bother, add this later
        self._x0 = np.array([t.x0 for t in particles_list])
        self._y0 = np.array([t.y0 for t in particles_list])
        self._xv0 = np.array([t.xv0 for t in particles_list])
        self._yv0 = np.array([t.yv0 for t in particles_list])
        
        # properties (here assume constant)
        self.m = np.array([t.m for t in particles_list])
        self.q = np.array([t.q for t in particles_list])
        
        # bean counter
        #self.id_number = np.array([t.id_number for t in particles_list])
        self.id_number = np.arange(0, len(particles_list))
        self.eq_count =  np.array([t.eq_count for t in particles_list])
        # use np.repeat([1,2],3)
        # for repeating patter 1 1 1 2 2 2 etc
        
        
        # active coordinates
        self.xi = self._x0
        self.yi = self._y0
        self.xvi = self._xv0
        self.yvi = self._yv0
        
        self._Post_init()
        return
    #end
    
    def _Post_init(self):
        # perform post initialiser tasks
        
        #1. set q==0 flags using np logical arrays
        self._q_flags = (self.q == 0)
        # if flag is true => ignore it from all particle force calculations
        return
        
    def _Compute_distances(self):
        b_use = np.logical_not( self._q_flags )
        # ^ equivalent to `~self._q_flags`
        
        x = self.xi[b_use]
        y = self.yi[b_use]
        #unique combos = nCr-> n! / k!(n-k)!
        # n = len(x); k = 2 (2 choices)
# end | `Particle_Vec` class

class Bounding_Wall():
    """
    Bounding_Wall
    (call as:)
        bw = Bounding_Wall( {named args here} )
    
    Boundary Wall object for particle simulation. Has methods for
    computing wall forces exerted on particles when particles collide
    with the wall. At the present, this is modelled as an elastic
    hookean (hooks law) spring force, in the form
    
                F := stiffness * delta_x
                
    where delta_x := x_particle - x_wall
    (note: x_wall is a lookup function based on the particles position)
    
    This Class provides no dynamic ability to assign custom wall forces
    however, this should be easy to modify.
    """
    #" Here are properties shared across any creation of boundary walls"
    #" note there should only be one Bounding_Wall 'object' created"
    # Wall boundary coordinates (moved to `__init__`)
    #x_left  = -2.0*0.5
    #x_right =  2.0*0.5
    #y_top   =  1.0*1
    #y_bot   = -1.0*1
    
    # Wall spring stiffness coefficient, currently in form of `F = k * delta_x`
    #stiffness = 1e3
    # float
    EPS = 1e-15
    
    def __init__(self, 
                 x_left  =  -2.0 * 0.5,
                 x_right =   2.0 * 0.5,
                 y_top   =   1.0 * 1,
                 y_bot   =  -1.0 * 1,
                 stiffness = 1.0e4):
        """ Initialiser - creates the object
        Similar to writing "Float(2.314)" which creates a float number.
        
        Here it creates the bounding wall, after which we can use it's
        collection of functions during the simulation.
        
        I could have just wrote it as functions, but originally started
        this way and I never bothered to re-write it
        """
        # quick fix 20240115 2049 enable boundary modifications via
        # initialiser / move from common attribute to init attr. :
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bot = y_bot
        
        self.stiffness = stiffness
        # generate force lookup table
        # possibly change to (0,1) output format as sign may be absorbed by
        # delta x/y value in `Force_of_wall`
        #print(self.x_left)
        
        # Formally was EPS = 1e-15, here forcing it to be same value
        # as stated in properties listed (edit: this can be replaced
        # with a numpy function to perform relative float point prec.
        # checks)
        EPS = self.EPS
        
        # use nearest neighbour interpolation to get correction sign for
        # direction in `F = sign_mod * k * delta_x`
        # (note: this wasn't planned out well, so there is room for
        #        improvement)
        self.sign_modifier_lookup_x = interp1d([self.x_left - EPS, self.x_left,
                                                self.x_right, self.x_right + EPS],
                                               [1, 0, 
                                                 0, -1],
                                               kind="nearest",
                                               bounds_error = False,
                                               fill_value=(-1,-1),
                                               )
        
        self.sign_modifier_lookup_y = interp1d([self.y_bot - EPS, self.y_bot,
                                                self.y_top, self.y_top + EPS],
                                               [1, 0, 
                                                 0, -1],
                                               kind="nearest",
                                               bounds_error = False,
                                               fill_value=(-1,-1),
                                               )
        
        # create interpolation for wall position reference in `delta_x`
        # i.e.,   `delta_x = x_particle - x_wall_reference`
        # or equivalently `= x_particle - x0_reference`
        self.wall_x0_reference = interp1d([self.x_left, self.x_right],
                                          [self.x_left, self.x_right],
                                          kind="nearest",
                                          bounds_error = False,
                                          fill_value=(self.x_left,self.x_right),
                                          )
        
        self.wall_y0_reference = interp1d([self.y_bot, self.y_top],
                                          [self.y_bot, self.y_top],
                                          kind="nearest",
                                          bounds_error = False,
                                          fill_value=(self.y_bot, self.y_top),
                                           )
    #end `__init__`
    
    def Force_of_wall(self, particle):
        delta_x = particle.x - self.wall_x0_reference(particle.x)
        delta_y = particle.y - self.wall_y0_reference(particle.y)
        
        # if delta_x values get too small, just force zero them
        # (this actually might not be needed)
        if abs(delta_x) < self.EPS:
            delta_x = 0;
        if abs(delta_y) < self.EPS:
            delta_y = 0;
        
        force_wall_x = self.stiffness \
                     * self.sign_modifier_lookup_x(particle.x) \
                     * delta_x
        
        force_wall_y = self.stiffness \
                     * self.sign_modifier_lookup_y(particle.y) \
                     * delta_y
        return force_wall_x, force_wall_y
    
    
    # below is outdated: see {A#1} comment
    def Force_of_wall_arrayed_x(self, x_array):
        # This function enables passing a list (array) of x_particle
        # positions and computing all their forces at once
        # e.g., [xp1, xp2, xp3, ...]
        # => delta_x = [xp1, xp2, xp3, ...] - [xref1, xref2, xref3, ...]
        # hence
        # force_wall_x = k * [sign1, sign2, ...] * [dx1, dx2, ...]
        #              = [k*s1*dx1, k*s2*dx2, k*s3*dx3, ...]
        delta_x = x_array - self.wall_x0_reference(x_array)
        #delta_y = particle.y - self.wall_y0_reference(particle.y)
        
        force_wall_x = self.stiffness \
                     * self.sign_modifier_lookup_x(x_array) \
                     * delta_x
        
        
        return force_wall_x
    
    def Force_of_wall_arrayed_y(self, x_array):
        # same as above, just wrote hap hazardly, you could merge
        # both into the same function that has the wall_reference
        # function as an input - in fact that would be better!
        
        delta_x = x_array - self.wall_y0_reference(x_array)
        #delta_y = particle.y - self.wall_y0_reference(particle.y)
        
        force_wall_x = self.stiffness \
                     * self.sign_modifier_lookup_y(x_array) \
                     * delta_x
        
        
        return force_wall_x
    
    # {A#1} comment: here is the "better version" of force of wall arrayed
    # it is better as it uses "do not repeat yourself" - i.e., both previous
    # functions were identical except for the reference force
    # cntr+f for `new way: {A#1-A}` to see fix
    def Force_of_wall_arrayed_generalised(self, u_array, 
                                                wall_reference_fnc,
                                                sign_modifier_lookup_fnc):
        # same as above, just wrote hap hazardly, you could merge
        # both into the same function that has the wall_reference
        # function as an input - in fact that would be better!
        
        delta_u = u_array - wall_reference_fnc(u_array)
        #delta_y = particle.y - self.wall_y0_reference(particle.y)
        
        force_wall_u = self.stiffness \
                     * sign_modifier_lookup_fnc(u_array) \
                     * delta_u
        
        
        return force_wall_u
# =============================================================================
# def Estimate_RAM(Q = 6, N = 4, d_bits = 64):
#     return 
# =============================================================================

#plist = [Particle(), Particle(x0=-1, y0=-2), Particle(x0=2,y0=2), Particle(x0=-2,y0=-3)]


# =============================================================================
# plist = [Particle(x0 = -1.0, y0 = 0.0, xv0 = 0.25, yv0 = 0.25), 
#          Particle(x0 = 1.0, y0 = 0),
#          Particle(x0 = 0.0, y0 = -0.10, mass = 2), 
#          Particle(x0= 0.0, y0=1, yv0 = 0.5),
#          Particle(x0=  0.5, y0 = .5, yv0 = +0.1),
#          Particle(x0= -0.5, y0 = -.5, yv0 = -0.1),]
# 
# plist = [Particle(x0 = -1.0, y0 = 0.0, xv0 = 1.25, yv0 = 1.25*0), 
#          Particle(x0 = 1.0, y0 = 0.0, xv0 = -1.25, yv0 = 0 ),
#          Particle(x0 = 0.0, y0 = -1, xv0 = -1.25*0, yv0 = +1.25), 
#          Particle(x0= 0.0, y0=1.0, xv0 = 1.25*0, yv0 = -1.25)]
# =============================================================================


# line after overrides this one:, please run small case first
plist = [Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = np.random.rand(), yv0 = np.random.rand()),
         ]
         #Particle(x0 = 0, y0 = 10, xv0 = 0, yv0 = 0, mass = 1e3, charge =1e4),]

# =============================================================================
# =============================================================================
plist = [Particle(x0 = -.1, y0 = 0, xv0 = 0, yv0 = 0.0, charge = 4),
          Particle(x0 = .1, y0 = 0, xv0 = 0, yv0 = 0.0, charge = 4),
          Particle(x0 = 0, y0 = .1, xv0 = -0.0, yv0 = 0., charge = 4),
          Particle(x0 = 0, y0 = -.1, xv0 = 0.2, yv0 = 0.0, charge = 8),
          Particle(x0 = -.1, y0 = -.1, xv0 = 0.2, yv0 = 0.0, charge = 4)]
# =============================================================================
# =============================================================================
# =============================================================================
# plist = [Particle(x0 = -0.1, y0 = 0, xv0 = 1.0, yv0 = 3, charge = 2),
#          Particle(x0 = 0.1, y0 = 0, xv0 = -1.0, yv0 = 3, charge = 2),]
# =============================================================================
# =============================================================================
# plist = [Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = 0.04, yv0 = 0),
#          Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = 0.04, yv0 = 0),
#          Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = 0.04, yv0 = 0),
#          Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = 0.04, yv0 = 0),
#          Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = 0.04, yv0 = 0),
#          Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = 0.04, yv0 = 0),
#          Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = 0.04, yv0 = 0),
#          Particle(x0 = np.random.rand(), y0 = np.random.rand(), xv0 = 0.04, yv0 = 0),
#          Particle(x0 = 0, y0 = 10, xv0 = 0, yv0 = 0, mass = 1e3, charge =1e4),]
# =============================================================================
n = len(plist)

combos = int( 2 * np.math.factorial( ( n ) ) \
       / (2.0 * np.math.factorial( n - 2 )) )
       
if (n > 40):
    raise ValueError(f'Number of particles has lead to combinations > 144000; your combos = {combos}')


pvec = Particle_Vec(plist)

bw = Bounding_Wall()
bw.stiffness = 1e4

st = State(bw, pvec)
st.kay = +1.0 # -1 for attraction; +1 for electro-coulumbic force

print(f"{st._dx}\n{st._dy}\n{st._dist_terms}\n\n")
print("\n")
#st._Compute_distances()
print(f"{st._dx}\n\n{st._dy}\n\n\n{st._dist_terms}")

# expected_force = -1.0 * 1 * 1 * (np.sqrt(2))

t_span = [0.0, 5.0]
print(f"Initial _u = {st._u}")
print(" (Solver start)")
sol = solve_ivp(st._Ode_eq,
                t_span,
                st._u,
                method = 'RK45',
                rtol = 1e-3,
                atol = 1e-3,
                max_step = 1,
                dense_output = True,
                )
#max_step = ,
print(f"Post solver _u = {st._u}")
print(" ( Solver complete ) ")
fig, axs = plt.subplots(1)
sz = len(plist)
idx = np.arange(0, len(st._u))
idx_s = idx[st._b_u_x]
jdx_s = idx[st._b_u_y]

ENABLE_ANI = True
if not(ENABLE_ANI):
    for ix in range(len(idx_s)):
        axs.plot(sol.y[idx_s[ix],:], sol.y[jdx_s[ix],:], '.-' )

#axs.plot(sol.y[:,1], sol.y[:,1], '.')
#axs.plot(sol.t, sol.y[3,:], '.')

DURATION = 15 # milliseconds; here 5 seconds
# seconds (4 is a bit fast)
FPS_INPUT = 60 # fps
FRAMES = DURATION * FPS_INPUT

if ENABLE_ANI:
    t_lin = np.linspace(sol.t[0], sol.t[-1], FRAMES)
    sol_lin = sol.sol(t_lin)
    


def animate(i):
    axs.clear()
    #points_xy = sol.y[[0,3,1,4,2,5],:]
    
    for ix in range(len(idx_s)):
        #axs.plot(sol_lin.y[idx_s[ix],i], sol_lin.y[jdx_s[ix],i], '.-', markersize = 12)
        axs.plot(sol_lin[idx_s[ix],i], sol_lin[jdx_s[ix],i], '.-', markersize = 12)
    
# =============================================================================
#     axs.plot(sol_lin[0,i],
#              sol_lin[3,i],
#              '.-',
#              markersize=12,
#              )
#     axs.plot(sol_lin[1,i],
#              sol_lin[4,i],
#              '.-',
#              markersize=12,
#              )
#     axs.plot(sol_lin[2,i],
#              sol_lin[5,i],
#              '.-',
#              markersize=12,
#              )
# =============================================================================
    axs.set_xlim(left = st.bw.x_left*1.8, right = st.bw.x_right*1.8)
    axs.set_ylim(bottom = st.bw.y_bot*1.8, top = st.bw.y_top*1.8)
# end
# =============================================================================
# 
# def animate2(i):
#     axs.clear()
#     #points_xy = sol.y[[0,3,1,4,2,5],:]
#     
#     for ix in range(len(idx_s)):
#         axs.plot(sol.y[idx_s[ix],i], sol.y[jdx_s[ix],i], '.-', markersize = 12)
#     
# # =============================================================================
# #     axs.plot(sol_lin[0,i],
# #              sol_lin[3,i],
# #              '.-',
# #              markersize=12,
# #              )
# #     axs.plot(sol_lin[1,i],
# #              sol_lin[4,i],
# #              '.-',
# #              markersize=12,
# #              )
# #     axs.plot(sol_lin[2,i],
# #              sol_lin[5,i],
# #              '.-',
# #              markersize=12,
# #              )
# # =============================================================================
#     axs.set_xlim(left = st.bw.x_left*1.2, right = st.bw.x_right*1.2)
#     axs.set_ylim(bottom = st.bw.y_bot*1.2, top = st.bw.y_top*1.2)
# # end
# =============================================================================

if ENABLE_ANI:
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import PillowWriter
    # interval is in millisec so `1000 * (1 / FPS)`
    ani = FuncAnimation(fig, animate, frames = len(t_lin), interval = 1000 / FPS_INPUT, repeat = False)
    writer = PillowWriter(fps=FPS_INPUT,
                        metadata=dict(artist='Me'),
                        bitrate=1800)
    #ani.save('scatter_19.gif', writer=writer)
    
    
# =============================================================================
# if ENABLE_ANI:
#     from matplotlib.animation import FuncAnimation
#     from matplotlib.animation import PillowWriter
#     # interval is in millisec so `1000 * (1 / FPS)`
#     ani = FuncAnimation(fig, animate2, frames = len(sol.t), interval = DURATION / len(sol.t), repeat = False)
#     writer = PillowWriter(fps=15,
#                         metadata=dict(artist='Me'),
#                         bitrate=1800)
#     ani.save('scatter.gif', writer=writer)
# =============================================================================
