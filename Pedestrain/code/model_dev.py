"""
Yadu Raj Bhageria
Prasun K. Ray
Imperial College London
Mathematics Department

Contains:
initialize_global_parameters
check_model_ready
print_model_parameters
reset_model
distance_to_wall
f_w
f_j
bypass_funcs
f
compute_d_h
compute_alpha_des
compute_bodycollision_acceleration
compute_destinations
move_pedestrians
update_model
advance_model
"""
#------------------------------------------------------------------------------#
import math
import numpy as np
import matplotlib.pyplot as plt
#To parallelize the code
import time
import scipy.spatial.distance as scidist
import ped_utils as putils
#------------------------------------------------------------------------------#
#To precompile certain functions in order to make them faster - unsure about speedup at present
# from numba import jit, double, int8
#------------------------------------------------------------------------------#
def initialize_global_parameters():
    """ To intialize global parameters that are dependent on initial conditions or default settings preffered """
    global variablesready
    if variablesready:
        #Makes sure the scopes of the variables are global to the module
        global alpha_0, x_full, gap, H, alpha_current, alpha_des, f_alpha_des, v_des, contact_p, contact_w, n_walls
        global r, mass, v, v_0, v_full

        #angle to destination
        alpha_0 = np.arctan2((o[1]-x[1]),(o[0]-x[0]))
        #Initalize the array that stores movement values over time
        x_full = np.copy(x)
        gap = np.zeros((n,n))
        #Field of Vision for each of the pedestrians
        #H = np.random.uniform(H_min,H_max,n)
        H = H_min*np.ones(n)
        #set initial alpha_direction to alpha_0
        alpha_current = np.copy(alpha_0)
        alpha_des = np.zeros(n)
        f_alpha_des = np.zeros(n)
        #Array to store v_des
        v_des = np.zeros(n)
        #Store information about persons in contact with people and walls

        if n_walls is None:
            n_walls = 0
        contact_p = np.zeros((n,n))
        contact_w = np.zeros((n,n_walls))

        if np.shape(mass) != (n,):
            mass = np.random.uniform(60,100,n)
        #Radius, r = mass/320
        r = mass/320

        #If starting starting velocities are not specified then its assumed that they are zero for all people
        if np.shape(v) != (2,n):
            v = np.zeros((2,n))
        v_full = np.copy(v)

        if np.shape(v_0) != (n,):
            v_0 = 1.3*np.ones(n)

        #For clarity
        variablesinitialized = True
        if instructions: print ("%d cores in use" %(num_cores))
    else:
        if instructions: print ("Not all required variables initialized and checked. To not avoid checking manually configure variablesready to True")
#------------------------------------------------------------------------------#
def check_model_ready():
    """ Make sure all neccessary parameters for the model are initalized properly and allows user to call initialize_global_parameters() """

    global variablesready

    if n is None:
        if instructions: print("value of n not given")
    else:
        variablesready = True

        if x is None or np.shape(x) != (2,n):
            if instructions: print("position values array, x, not initalized or not in the right shape (2xn)")
            variablesready = False

        if o is None or np.shape(o) != (2,n):
            if instructions: print("destination values array, o, not initalized or not in the right shape (2xn)")
            variablesready = False

        if mass is None or np.shape(mass) != (n,):
            if instructions: print("mass array not initialized or not with correct shape (n). It will be initailized with default values when initalizing global parameters - randomly uniform values between 60 and 100")

        if v_0 is None or np.shape(v_0) != (n,):
            if instructions: print("comfortable walking speed array, v_0, not initialized or not with correct shape (n). It will be initailized with default values of 1.3m/s when initalizing global parameters")

        if v is None or np.shape(v) != (2,n):
            if instructions: print("initial velocity array, v, not initialized or not with correct shape (2xn). It will be initailized with default values of zeros when initalizing global parameters")

        if n_walls is None:
            if instructions: print("number of walls, n_walls, not initalized. It will be assumed to be 0 when initalizing global parameters")
        else:
            if walls is None or np.shape(walls) != (5,n_walls):
                if instructions: print("numbers of walls initalized but array to store information about the walls not initialized or not with correct shape (5xn)")
                variablesready = False

    if variablesready:
        if instructions: print("All necessary variables have been initalized. Call initialize_global_parameters() to initaize dependent parameters")
    else:
        if instructions: print("Model is not ready. Please initialize required parameters")
#------------------------------------------------------------------------------#
def print_model_parameters():
    print ("tau = %4.2f, angular resolution in degrees = %4.2f, d_max = %4.2f, k = %4.2e, t = %4.2f" %( tau, math.degrees(ar), d_max, k, t ) )
#------------------------------------------------------------------------------#
def reset_model():
    """ Resets all inital conditions and sets model parameters to their default values """

    global variablesready, tau, ar, d_max, k, t, H_min, H_max, instructions, n, x, o, mass, v_0, v, n_walls, walls, color_p

    #Parameters of the model
    variablesready = False
    tau = 0.5 #second heurostic constant
    ar = math.radians(0.1) #angular resolution
    d_max = 10. #Horizon distance
    k = 5e3 #body collision constant
    t = 0 #Initial time set to 0
    H_min = math.radians(75)
    H_max = math.radians(75)
    instructions = False
    time_step = 0.05

    #Neccessary variables that that need to be initalized properly
    n = None #integer
    x = None #array of size 2xn
    o = None #array of size 2xn
    #Optional - default values initialized if not done so manually in func above
    mass = None #array of size n
    v_0 = None #array of size n
    v = None #array of size 2xn
    n_walls = None #integer
    walls = None #array of size 5xn - a,b,c,startwal, endwal
    #Optional - Not initalized if not specified as it has limited use
    color_p = None

#------------------------------------------------------------------------------#
def distance_to_wall(i,w):
    """ Computes distance from person i to wall w """

    #For clarity: person, i, values
    xi = x[0][i]
    yi = x[1][i]
    ri = r[i]

    #For clarity: wall, w, values
    wall = walls[:,w]
    a = wall[0]
    b = wall[1]
    c = wall[2]
    wall_start = wall[3]
    wall_end = wall[4]

    #Extract start and end points, (x,y), of wall
    if b != 0:
        x1 = wall_start
        x2 = wall_end
        y1 = - (a * x1 + c) / b
        y2 = - (a * x2 + c) / b
    else:
        x1 = x2 = - c / a
        y1 = wall_start
        y2 = wall_end

    #Compute distance from person i to wall
    tx = x2-x1
    ty = y2-y1
    val =  ((xi - x1) * tx + (yi - y1) * ty) / (tx*tx + ty*ty)
    if val > 1:
        val = 1
    elif val < 0:
        val = 0
    x_val = x1 + val * tx
    y_val = y1 + val * ty
    dx = x_val - xi
    dy = y_val - yi
    dist = math.sqrt(dx*dx + dy*dy)

    return dist
#------------------------------------------------------------------------------#

#Not sure if this speedup in significant: @jit
def f_w( alpha, i, wall, xi, yi, v_0i, ri, d_max, v_xi, v_yi):
    """ Compute the distance to collision with walls"""

    #Extract values from the wall array for clarity
    a = wall[0]
    b = wall[1]
    c = wall[2]
    wall_start = wall[3] - ri
    wall_end = wall[4] + ri
    if np.abs(v_yi)>1e-14:
        m = v_xi/v_yi
    elif v_xi>0:
        m = np.inf
    else:
        m = -np.inf

    #Deal with case when velocity is zero
    d = a*v_xi + b*v_yi
    if d == 0:
        return d_max

    #Check if there is interception with the wall in direction alpha and if not return d_max
    #For horizontal walls
    if a == 0:
        y_wall = -c/b
        if v_yi == 0:
            return d_max
        elif v_yi>0:
            if yi>y_wall:
                return d_max
            else:
                delta_y=y_wall-(yi+ri)

            x_intercept = xi + delta_y * m
            if x_intercept < wall_start or x_intercept > wall_end:
                return d_max
        else:
            if yi<y_wall: #upper wall
                return d_max
            else:
                delta_y=(yi-ri)-y_wall

                x_intercept = xi - delta_y * m
                if x_intercept < wall_start or x_intercept > wall_end:
                    return d_max
    #For vertical walls
    if b == 0:
        if np.abs(v_xi)>1e-14:
            m = v_yi/v_xi
        elif v_yi>0:
            m = np.inf
        else:
            m = -np.inf

        x_wall = -c/a
        if v_xi == 0:
            return d_max
        elif v_xi > 0:
            if xi>x_wall:
                return d_max
            else:
                delta_x = x_wall-(xi+ri)
                y_intercept = yi + delta_x * m
                if y_intercept < wall_start or y_intercept > wall_end:
                    return d_max
        else:
            if xi<x_wall:
                return d_max
            else:
                delta_x = (xi-ri)-x_wall
                y_intercept = yi - delta_x * m
                if y_intercept < wall_start or y_intercept > wall_end:
                    return d_max
    #1. Need to have one for diagonal walls as well.
    #2. Can perhaps speedup process by utilizing above calculated values for returning f_alpha

    # if d < 0:
    #     #Unsure whether to just return d_max or convert wall intergers
    #     return d_max
        """ Need to check if I just return d_max or should I swap the negatives and positives
        a = -a
        b = -b
        c = -c
        d = -d
        """

    delta_tvals = np.array([None] * 2)
    #Solve for time to collision with wall
    #sqrt_d = math.sqrt( d )
    r_component = ri*math.sqrt(a**2 + b**2)
    abc_component = - a*xi - b*yi - c
    delta_tvals[0] = (   r_component + abc_component) / d #was sqrt(d) instead of d in original code
    delta_tvals[1] = ( - r_component + abc_component) / d

    #Pick the smallest positive delta_t value and return d_max for no positive values
    if max(delta_tvals) < 0:
        return d_max
    else:
        delta_t = np.min(delta_tvals[delta_tvals > 0])

    #Compute and return distance to collision
    dist = v_0i * delta_t
    f_alpha = min(dist,d_max)
    return f_alpha
#------------------------------------------------------------------------------#
def f_j(alpha,i,j,dx_ij,dy_ij,d_max,rsum_ij,v_0i,v_xj,v_yj,dv_x,dv_y, quad_A2_j,quad_B_ij,quad_C2_ij,quad_D_ij):
    """ Compute the distance to colision with person j"""

    #Deals with cases when quadratic does not need to solved
    #Contained in a separate function (below) to try improve speeds using jit but so far to no avail

    if gap[i,j]<rsum_ij:
        return 0
    elif gap[i,j]==rsum_ij:
        bypass = bypass_funcs(dx_ij,dy_ij,alpha,d_max)
        if bypass != -1:
            return bypass
    if v_0i == 0 and v_xj == 0 and v_yj == 0:
        return 0

    #Minimize floating point operations by using temp variables

    #Coefficients of the quadratic equation to be solved
    #quad_A = (dv_x)**2 + (dv_y)**2
    #quad_B = 2*(dv_x)*(dx_ij) + 2*(dv_y)*(dy_ij)
    #quad_C = (xdiff)**2 + (ydiff)**2 - (rsum_ij)**2
    #Solve the Quadratic for the smallest positive root
    delta_t = putils.smallest_positive_quadroot(quad_A2_j, quad_B_ij,quad_D_ij)

    #Find and return f_alpha if positive root exists and d_max if not
    if delta_t > 0:
        dist = v_0i*delta_t
        f_alpha = min(dist,d_max)
        return f_alpha
    else:
        return d_max
#------------------------------------------------------------------------------#
#No speedup: @jit
def bypass_funcs(dx_ij,dy_ij,alpha,d_max):

    #Check whether in contact and if so then return d_max or 0 depending on alpha

    #Check angle of possible movement - currently set to 90'
    #b_delta = math.asin((rsum_ij)/dist)
    b_delta = np.pi/2
    b_direction = np.arctan2(-dy_ij, -dx_ij)

    #Lower and upper bounds for obstructed horizon
    b1, b2 = putils.wrap_angle( np.array([ b_direction - b_delta, b_direction + b_delta]) )

    #If alpha points towards j then return 0 else return d_max
    if (alpha > b1 or alpha < b2):
        return 0
    else:
        return d_max

    #If none of the special cases apply return a negative value
    return -1
#------------------------------------------------------------------------------#
def f(alpha,i,params,debug=False):
    """ Compute the minimum distance to collision in this direction"""

    rsum_i,dx_i,dy_i,quad_C2_i,in_field,(v_xi,v_yi),(dv_x,dv_y),quad_A2 = params

    #Compute movement in direction alpha at a comfortable walking speed

    quad_B_i = 2*(dv_x*dx_i + dv_y*dy_i) #used in smallest_positive_quadroot
    quad_D_i = quad_B_i**2-quad_A2*quad_C2_i

    #Find distance to collisions with persons
    f_persons = d_max
    for j in in_field:
        if ( i!=j ):
            f_persons = min( f_persons, f_j( alpha, i, j, dx_i[j],dy_i[j], d_max, rsum_i[j],v_0[i], v[0,j],v[1,j],dv_x[j],dv_y[j], quad_A2[j],quad_B_i[j],quad_C2_i[j],quad_D_i[j]) )
    #Find distance to collisions with walls
    f_walls = d_max
    for w in range(n_walls):
        f_walls = min( f_walls , f_w( alpha, i, walls[:,w], x[0][i], x[1][i], v_0[i], r[i], d_max, v_xi, v_yi) )
#        print(i,alpha*180/pi,w,f_walls)

    #Choose the smallest distance to collision
    f_alpha = min( f_persons , f_walls)

    #If collision is further than target destination in direction of alpha then set
    #f_alpha to d_max so that distance function gives a value of 0 in this direction
    if abs(alpha - alpha_0[i]) <= ar/2:
        d_des = math.hypot(x[0][i] - o[0][i], x[1][i] - o[1][i])
        if d_des < f_alpha:
            f_alpha = d_max

    return f_alpha,f_walls,f_persons
#------------------------------------------------------------------------------#
def compute_d_h(i,params,alpha_out,d_w,d_p):
    """
    compute shortest distance to collision along angle alpha_out
    """

    rsum_i,dx_i,dy_i,quad_C2_i,in_field,(v_xi,v_yi),(dv_x,dv_y),quad_A2_i = params

    quad_B_i = 2*(np.cos(alpha_out)*dx_i + np.sin(alpha_out)*dy_i) #used in smallest_positive_quadroot
    quad_D_i = quad_B_i**2-quad_A2_i*quad_C2_i

    #compute d_p
    d_p_new = 100000.0

    #check for contact
    if (d_p==0):
        d_p_new = 0
    else:
        #loop through in_field
        for j in in_field:
            #compute spq
            if i != j:
                delta_t = putils.smallest_positive_quadroot(2.0,quad_B_i[j],quad_D_i[j])
                if delta_t>=0: d_p_new = min(d_p_new,delta_t)

    d_h_out = min(d_p_new,d_w)

    return d_h_out
#------------------------------------------------------------------------------#
#To profile speed of the code line by line: @profile
def compute_alpha_des(i, params,display_falpha = False, display_distancefunc = False):
    """ Compute the minimum distance function to find alpha_des over
    the horizon of alpha values"""

    #distance to walls
    for w in range(n_walls):
        contact_w[i][w] = distance_to_wall(i,w)
        if contact_w[i][w] >= r[i]: contact_w[i][w] = 0
    #Set the range of alphas to compute f_alpha over
    alphas = np.arange(alpha_current[i] - H[i], alpha_current[i] + H[i], ar)
    #Make sure alpha values are between 0 and 2*pi
    alphas = putils.wrap_angle(alphas)
    #Compute the values for each of the alphas
    v_xi_a = np.cos(alphas)*v_0[i]
    v_yi_a = np.sin(alphas)*v_0[i]

    dv_x = np.subtract.outer(v_xi_a,v[0])
    dv_y = np.subtract.outer(v_yi_a,v[1])

    quad_A2 = 2*((dv_x)**2 + (dv_y)**2) #used in smallest_positive_quadroot

    if (n<-3):
        #In parallel for enough pedestrians
        #f_alphas = Parallel(n_jobs=num_cores)(delayed(f)(alphas[index],i,params) for index in range(len(alphas)))
        print("not used")
    else:
        #In serial for few pedestrians
        f_alphas = np.zeros(len(alphas))
        f_walls_array = np.zeros_like(f_alphas)
        f_persons_array = np.zeros_like(f_alphas)
        for index in range(len(alphas)):

            params[5] = (v_xi_a[index],v_yi_a[index])
            params[6] = (dv_x[index],dv_y[index])
            params[7] = quad_A2[index]
            f_alphas[index],f_walls_array[index],f_persons_array[index] = f(alphas[index],i,params)


    #Distance function for given alphas and f_alphas
    distances = d_max ** 2 + np.power(f_alphas,2) - 2. * d_max * np.multiply(f_alphas, np.cos(alpha_0[i] - alphas))

    #Set alpha_des to minimum of value given by the distance function
    min_distance_index = np.argmin(distances)
    alpha_out = alphas[min_distance_index]
    f_alpha_out = f_alphas[min_distance_index]

    #Find distance to collision along desired
    #direction
    d_w = f_walls_array[min_distance_index]
    d_p = f_persons_array[min_distance_index]
    d_h_i = compute_d_h(i,params,alpha_out,d_w,d_p)

    #If plots are asked for
    if display_falpha:
        putils.plot_f_alpha(i, alphas, f_alphas)
    if display_distancefunc:
        putils.plot_distance_func(i, alphas, distances)

    #Format and return output
    result = [alpha_out, f_alpha_out, d_h_i]
    return result

#------------------------------------------------------------------------------#
def compute_bodycollision_acceleration(i):
    """ Returns acceleration in [x,y] directions caused by body collisions
        for person i current positions of persons """

    axt = 0
    ayt = 0
    ri = r[i]
    #Collisions due to persons
    for j in range(n):
        if contact_p[i][j] != 0:
            kg = k * (ri + r[j] - contact_p[i][j])
            nx = x[0][i] - x[0][j]
            ny = x[1][i] - x[1][j]
            size_n = math.hypot(nx,ny)
            nx = nx / size_n
            ny = ny / size_n
            fx = kg * nx
            fy = kg * ny
            ax = fx / mass[i]
            ay = fy / mass[i]
            axt = axt + ax
            ayt = ayt + ay
    #Collisions due to walls
    for w in range(n_walls):
        if contact_w[i][w] != 0:
            kg = k * (ri - contact_w[i][w])
            #find normal direction to wall
            wall = walls[:,w]
            a = wall[0]
            b = wall[1]
            c = wall[2]
            wall_start = wall[3] - ri
            wall_end = wall[4] + ri
            if a == 0:
                nx = 0
                if x[1][i] >= -c/b:
                    ny = 1
                else:
                    ny = -1
            elif b == 0:
                if x[0][i] >= -c/a:
                    nx = 1
                else:
                    nx = -1
                ny = 0
            else:
                '''Add direction of normal vector based on location of person i'''
                nx = 1
                ny = b / a
                ntot = math.sqrt( nx * nx + ny * ny )
                nx = nx / ntot
                ny = ny / ntot
            '''Deal with case when end of the wall point is in contact with a person
             as it changes the normal direction of the contact force with the wall'''
            fx = kg * nx
            fy = kg * ny
            ax = fx / mass[i]
            ay = fy / mass[i]
            axt = axt + ax
            ayt = ayt + ay

    return [axt, ayt]

#------------------------------------------------------------------------------#
def compute_destinations():
    """ Calculates v_des, f_alpha_des, and alpha_des values for all persons"""
    global alpha_des, f_alpha_des, v_des, gap, contact_p,in_field, d_h


    d_h = tau*v_0

    rsum = np.add.outer(r,r) #ri+rj
    gap = scidist.squareform(scidist.pdist(x.T)) #distance between i and j
    contact_p = np.zeros_like(gap)
    contact_p[gap<rsum] = gap[gap<rsum]

    dx,dy = np.subtract.outer(x[0],x[0]),np.subtract.outer(x[1],x[1])
    quad_C2 = 2*(dx**2 + dy**2 - rsum**2) #used in smallest_positive_quadroot
    b_dir = np.arctan2(dy,dx) #used to compute in_field below
    phi1 = putils.wrap_angle(alpha_current-H_min)
    phi2 = putils.wrap_angle(alpha_current+H_min)

    del_phi = np.abs(b_dir-alpha_current)
    for i in range(n):
        in_field = np.where((del_phi[:,i]<H_min) | ((2*np.pi-del_phi[:,i])<H_min)) #Assumes H_min=H_max
        params = [rsum[i,:],dx[i,:],dy[i,:],quad_C2[i,:],in_field[0],None,None,None]
        alpha_des[i],f_alpha_des[i],d_h_i = compute_alpha_des(i,params)
        d_h[i] = min(d_h[i],d_h_i)

#    v_des = np.min(np.vstack([v_0,f_alpha_des/tau]),axis=0)
    v_des = np.min(np.vstack([v_0,d_h/(1*tau)]),axis=0)

#------------------------------------------------------------------------------#
def move_pedestrians():
    """ Moves all pedestrians forward in time by time_step based on calculated v_des and alpha_des values"""
    #Global values being saved
    global v, x

    #acceleration due to body collisions - needs to computed before moving the pedestrians to ensure both people colliding feel the force
    abcx = np.zeros(n)
    abcy = np.zeros(n)
    for i  in range(n):
        [abcx[i],abcy[i]] = compute_bodycollision_acceleration(i)

    #acceleration
    a = np.zeros_like(v)
    a[0,:] = (np.cos(alpha_des)*v_des-v[0,:])/tau + abcx
    a[1,:] = (np.sin(alpha_des)*v_des-v[1,:])/tau + abcy

    #update
    v = v + a * time_step
    x = x + v * time_step
#------------------------------------------------------------------------------#
def update_model():
    """ Once alpha_des, v_des have been calculated and pedestrians have moved forward in time to ready the model for the next iteration"""
    global alpha_0, alpha_current, x_full, v_full, t
    #update alpha_0 values
    alpha_0 = np.arctan2((o[1]-x[1]),(o[0]-x[0]))
    alpha_current = np.arctan2(v[1,:],v[0,:])
    ind = v[0,:]==0
    ind[v[1,:]!=0]=False
    alpha_current[ind]=alpha_0[ind]
    #save information about positions of each individual
    x_full = np.dstack((x_full,x))
    v_full = np.dstack((v_full,v))
    #increment time
    t = t + time_step
#------------------------------------------------------------------------------#
def advance_model():
    """Advances current model in time by time_step"""

    compute_destinations()
    move_pedestrians()
    update_model()
#------------------------------------------------------------------------------#


#Initialize values of None for all variables
reset_model()
