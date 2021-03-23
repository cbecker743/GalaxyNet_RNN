import tqdm
import numpy as np
import tensorflow as tp
import astropy.cosmology as ac
import scipy.stats as st
import halotools.mock_observables as htmo
import catalogues as cat
from scaling import *
from observations import *

class PSOParticle():

    def __init__(self, position, velocity, number=0):
        self.position      = position
        self.value         = np.inf
        self.pbest_postion = position
        self.pbest_value   = np.inf
        self.velocity      = velocity
        self.number        = number
        
    def __str__(self):
        message1 = 'Particle {}:\n'.format(self.number)
        message2 = '   Position: {}\n'.format(self.position)
        message3 = '   Velocity: {}\n'.format(self.velocity)
        message4 = '   Personal best position: {}\n'.format(self.pbest_postion)
        message5 = '   Best personal loss: {}'.format(self.pbest_value)
        message  = message1+message2+message3+message4+message5
        return message

class PSOSwarm():
    
    def __init__(self, n_particles, start_position, w=0.5, c1=0.8, c2=0.9, a=1.0, b=1.0,
                 init_pos=0.1, init_vel=0.5, seed=42, w_min=None, gstop=0.0):
        self.n_particles = n_particles
        self.n_dimensions = start_position.size
        self.start_position = start_position
        self.init_pos = init_pos
        self.init_vel = init_vel
        self.seed = seed
        self.rng = np.random.RandomState()
        self.particles = self.set_particles()
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        self.a  = a
        self.b  = b
        self.gmean_value = np.inf
        self.gbest_value = np.inf
        self.gbest_position = start_position
        self.gbest_index = 0
        self.w_max = w
        self.w_min = w_min
        self.position_history = None
        self.velocity_history = None
        
    def set_particles(self):
        self.rng.seed(self.seed)
        positions = []
        particles = []
        for i in range(self.n_particles):
            position = (1.0 + self.init_pos * self.rng.uniform(-1.0,1.0,size=self.start_position.size)) * self.start_position
            positions.append(position)
        for i in range(self.n_particles):
            position = positions[i]
            other_positions = positions.copy()
            del other_positions[i]
            index = np.arange(len(other_positions))
            index = self.rng.choice(index)
            other_position = other_positions[index]
            velocity = self.init_vel * (other_position - position)
            particle = PSOParticle(position,velocity,i)
            particles.append(particle)
        return particles
    
    def print_particles(self):
        for particle in self.particles:
            print(particle.__str__())
            
    def get_loss(self, loss_function, psodict):       
        for particle in tqdm.tqdm_notebook(self.particles, desc='Particles', leave=False):
            particle.value = loss_function(particle.position,psodict=psodict)
            print('Loss {}: {}'.format(particle.number,particle.value))

    def set_pbest(self):
        for particle in self.particles:
            if (particle.pbest_value > particle.value):
                particle.pbest_value = particle.value
                particle.pbest_position = particle.position

    def set_gbest(self, iteration):
        for i,particle in enumerate(self.particles):
            if (self.gbest_value > particle.value):
                self.gbest_value = particle.value
                self.gbest_position = particle.position
                self.gbest_index = np.array([iteration,i])
                
    def get_gmean(self, gstop, loss_max=1.0e10, n_loss=10):
        loss_values = np.array([particle.value for particle in self.particles])
        loss_values = loss_values[loss_values < loss_max]
        if loss_values.size > n_loss:
            loss_values = loss_values[loss_values.argsort()[:n_loss]]        
        gmean_value = loss_values.mean()
        if np.sqrt((gmean_value-self.gmean_value)**2.0) <= gstop:
            stop = True
        else:
            stop = False
        self.gmean_value = gmean_value
        return stop
        
    def move_particles(self):
        for particle in self.particles:
            particle.velocity = (self.w * particle.velocity) + (self.c1 * self.rng.random()) * (particle.pbest_position - particle.position) + (self.c2 * self.rng.random()) * (self.gbest_position - particle.position)
            particle.position = self.a * particle.position + self.b * particle.velocity

    def update_history(self):
        positions  = np.empty((0,self.n_dimensions))
        velocities = np.empty((0,self.n_dimensions))
        for particle in self.particles:
            positions  = np.concatenate([positions, particle.position.reshape(1,-1)],axis=0)
            velocities = np.concatenate([velocities,particle.velocity.reshape(1,-1)],axis=0)
        self.position_history = np.concatenate([self.position_history,positions.reshape(1,positions.shape[0],positions.shape[1])],axis=0)
        self.velocity_history = np.concatenate([self.velocity_history,velocities.reshape(1,positions.shape[0],positions.shape[1])],axis=0)
                
    def save_history(self, file):
        hf = hdf5.File(file, 'w')
        hf.create_dataset('Positions',  data=self.position_history)
        hf.create_dataset('Velocities', data=self.velocity_history)
        hf.close()
        
    def save_gbest(self, file):
        hf = hdf5.File(file, 'w')
        hf.create_dataset('Best_position', data=self.gbest_position)
        hf.create_dataset('Best_loss',     data=self.gbest_value)
        hf.create_dataset('Best_index',    data=self.gbest_index)
        hf.close()
                
    def train(self, loss_function, psodict, n_iterations=10, hist_file='pso.history.h5', gbest_file='pso.gbest.h5', gstop=0.0, n_loss=10):
                
        self.rng.seed(self.seed+1)
        self.position_history = np.empty((0,self.n_particles,self.n_dimensions))
        self.velocity_history = np.empty((0,self.n_particles,self.n_dimensions))
        iteration = 0
        for iteration in tqdm.tqdm_notebook(range(n_iterations),desc = 'Steps'):
            if self.w_min is not None:
                self.w = self.w_max - (self.w_max - self.w_min) * (iteration / n_iterations)
            self.get_loss(loss_function,psodict=psodict)
            self.set_pbest()    
            self.set_gbest(iteration)
            self.update_history()
            self.save_history(hist_file)
            self.save_gbest(gbest_file)
            self.move_particles()
            stop = self.get_gmean(gstop=gstop,n_loss=n_loss)
            print('Mean loss:',self.gmean_value)
            print('Best loss:',self.gbest_value)
            if stop:
                print('Mean global loss does no longer change. Stopping the training process.')
                break
            iteration += 1 
        print('Best position: ', self.gbest_position)
        print('Best loss: ', self.gbest_value)

################################################################
#start MSO
################################################################
                
class MSOParticle():

    def __init__(self, position, velocity, mso_type, number=0, q=0):
        self.position      = position
        self.value         = np.inf
        self.pbest_postion = position
        self.pbest_value   = np.inf
        self.velocity      = velocity
        self.number        = number
        self.charge        = self.set_charges(mso_type, number, q)
        
    def __str__(self):
        message1 = 'Particle {}:\n'.format(self.number)
        message2 = '   Position: {}\n'.format(self.position)
        message3 = '   Velocity: {}\n'.format(self.velocity)
        message4 = '   Personal best position: {}\n'.format(self.pbest_postion)
        message5 = '   Best personal loss: {}\n'.format(self.pbest_value)
        message6 = '   Charge: {}'.format(self.charge)
        message  = message1+message2+message3+message4+message5+message6
        return message
    
    def set_charges(self, mso_type, number, q):
        if mso_type == 'atomic':
            if (number % 2) == 0:
                charge = 0
            else:
                charge = q
        elif mso_type == 'charged':
            charge = q
        elif mso_type == 'neutral':
            charge = 0
        return charge
    
class MSOSwarm():
    
    def __init__(self, n_swarms, n_particles, start_position, w_max, c1_max, c2_min, c3_max, a, b, seed, init_pos, init_vel,
                 mso_type, q):
        self.n_swarms = n_swarms
        self.n_particles = n_particles
        self.n_dimensions = start_position.size
        self.start_position = start_position
        self.seed = seed
        self.init_pos = init_pos
        self.w = w_max
        self.c1 = c1_max
        self.c2 = c2_min
        self.c3 = c3_max
        self.a = a
        self.b = b
        self.init_vel = init_vel
        self.smean_value = np.inf
        self.sbest_value = np.inf
        self.sbest_iteration_value = np.inf
        self.sbest_position = start_position
        self.gbest_position = start_position
        self.rng = np.random.RandomState()
        self.particles = self.set_particles(mso_type, q)

    def set_particles(self, mso_type, q):
        self.rng.seed(self.seed)
        positions = []
        particles = []
        for i in range(self.n_particles):
            position = (1.0 + self.init_pos * self.rng.uniform(-1.0,1.0,size=self.start_position.size)) * self.start_position
            positions.append(position)
        for i in range(self.n_particles):
            position = positions[i]
            other_positions = positions.copy()
            del other_positions[i]
            index = np.arange(len(other_positions))
            index = self.rng.choice(index)
            other_position = other_positions[index]
            velocity = self.init_vel * (other_position - position)
            particle = MSOParticle(position, velocity, mso_type, i, q)
            particles.append(particle)
        return particles
            
    def get_loss(self, loss_function, psodict, s_number):
            for particle in tqdm.tqdm_notebook(self.particles, desc='Particles', leave=False):
                particle.value = loss_function(particle.position,psodict=psodict)
                print('Loss Swarm {} Particle(q = {}) {}: {}'.format(s_number,particle.charge,particle.number,particle.value))
            
    def set_pbest(self):
        for particle in self.particles:
            if (particle.pbest_value > particle.value):
                particle.pbest_value = particle.value
                particle.pbest_position = particle.position
    
    def set_sbest(self, iteration, s_number):
        numbers = []
        for i, particle in enumerate(self.particles):
            numbers.append(particle.value)
            if (self.sbest_value > particle.value):
                self.sbest_value = particle.value
                self.sbest_position = particle.position
            self.sbest_iteration_value = np.min(numbers)
            
            
                
    def get_smean(self, loss_max=1.0e10, n_loss=10):
        loss_values = np.array([particle.value for particle in self.particles])
        loss_values = loss_values[loss_values < loss_max]
        if loss_values.size > n_loss:
            loss_values = loss_values[loss_values.argsort()[:n_loss]]        
        smean_value = loss_values.mean()
        self.smean_value = smean_value
      
    def move_particles(self, swarms, s_number, n_swarms, iteration, q_desc, alpha, RC, RP):
        if n_swarms == 1:
            for particle in self.particles:
                particle_index = particle.number
                particle.velocity = (self.w * particle.velocity) + (self.c1 * self.rng.random()) * (particle.pbest_position - particle.position) + (self.c2 * self.rng.random()) * (self.sbest_position - particle.position)
                particle.position = self.a * particle.position + self.b * particle.velocity
        else:
            for particle in self.particles:
                particle_index = particle.number
                acc = self.get_acceleration(swarms, particle, particle_index, s_number, iteration, q_desc, alpha, RC, RP)
                particle.velocity = (self.w * particle.velocity) + (self.c1 * self.rng.random()) * (particle.pbest_position - particle.position) + (self.c2 * self.rng.random()) * (self.sbest_position - particle.position) + (self.c3 * self.rng.random()) * (self.gbest_position - particle.position) + acc
                #print('Hyperparameter_self: c1: {} c2: {} c3: {} w: {}'.format(self.c1,self.c2,self.c3,self.w))
                particle.position = self.a * particle.position + self.b * particle.velocity

#         else:
#             for particle in self.particles:
#                 self.c2 = self.calculate_c2(self.c2test_max, self.c2test_min, particle.pbest_value, self.sbest_value)
#                 print('c2_2: {}'.format(self.c2))
#                 particle_index = particle.number
#                 acc = self.get_acceleration(swarms, particle, particle_index, s_number)
#                 particle.velocity = (self.w * particle.velocity) + (self.c2 * self.rng.random()) * (self.sbest_position - particle.position) + (self.c3 * self.rng.random()) * (self.gbest_position - particle.position) + acc
#                 particle.position = self.a * particle.position + self.b * particle.velocity
    
#     def calculate_c2(self, c2_max, c2_min, particle_bvalue, swarm_bvalue):
#         m = (swarm_bvalue - particle_bvalue)/(swarm_bvalue + particle_bvalue)
#         c2 = (c2_min + c2_max)/2 + (c2_max - c2_min)/2 + (np.exp(-m) - 1)/(np.exp(-m) + 1)
#         return c2

    
    def get_acceleration(self, swarms, active_particle, particle_index, swarm_index, iteration, q_desc, alpha, RC, RP):
        accs =[]
        acc = []
        for swarm_counter, swarm in enumerate(swarms):
            for particle_counter, particle in enumerate(swarm.particles):
                diff = active_particle.position - particle.position
#                 print('diff: {}'.format(np.linalg.norm(diff)))
                if swarm_counter != swarm_index or particle_counter != particle_index: #prohibit operation for same particle
                    if q_desc and iteration >= 5: # implements schedule for the charge q 
                        if active_particle.charge == 0:
                            if RC <= np.linalg.norm(diff) and RP >= np.linalg.norm(diff):
                                acc2 = ((active_particle.charge*particle.charge)/(np.linalg.norm(diff))**3)*(diff)
                            elif np.linalg.norm(diff) < RC:
                                acc2 = (active_particle.charge*particle.charge*diff)/(RC**2*np.linalg.norm(diff))
                            elif np.linalg.norm(diff) > RP:
                                acc2 = np.zeros(np.array(diff).shape[0]).tolist()
                            accs.append(acc2)
                        else:
                            q_eff = active_particle.charge*alpha**iteration + 1
                            if RC <= np.linalg.norm(diff) and RP >= np.linalg.norm(diff):
                                acc2 = ((active_particle.charge*alpha**iteration + 1)*(particle.charge)/(np.linalg.norm(diff))**3)*(diff)
                            elif np.linalg.norm(diff) < RC:
                                acc2 = ((active_particle.charge*alpha**iteration + 1)*(particle.charge)*diff)/(RC**2*np.linalg.norm(diff))
                            elif np.linalg.norm(diff) > RP:
                                acc2 = np.zeros(np.array(diff).shape[0]).tolist()
                            accs.append(acc2)
                    else: #classic charge model without q_desc
                        if RC <= np.linalg.norm(diff) and RP >= np.linalg.norm(diff):
                            acc2 = ((active_particle.charge*particle.charge)/(np.linalg.norm(diff))**3)*(diff)
                        elif np.linalg.norm(diff) < RC:
                            acc2 = (active_particle.charge*particle.charge*diff)/(RC**2*np.linalg.norm(diff))
                        elif np.linalg.norm(diff) > RP:
                            acc2 = np.zeros(np.array(diff).shape[0]).tolist()
                        accs.append(acc2)
                    
        for counter, i in enumerate(accs):
            if counter == 0: 
                acc.append(i)
            else:
                sum_vector = [sum(x) for x in zip(acc[0], i)] #
                acc = []
                acc.append(sum_vector)
        return acc[0]
                
    def particle_death(self, pDeath, gbest, mso_type, q):
        dead_particles = []
        for number, particle in enumerate(self.particles):
            if particle.pbest_value <= gbest: #prevent global best particle from dying
                pass
            else:
                num1 = self.rng.randint(0,1000)
                if num1 < pDeath*1000:
                    del self.particles[number]
                    position = (1.0 + self.init_pos * self.rng.uniform(-1.0,1.0,size=self.start_position.size)) * self.start_position
                    other_positions = [other_particle.position for other_particle in self.particles]
                    index = np.arange(len(other_positions))
                    index = self.rng.choice(index)
                    other_position = other_positions[index]
                    velocity = self.init_vel * (other_position - position)
                    particle = MSOParticle(position, velocity, mso_type, number, q)
                    self.particles[number:number] = [particle]
                    dead_particles.append(number)
        return dead_particles
    
    def particle_swap(self, swarms, pSwap, s_number):
        swapped_particles = []
        asd = 0 #only 1 swap allowed
        for number, active_particle in enumerate(self.particles):
            if asd == 0:
                num1 = self.rng.randint(0,1000)
                if num1 < pSwap*1000:
                    asd = 1
                    swarm_index = self.rng.randint(0,len(swarms)-1)
                    if swarm_index == s_number:
                        print('ERROR cant swap with same swarm')
                        if swarm_index !=0:
                            swarm_index = swarm_index - 1
                        else:
                            swarm_index =swarm_index + 1    
                    particle_index = self.rng.randint(0,self.n_particles-1)
                    new_number = swarms[swarm_index].particles[particle_index].number
                    swarms[swarm_index].particles[particle_index].number = number
                    active_particle.number = new_number
                    swarms[swarm_index].particles[new_number+1:new_number+1] = [active_particle]
                    self.particles[number+1:number+1] = [swarms[swarm_index].particles[particle_index]]
                    del swarms[swarm_index].particles[particle_index]
                    del self.particles[number]
                    swapped_particles.append(tuple((number, new_number)))
                    swapped_particles.append(tuple((0, swarm_index)))
        return swapped_particles
                                 

    def print_particles(self):
        for particle in self.particles:
            print(particle.__str__())

    def save_history(self, file):
        hf = hdf5.File(file, 'w')
        hf.create_dataset('Positions',  data=self.position_history)
        hf.create_dataset('Velocities', data=self.velocity_history)
        hf.close()

class MSO():
    
    def __init__(self, n_swarms, n_particles, start_position, mso_type='neutral',pDeath = 0.01, pSwap=0.01, w_max=0.9,
                 w_min=0.4, c1_max=1.1, c1_min=0.9, c2_max=1.1, c2_min=0.9, c3_max=0.45, c3_min=0.2, a=1.0, b=1.0,
                 seed=42, init_pos=0.05, init_vel=0.5, q=1.25, q_desc = True, alpha=0.7, RC = 0.8, RP = np.sqrt(3)*2):
        self.n_swarms = n_swarms
        self.n_particles = n_particles 
        self.start_position = start_position 
        self.swarms = []
        self.swarms_best_pos = np.array([[np.zeros(start_position.size) for i in range(1)] for j in range(n_swarms)]).reshape(n_swarms,start_position.size)
        self.swarms_best_loss = np.array([[[0] for i in range(1)] for j in range(self.n_swarms)]).reshape(n_swarms)
        self.w_max = w_max
        self.w_min = w_min
        self.w = w_max
        self.c1_max = c1_max
        self.c1_min = c1_min
        self.c1 = c1_max
        self.c2_max = c2_max
        self.c2_min = c2_min
        self.c2 = c2_min
        self.c3_max = c3_max
        self.c3_min = c3_min
        self.c3 = c3_max
        self.a = a
        self.b = b
        self.pDeath = pDeath
        self.pSwap = pSwap
        self.seed = seed
        self.init_pos = init_pos
        self.init_vel = init_vel
        self.g_best = np.inf 
        self.gbest_value = np.inf
        self.gbest_position = start_position
        self.mso_type = mso_type
        self.q = q
        self.q_desc = q_desc
        self.alpha = alpha
        self.RP = RP
        self.RC = RC
        self.set_swarms = self.set_swarms()

    def set_swarms(self):
        for i in np.arange(0, self.n_swarms):
            self.swarms.append(MSOSwarm(self.n_swarms, self.n_particles, self.start_position, self.w_max, self.c1_max, 
                                        self.c2_max, self.c3_max, self.a, self.b, self.seed, self.init_pos, self.init_vel,
                                        self.mso_type, self.q))
            self.seed = self.seed + 1
    
    def train_MSO(self, loss_function, psodict, n_iterations,phase=False, gbest_file='pso.gbest.h5'):
        history = np.zeros((self.n_swarms, n_iterations))
        for counter_it, iteration in enumerate(tqdm.tqdm_notebook(range(n_iterations),desc = 'Steps')):
            self.w = self.linear_decreasing_constant(self.w_max, self.w_min, iteration, n_iterations)
            self.c1 = self.linear_decreasing_constant(self.c1_max, self.c1_min, iteration, n_iterations) 
            self.c2 = self.linear_increasing_constant(self.c2_max, self.c2_min, iteration, n_iterations)
            if phase == True:
                if counter_it >= n_iterations/4:
                    self.c3 = self.linear_decreasing_constant(self.c3_max, self.c3_min, iteration, n_iterations)
                else:
                    self.c3 = 0
            else:
                self.c3 = self.linear_decreasing_constant(self.c3_max, self.c3_min, iteration, n_iterations)
            for s_number, swarm in enumerate (self.swarms):
                swarm.w = self.w
                swarm.c1 = self.c1
                swarm.c2 = self.c2
                swarm.c3 = self.c3
                death_list = swarm.particle_death(self.pDeath, self.gbest_value, self.mso_type, self.q) 
                swap_list = swarm.particle_swap(self.swarms, self.pSwap, s_number)
                swarm.get_loss(loss_function, psodict=psodict, s_number=s_number)
                swarm.set_pbest() 
                swarm.set_sbest(iteration, s_number)
                if (self.gbest_value > swarm.sbest_value):
                    self.gbest_value = swarm.sbest_value
                    self.gbest_position = swarm.sbest_position
                swarm.gbest_position = self.gbest_position
                self.swarms_best_pos[s_number]  = swarm.sbest_position
                self.swarms_best_loss[s_number] = swarm.sbest_value
                history[s_number][counter_it] = swarm.sbest_iteration_value
                self.save_gbest(history, gbest_file)
                if len(death_list) == 1:
                    print('Particle {} died ... and was replaced'.format(death_list[0]))
                elif len(death_list) > 1:
                    print('Particle {} died ... and were replaced'.format(str(death_list)[1:-1]))
                if len(swap_list) == 2:
                    print('Swarm  {} Particle {} swapped with Swarm {} Particle {}'.format(s_number,swap_list[0][0],swap_list[1][1],swap_list[0][1]))
                swarm.move_particles(self.swarms, s_number, self.n_swarms, iteration, self.q_desc, self.alpha, self.RC, self.RP) 
                swarm.get_smean()
                print('Mean loss:',swarm.smean_value)
                print('Best Swarm loss:',swarm.sbest_value)
                print('Best MS loss:',self.gbest_value)
                print('Hyperparameter: c1: {} c2: {} c3: {} w: {}'.format(swarm.c1,swarm.c2,swarm.c3,swarm.w))

                
    def save_gbest(self, history, file):
        hf = hdf5.File(file, 'w')
        hf.create_dataset('Best_position', data=self.gbest_position)
        hf.create_dataset('Best_loss',     data=self.gbest_value)
        hf.create_dataset('Swarm_best_positions',    data=self.swarms_best_pos)
        hf.create_dataset('Swarm_best_losses',    data=self.swarms_best_loss)
        hf.create_dataset('Swarm_history',    data=history)
        hf.close()
    
    def linear_increasing_constant(self, c_max, c_min, iteration, n_iteration):
        constant = (c_min - c_max) * ((n_iteration - iteration)/ n_iteration) + c_max
        return constant
    
    def linear_decreasing_constant(self, c_max, c_min, iteration, n_iteration):
        constant = (c_max - c_min) * ((n_iteration - iteration)/ n_iteration) + c_min
        return constant
    

################################################################
#start Simulated Annealing
################################################################

class simulated_annealing():
    
    def __init__(self, start_position):
        self.start_position = start_position
        self.final_state = start_position
        self.final_loss = np.inf
        self.rng = np.random.RandomState(seed = 42)
        
    def random_neighbour(self, x, scale):
        pos = x + ((self.rng.uniform(-1,1,size=x.size)) * scale)
            
        return pos 
    
    def acceptance_probability(self, cost, new_cost, temperature):
        if new_cost < cost:
            return 1
        else:
            p = np.exp(- (new_cost - cost) / temperature)
            return p
  
    # additive cooling schemes
    def additive_linear_decay(self, c0, cn, iteration, n_iteration):
        c = (c0 - cn) * ((n_iteration - iteration)/ n_iteration) + cn
        return c
    
    def additive_quadratic_decay(self, c0, cn, iteration, n_iteration):
        c = (c0 - cn) * ((n_iteration - iteration)/ n_iteration)**2 + cn
        return c
    
    def additive_exp_decaying(self, c0, cn, iteration, n_iterations):
        c = cn + (c0 - cn)*(1/(1+np.exp(2*np.log(c0-cn)*(iteration-0.5*n_iterations)/n_iterations)))
        return c
    
    def additive_trigonometric_decay(self, c0, cn, iteration, n_iteration):
        c = cn +0.5*(c0-cn) * (1+ np.cos(iteration*np.pi/n_iteration))
        return c
    
    # multiplicative cooling shcmes
    def multiplicative_exp_decay(self, c0, iteration, alpha = 0.95):
        c = c0 * alpha**iteration
        return c
    
    def multiplicative_log_decay(self, c0, iteration, alpha = 1.5):
        c = c0 / (1+alpha*np.log10(1+iteration))
        return c
    
    def save_gbest(self, file):
        hf = hdf5.File(file, 'w')
        hf.create_dataset('Best_position', data=self.final_state)
        hf.create_dataset('Best_loss',     data=self.final_loss)
        hf.close()
    
    def train(self, loss_function, psodict,scale_max, scale_min, gbest_file='sa_best_2.h5', maxsteps=1000, T0 = 300, Tn = 0,
              alpha=0.95,  debug=True):
        state = self.start_position
        cost = loss_function(state, psodict)
        states, costs = [state], [cost]
        for step in range(maxsteps):
            T = self. multiplicative_exp_decay(T0, step)
            scale = self.additive_linear_decay(scale_max, scale_min, step, maxsteps)
            new_state = self.random_neighbour(state, scale)
            new_cost = loss_function(new_state, psodict)
            if debug: print('Step: {}, T: {}, Scale: {} Cost: {}, New_cost: {}'.format(step, T,scale, cost, new_cost))
            if self.acceptance_probability(cost, new_cost, T) > self.rng.random():
                #print(self.rng.random())
                state, cost = new_state, new_cost
                print('accepted')
                states.append(state)
                costs.append(cost)
            self.final_state = states[-1:]
            self.final_loss = costs[-1:] 
            self.save_gbest(gbest_file)   

################################################################
#start Optimizer ... brute force approach
################################################################ 
class optimizer():
    
    def __init__(self, start_position):
        self.final_state = start_position
        self.final_loss = np.inf
        
    def train(self, loss_function, psodict, pos = 0, delta = 1, scale=0.1, gbest_file='optimizer.h5'): 
        self.final_loss = loss_function(self.final_state, psodict)
        number = np.round((self.final_state.size - pos)/delta) 
        for i in tqdm.tqdm_notebook(np.arange(number-1), desc='steps', leave=False):
            change_max2 = delta*(i+1) + pos 
            change_min2 = delta*i + pos 
            params = []
            for counter, value in enumerate (self.final_state):
                if counter >= change_min2 and counter < change_max2:
                    value = value + scale 
                    params.append(value)
                else:
                    params.append(value)
            params2 = np.array(params)
            new_cost = loss_function(params2, psodict)
            print('+++ new_cost: {} old_cost: {} _____range: {} - {}'.format(new_cost, self.final_loss,change_min2,change_max2))
            if new_cost < self.final_loss:
                self.final_loss = new_cost
                self.final_state = params2
                self.save_gbest(gbest_file) 
            else:
                params =[]
                for counter2, value2 in enumerate (self.final_state):
                    if counter2 >= change_min2 and counter2 < change_max2:
                        value2 = value2 - scale 
                        params.append(value2)
                    else:
                        params.append(value2)
                params2 = np.array(params)
                new_cost = loss_function(params2, psodict)
                print('--- new_cost: {} old_cost: {} _____range: {} - {}'.format(new_cost, self.final_loss,change_min2,change_max2))

                if new_cost < self.final_loss:
                    self.final_loss = new_cost
                    self.final_state = params2
                    self.save_gbest(gbest_file) 
                            
    def save_gbest(self, file):
        hf = hdf5.File(file, 'w')
        hf.create_dataset('Best_position', data=self.final_state)
        hf.create_dataset('Best_loss',     data=self.final_loss)
        hf.close()