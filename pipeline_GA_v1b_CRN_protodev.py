# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as pl
import myokit
import numpy as np
#import csv
#import time
from multiprocessing import Pool

#####################################################################
#1) Protocol optimization with GA
#####################################################################
num_par = 4
sol_per_pop = 200
Tournament_size = 4
num_parents_mating = int(0.1*sol_per_pop)
num_generations = 50 #20 #50
hold = -80.1
t_hold = 100.0

# Creating the initial population
V_t = []
for i in range(sol_per_pop):
    aaa = np.random.uniform(-120, 50, num_par) #-150.01
    a = []
    for aa in aaa:
        if aa > -1 and aa < 1:
            aa = 1.01
        a.append(aa)
    b = np.random.uniform(10.0, 200.0, num_par) #100.0
    ramp = np.random.randint(2, size=num_par)
    combi = list(zip(a, b, ramp))
    V_t.append(combi)
new_population = V_t

# Callable functions
def cal_pop_fitness(Vt):
    V = np.array(Vt)[:, 0]
    t = np.array(Vt)[:, 1]
    ramp = np.array(Vt)[:, 2]
    #print Vt
    #print V
    #print t
    #print ramp
    
    m, p, s = myokit.load("atrial model/CRN_1998_human_atrial_CM_2D.mmt")

    #Set up voltage-clamp simulation
    # Get pacing variable, remove binding
    p = m.get('membrane.level')
    p.set_binding(None)
    # Get membrane potential, demote to an ordinary variable
    v = m.get('membrane.V')
    v.demote()
    v.set_rhs(0)
    #v.set_binding('pace') # Bind to the pacing mechanism
    mem = m.get('membrane')            
    vp = mem.add_variable('vp')
    vp.set_rhs(0)
    vp.set_binding('pace')    

    proto = myokit.Protocol()
    proto.add_step(V[0], t[0])
    piecewise_func = 'piecewise(' 

    for i in range(num_par):
        if ramp[i+1] > 0:
            proto.add_step(V[i+1], t[i+1])     
            start = np.sum(t[: i+1])
            end = np.sum(t[: i+2])
            slope = (V[i] - V[i+1])/(start - end)
            intercept = V[i+1] - (slope*end)           
            v_mem = mem.add_variable('volt' + str(i)) # + str(i) + str(j))
            v_mem.set_rhs(str(intercept) + ' + ' + str(slope) + ' * environment.time')
            piecewise_func += "(environment.time >= " + str(start) + " and environment.time < " + str(end) + "), volt" + str(i) + ", "   
            #piecewise_func += "(environment.time >= " + str(start) + " and environment.time < " + str(end) + "), volt" + str(i) + str(j) + ", "   
            #v.set_rhs("piecewise((environment.time >= " + str(start) + " and environment.time < " + str(end) + "), v1, vp)")
        else:
            proto.add_step(V[i+1], t[i+1])
            start = np.sum(t[: i+1])
            end = np.sum(t[: i+2])
            v_mem = mem.add_variable('volt' + str(i)) # + str(i) + str(j))
            v_mem.set_rhs(str(V[i+1])) 
            piecewise_func += "(environment.time >= " + str(start) + " and environment.time < " + str(end) + "), volt" + str(i) + ", "           
            #piecewise_func += "(environment.time >= " + str(start) + " and environment.time < " + str(end) + "), volt" + str(i) + str(j) + ", "            

    piecewise_func += 'vp)'  
    #print(piecewise_func)

    v.set_rhs(piecewise_func)
    
    proto.add_step(V[0], t[0])

    s = myokit.Simulation(m, proto)

    try:
        d = s.run(sum(t) + t_hold, log_interval = 0.1) #, log = ['environment.time', 'membrane.V', 'rapid_delayed_rectifier_K_current.i_Kr', 'membrane.I_tot_abs'], log_interval = 0.1)
        print("Successfully run CRN model")
    except:
        print("Crash in CRN model")
        return [0], [0]
    
    peakcurrent = np.max(d['rapid_delayed_rectifier_K_current.i_Kr'])
                     
    return d, peakcurrent

def eval_fitness(ind_vars):
    ind = ind_vars[0]
    showit = ind_vars[1]
    #track_time = time.time()
    init = np.array([[hold, t_hold, 0]])
    ind_init = np.concatenate((init, ind), axis=0)
    datalog, peak_curr = cal_pop_fitness(ind_init)
                        
    try: 
        tot_mem_curr_abs = np.absolute(datalog["fast_sodium_current.i_Na"]) + np.absolute(datalog["time_independent_potassium_current.i_K1"]) + np.absolute(datalog["transient_outward_K_current.i_to"]) + np.absolute(datalog["ultrarapid_delayed_rectifier_K_current.i_Kur"]) + np.absolute(datalog["rapid_delayed_rectifier_K_current.i_Kr"]) + np.absolute(datalog["slow_delayed_rectifier_K_current.i_Ks"]) + np.absolute(datalog["background_currents.i_B_Na"]) + np.absolute(datalog["background_currents.i_B_Ca"]) + np.absolute(datalog["sodium_potassium_pump.i_NaK"]) + np.absolute(datalog["sarcolemmal_calcium_pump_current.i_CaP"]) + np.absolute(datalog["Na_Ca_exchanger_current.i_NaCa"]) + np.absolute(datalog["L_type_Ca_channel.i_Ca_L"])
        cost = np.divide(np.absolute(datalog['rapid_delayed_rectifier_K_current.i_Kr']), tot_mem_curr_abs)
        np.savetxt("cost.txt", np.c_[datalog['environment.time'], datalog['membrane.V'], datalog['rapid_delayed_rectifier_K_current.i_Kr'], tot_mem_curr_abs, cost], delimiter = ",")
        max_cost = np.max(cost)
    except:
        max_cost = 0
        
    fitness_tot = max_cost
    print(fitness_tot)

    # Display final protocol
    if showit == 1:
        pl.figure("Best protocol")
        pl.subplot(4,1,1)
        pl.title("Voltage Protocol")
        pl.plot(datalog['environment.time'], datalog['membrane.V'], '-k')  
        pl.subplot(4,1,2)
        pl.title("Total current")
        pl.plot(datalog['environment.time'], tot_mem_curr_abs, '-r')          
        pl.subplot(4,1,3)
        pl.title("Current trace")
        pl.plot(datalog['environment.time'], datalog['rapid_delayed_rectifier_K_current.i_Kr'], '-b')
        pl.subplot(4,1,4)
        pl.title("Cost")
        pl.plot(datalog['environment.time'], cost, '-m')
        pl.savefig("traces/Final_Protocol.png")
        pl.show()
        #pl.close()   

    return fitness_tot

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, np.array(pop).shape[1], 3))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = np.array(pop)[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents 
    
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        rd = np.random.randint(0,offspring_size[0]-1, size=1) 
        parent1_idx = rd%np.array(parents).shape[0]
        # Index of the second parent to mate.
        rd2 = np.random.randint(0,offspring_size[0]-1, size=1) 
        parent2_idx = (rd2)%np.array(parents).shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = np.array(parents)[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = np.array(parents)[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations, lowbound, upbound):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(lowbound, upbound, 1)
            offspring_crossover[idx, gene_idx][:2] = offspring_crossover[idx, gene_idx][:2] * (1.0 - random_value)
            if offspring_crossover[idx, gene_idx][0] > 50:
                offspring_crossover[idx, gene_idx][0] = 50
            elif offspring_crossover[idx, gene_idx][0] < -120:
                offspring_crossover[idx, gene_idx][0] = -120
                #random_value = 0 #np.random.uniform(lowbound, upbound, 1)
                #offspring_crossover[idx, gene_idx][:2] = offspring_crossover[idx, gene_idx][:2] * (1.0 - random_value)
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

###################################################################################################################################
p = Pool()

print("Now running the initial selection process...")
fit_sum = []
all_ind_arr = []
for ran in range(sol_per_pop):
    #print(ran)
    showit = 0
    new_ind_list = []
    new_ind_list.append(new_population[ran])
    new_ind_list.append(showit)
    all_ind_arr.append(new_ind_list)

fit_sum = p.map(eval_fitness, all_ind_arr)
#print(fit_sum)
fit_sum = np.array(fit_sum)

new_population_select = []
fitness_select = []
for r in range(sol_per_pop):
    print(f"Selection No.{r+1}")
    lis = []
    for i in range(Tournament_size):
        r = np.random.randint(0,sol_per_pop-1, size=1) 
        if r not in lis: lis.append(r)
        else: r = np.random.randint(0,sol_per_pop-1, size=1); lis.append(r)
    print([item[0] for item in lis])

    fit_select = fit_sum[np.array(lis)]        
    max_fit = np.max(fit_select)
    fit_sum_idx = np.where(fit_sum == max_fit)   
    #print(fit_sum_idx[0])
    #print(new_population[int(fit_sum_idx[0])])
    new_population_select.append(new_population[int(fit_sum_idx[0])])
    fitness_select.append(max_fit)
  
new_population = new_population_select
fitness = fitness_select
len_pop = len(new_population)
#print(fitness)
np.save("Selected_population", new_population)
np.save("Selected_fitness", fitness)
print(f"Selection is complete. There are {len_pop} individuals in this population. Now running generations...")

new_population = np.load("Selected_population.npy") #np.load("Selected_population_Latest_GenX.npy") #np.load("Selected_population.npy")
fitness = np.load("Selected_fitness.npy") #np.load("Selected_fitness_Latest_GenX.npy") #np.load("Selected_fitness.npy")
for generation in range(0, num_generations): #range(X, num_generations): #range(num_generations):
    print("Generation " + str(generation+1))
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, num_parents_mating)
    # Generating next generation using crossover.
    pop_size = (sol_per_pop,num_par)
    offspring_crossover = crossover(parents, offspring_size=(pop_size[0]-np.array(parents).shape[0], num_par, 3))
    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover, num_par, -0.2, 0.2) #1, -0.2, 0.2) #num_par, -0.1, 0.1)
    # Creating the new population based on the parents and offspring.
    new_population = np.concatenate((parents, offspring_mutation), axis=0)
    # The best result in the current iteration.

    fitness = []
    all_ind_arr = []
    for j in range(sol_per_pop):
        #print("Post GA - Individual No. " + str(j+1))  
        showit = 0
        new_ind_list = []
        new_ind_list.append(new_population[j])
        new_ind_list.append(showit)
        all_ind_arr.append(new_ind_list)
    
    #fitness = list(map(eval_fitness, all_ind_arr))
    fitness = p.map(eval_fitness, all_ind_arr)
    
    pop_with_fitness = []    
    for j in range(sol_per_pop):
        pop_with_fitness.append([new_population[j], fitness[j]])
        
    print("Best list : " + str(fitness))
    print("Best result : " + str(np.max(fitness)))   

    best_match_idx_gen = np.where(fitness == np.max(fitness))
    np.savetxt(f"Optimized_Protocol_Gen_{generation+1}.txt", new_population[best_match_idx_gen][0])
    np.save("Selected_population_Latest_GenX", new_population)
    np.save("Selected_fitness_Latest_GenX", fitness)
     
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))

print("Best Individual: " + str(best_match_idx[0]+1))
print("Final solution : " + str(new_population[best_match_idx]))
print("Final solution fitness : " + str(np.array(fitness)[best_match_idx]))

# Save best solution
np.savetxt(f"Optimized_Protocol_{num_par}_sections.txt", new_population[best_match_idx][0])

### Reading pre-existing protocols ###
"""
optim_proto = np.loadtxt("protocol/Optimized_Protocol_4_sections_trimmedKR.txt")
"""
# Plotting final result
showit = 1
new_ind_list = []
new_ind_list.append(new_population[best_match_idx][0]) #(optim_proto)
new_ind_list.append(showit)
fitness = eval_fitness(new_ind_list)