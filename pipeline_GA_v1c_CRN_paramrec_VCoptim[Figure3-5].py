# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as pl
import myokit
import numpy as np
import csv
#import time
from multiprocessing import Pool

num_par = 44
hold = -80.1
t_hold = 100.0

# Callable functions
def cal_pop_fitness(Vt, varcond, AF):
    V = np.array(Vt)[:, 0]
    t = np.array(Vt)[:, 1]
    ramp = np.array(Vt)[:, 2]
    #print Vt
    #print V
    #print t
    #print ramp
    
    # m, p, s = myokit.load("atrial model/CRN_1998_human_atrial_CM_2D_tweaked.mmt")
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
    s.set_constant("parameters.INa_block", varcond[0])
    s.set_constant("parameters.Ito_block", varcond[1])
    s.set_constant("parameters.ICaL_block", varcond[2])
    s.set_constant("parameters.IKur_block", varcond[3])
    s.set_constant("parameters.IKr_block", varcond[4])
    s.set_constant("parameters.IKs_block", varcond[5])
    s.set_constant("parameters.IK1_block", varcond[6])
    s.set_constant("parameters.INCX_block", varcond[7])
    s.set_constant("parameters.INaK_block", varcond[8])
    s.set_constant("parameters.AF", AF)
    try:
        d = s.run(sum(t) + t_hold, log = ['environment.time', 'membrane.V', 'membrane.I_tot'], log_interval = 0.1)
        print("Successfully run CRN model")
    except:
        print("Crash in CRN model")
        return [0]
                         
    return d

def eval_fitness_PR(ind_vars):
    ind = ind_vars[0]
    var_target = ind_vars[1]
    var_test = ind_vars[2]
    showit = ind_vars[3]
    init = np.array([[hold, t_hold, 0]])
    VC_protocol_optim = np.concatenate((init, ind), axis=0)
    
    try:
        Imem_trace_base = np.loadtxt("Imem_traces_baseline.txt")
        datalog_PR_base = [1]
        print("Loading datalog_PR_base ...")
    except:
        datalog_PR_base = cal_pop_fitness(VC_protocol_optim, var_target, 1)
        np.savetxt("Imem_traces_baseline.txt", datalog_PR_base['membrane.I_tot'])
        Imem_trace_base = datalog_PR_base['membrane.I_tot']
        
    datalog_PR = cal_pop_fitness(VC_protocol_optim, var_test, 0)
    
    if datalog_PR_base == [0] or datalog_PR == [0]:
        error = 9999999999
        print("ERROR")
    else:
        low_range = 0#900 #5300 #5500 #12560 #2450 #2050 #1600 #7800 #1300 #this is the lower range of max contribution
        high_range = 42000 #1100 #5400 #6500 #12583 #2550 #2300 #1650 #8200 #1400 #this is the upper range of max contribution
        # RMSE_curr = np.sqrt((np.subtract(datalog_PR['membrane.I_tot'][low_range:high_range], Imem_trace_base[low_range:high_range]) ** 2).mean())
        # print("RMSE Current = " + str(RMSE_curr))
        # error = RMSE_curr
        # variance_tot = []
        # for x in range(low_range, high_range):#(len(datalog_PR['membrane.I_tot'])):
        #     variance = np.var([Imem_trace_base[x], datalog_PR['membrane.I_tot'][x]])
        #     variance_tot.append(variance)
        # error = np.mean(variance_tot) 
        # print("Mean Variance = " + str(error))
        # RMSE_peak = np.sqrt((np.subtract(np.min(datalog_PR['membrane.I_tot'][low_range:high_range]), np.min(Imem_trace_base[low_range:high_range])) ** 2).mean())
        # print("RMSE Peak = " + str(RMSE_peak))
        # error = RMSE_peak
        # RMSE_peak_na = np.sqrt((np.subtract(np.min(datalog_PR['membrane.I_tot'][1300:1400]), np.min(Imem_trace_base[1300:1400])) ** 2).mean())
        # RMSE_peak_to = np.sqrt((np.subtract(np.max(datalog_PR['membrane.I_tot'][9300:9500]), np.max(Imem_trace_base[9300:9500])) ** 2).mean())
        # RMSE_peak_cal = np.sqrt((np.subtract(np.min(datalog_PR['membrane.I_tot'][11340:11360]), np.min(Imem_trace_base[11340:11360])) ** 2).mean())
        # RMSE_peak_kur = np.sqrt((np.subtract(np.max(datalog_PR['membrane.I_tot'][13600:13750]), np.max(Imem_trace_base[13600:13750])) ** 2).mean())
        # RMSE_peak_kr = np.sqrt((np.subtract(np.max(datalog_PR['membrane.I_tot'][16180:16190]), np.max(Imem_trace_base[16180:16190])) ** 2).mean())
        # RMSE_peak_ks = np.sqrt((np.subtract(np.max(datalog_PR['membrane.I_tot'][28820:28860]), np.max(Imem_trace_base[28820:28860])) ** 2).mean())
        # RMSE_peak_k1 = np.sqrt((np.subtract(np.max(datalog_PR['membrane.I_tot'][35500:35600]), np.max(Imem_trace_base[35500:35600])) ** 2).mean())
        # RMSE_peak_ncx = np.sqrt((np.subtract(np.max(datalog_PR['membrane.I_tot'][40860:40880]), np.max(Imem_trace_base[40860:40880])) ** 2).mean())
        # RMSE_peak_nak = np.sqrt((np.subtract(np.min(datalog_PR['membrane.I_tot'][41860:41880]), np.min(Imem_trace_base[41860:41880])) ** 2).mean())
        # RMSE_peak_tot = RMSE_peak_na + RMSE_peak_to + RMSE_peak_cal + RMSE_peak_kur + RMSE_peak_kr + RMSE_peak_ks + RMSE_peak_k1 + RMSE_peak_ncx + RMSE_peak_nak
        # print("RMSE Peak Total = " + str(RMSE_peak_tot))
        # error = RMSE_peak_tot
        
        #### CRN AF setting ####
        RMSE_current_na = 4 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][1300:1400]), (Imem_trace_base[1300:1400])) ** 2).mean())
        RMSE_current_to = 3 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][9300:9500]), (Imem_trace_base[9300:9500])) ** 2).mean())
        RMSE_current_cal = 20 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][11340:11360]), (Imem_trace_base[11340:11360])) ** 2).mean())
        RMSE_current_kur = 20 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][13600:13750]), (Imem_trace_base[13600:13750])) ** 2).mean())
        RMSE_current_kr = 50 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][16180:16190]), (Imem_trace_base[16180:16190])) ** 2).mean())
        RMSE_current_ks = 6 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][28820:28860]), (Imem_trace_base[28820:28860])) ** 2).mean())
        RMSE_current_k1 = 19 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][35500:35600]), (Imem_trace_base[35500:35600])) ** 2).mean())
        RMSE_current_ncx = 50 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][40860:40880]), (Imem_trace_base[40860:40880])) ** 2).mean())
        RMSE_current_nak = 5 * np.sqrt((np.subtract((datalog_PR['membrane.I_tot'][41860:41880]), (Imem_trace_base[41860:41880])) ** 2).mean())
        RMSE_current_tot = RMSE_current_na + RMSE_current_to + RMSE_current_cal + RMSE_current_kur + RMSE_current_kr + RMSE_current_ks + RMSE_current_k1 + RMSE_current_ncx + RMSE_current_nak
        print("RMSE Na = " + str(RMSE_current_na))
        print("RMSE To = " + str(RMSE_current_to))
        print("RMSE CaL = " + str(RMSE_current_cal))
        print("RMSE Kur = " + str(RMSE_current_kur))
        print("RMSE Kr = " + str(RMSE_current_kr))
        print("RMSE Ks = " + str(RMSE_current_ks))
        print("RMSE K1 = " + str(RMSE_current_k1))
        print("RMSE NCX = " + str(RMSE_current_ncx))
        print("RMSE NaK = " + str(RMSE_current_nak))
        print("RMSE current Avg = " + str(RMSE_current_tot / 9))
        error = RMSE_current_tot / 9       
        
    if showit == 1:
        pl.figure("PostOptim_ParamRec")
        pl.subplot(3,1,1)
        pl.title("Voltage Protocol")
        #pl.plot(datalog_PR_base['environment.time'], datalog_PR_base['membrane.V'], '-k')
        pl.plot(datalog_PR['environment.time'], datalog_PR['membrane.V'], '-r')
        pl.subplot(3,1,2)
        pl.title("total current")
        pl.plot(datalog_PR['environment.time'], Imem_trace_base, '-k')
        pl.plot(datalog_PR['environment.time'], datalog_PR['membrane.I_tot'], '-r')   
        pl.subplot(3,1,3)
        pl.title("total current (error calculation)")
        pl.plot(datalog_PR['environment.time'][low_range:high_range], Imem_trace_base[low_range:high_range], '-k')
        pl.plot(datalog_PR['environment.time'][low_range:high_range], datalog_PR['membrane.I_tot'][low_range:high_range], '-r')         
        pl.savefig(f"traces/Trace_params_Recov_sample_no{rep}.png")        
        pl.show()
        
        test = new_population_PR[best_match_idx][0]
        
        pl.figure("Params_comparison")
        for i in range(0, num_par_PR):
            pl.axvspan(i+0.8, i+1.2, facecolor='b', alpha=0.1)
            pl.plot(i+1.05, test[i], "xr") 
            pl.ylim(-3.0, 1.0)
            pl.axhline(y=0.0, color='m', linestyle='--')
            pl.ylabel("Current block")                 
        pl.savefig(f"traces/Summary_params_Recov_sample_no{rep+1}.png")                    
        pl.show()       
    
        np.savetxt("datalog_CRN_finalGen_known.txt", np.c_[datalog_PR['environment.time'], datalog_PR['membrane.V'], Imem_trace_base], delimiter = ",")
        np.savetxt("datalog_CRN_finalGen_test.txt", np.c_[datalog_PR['environment.time'], datalog_PR['membrane.V'], datalog_PR['membrane.I_tot']], delimiter = ",")
    
    return error

def select_mating_pool_PR(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, np.array(pop).shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.min(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = np.array(pop)[max_fitness_idx, :]
        fitness[max_fitness_idx] = 99999999999
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

def mutation_PR(offspring_crossover, num_mutations, lowbound, upbound):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(lowbound, upbound, 1)    
            #print("Randval = " + str(random_value))
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] * random_value         
            if offspring_crossover[idx, gene_idx] > 1:
                offspring_crossover[idx, gene_idx] = 0.5
            elif offspring_crossover[idx, gene_idx] < -3.0:
                offspring_crossover[idx, gene_idx] = -1.5
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover
###################################################################################################################################
p = Pool()
####################################################################################################################################
#2) Parameter recovery with GA
####################################################################################################################################
best_sol = np.loadtxt("protocol/Optimized_Protocol_4_sections_trimmed_combiCRN.txt") #("protocol/Optimized_Protocol_4_sections_trimmedNAK.txt") #("protocol/Optimized_Protocol_4_sections_trimmed_combiCRN.txt") 
best_sol = [best_sol] 
print("From now on, we conduct parameter recovery simulations...")
var_cond_base = np.zeros(9)
#var_cond_base = [0, 0, 0, -0.06588453438403, 0.36, -0.86737903024905, 0.005748388179414, 0.030366462054599, -0.128588625268082] #low IKr
#var_cond_base = [0.014638594902597, -0.052987122128379, -0.052099145465729, 0.039060035555485, -0.47, 0.672683146866318, -0.029576900768351, -0.053066736271271, 0.079212124988695] #high IKr
#var_cond_base = [0, 0.1, 0.4, 0.630331412762434, 0, 0.1, 0.102115709983835, 0.090393466582999, -0.036186389878852] #low ICaL
#var_cond_base = [0.023402738083823, -0.212161208213756, -1.0, -1.02838662933105, 0, -0.704732847998764, -0.005092837974636, 0.127467688339633, -0.127332484508427] #high ICaL
#var_cond_base = np.array(var_cond_base)

# Genetic Algorithm variables
num_par_PR = len(var_cond_base)
sol_per_pop_PR = 200
Tournament_size_PR = 2
num_parents_mating_PR = int(0.1*sol_per_pop_PR)
num_generations_PR_max = 150
pop_size = (sol_per_pop_PR, num_par_PR)
samples = 1

with open("Predicted_scalingfact.csv", "w", newline='') as file1:
    writer1 = csv.writer(file1, delimiter=',')
    for rep in range(samples):
        print(f"Now running sample No.{rep+1} out of {samples}")
        
        #new_population_PR = np.random.uniform(-0.5, 0.5, size=pop_size) 
        new_population_PR = []
        for q in range(sol_per_pop_PR):
            varina = np.random.uniform(-3.0, 1.0, size=1) 
            varical =np.random.uniform(-3.0, 1.0, size=1) 
            varito = np.random.uniform(-3.0, 1.0, size=1)
            varikur = np.random.uniform(-3.0, 1.0, size=1) 
            varikr = np.random.uniform(-3.0, 1.0, size=1) 
            variks = np.random.uniform(-3.0, 1.0, size=1) 
            varik1 = np.random.uniform(-3.0, 1.0, size=1) 
            varincx = np.random.uniform(-3.0, 1.0, size=1) 
            varinak = np.random.uniform(-3.0, 1.0, size=1) 
            vari_tot = [varina[0], varito[0], varical[0], varikur[0], varikr[0], variks[0], varik1[0], varincx[0], varinak[0]]
            new_population_PR.append(vari_tot)
        new_population_PR = np.array(new_population_PR)
        np.savetxt(f"PR_initial_population_sample_no{rep+1}.txt", new_population_PR, delimiter=',')        
        print("Now running the initial selection process for parameters recovery...")
        
        fit_sum = []
        all_ind_arr = []
        for ran in range(sol_per_pop_PR):
            #print(ran[0])
            var_cond_test = new_population_PR[ran]
            showit = 0
            new_ind_list = []
            new_ind_list.append(best_sol[0])
            new_ind_list.append(var_cond_base)
            new_ind_list.append(var_cond_test)
            new_ind_list.append(showit)
            all_ind_arr.append(new_ind_list)
        
        #fit_sum = list(map(eval_fitness_PR, all_ind_arr))
        fit_sum = p.map(eval_fitness_PR, all_ind_arr)
        fit_sum = np.array(fit_sum)
        
        new_population_select_PR = []
        fitness_select_PR = []
        for r in range(sol_per_pop_PR):
            print(f"PR selection No.{r+1}")
            lis = []
            for i in range(Tournament_size_PR):
                r = np.random.randint(0,sol_per_pop_PR-1, size=1) 
                if r not in lis: lis.append(r)
                else: r = np.random.randint(0,sol_per_pop_PR-1, size=1); lis.append(r)
            print([item[0] for item in lis])
        
            fit_select = fit_sum[np.array(lis)]            
            min_fit = np.min(fit_select)
            fit_sum_idx = np.where(fit_sum == min_fit)   
            #print(fit_sum_idx[0][0])
            #print(new_population_PR[int(fit_sum_idx[0])])
            new_population_select_PR.append(new_population_PR[int(fit_sum_idx[0][0])])
            fitness_select_PR.append(min_fit)
            
        new_population_PR = new_population_select_PR
        fitness = fitness_select_PR
        len_pop = len(new_population_PR)
        #print(fitness)
        np.save("Selected_paramrec_population", new_population_PR)
        np.save("Selected_paramrec_fitness", fitness)        
        print(f"PR selection is complete. There are {len_pop} individuals in this population. Now running generations...")
        
        new_population_PR = np.load("Selected_paramrec_population.npy") #np.load("PR_population_Latest_Gen.npy") #np.load("Selected_paramrec_population.npy")
        fitness = np.load("Selected_paramrec_fitness.npy") #np.load("PR_fitness_Latest_Gen.npy") #np.load("Selected_paramrec_fitness.npy")
        for generation in range(0, num_generations_PR_max): #range(50, 100): #range(25, num_generations_PR_max): #range(num_generations_PR_max):
            min_fitness = np.min(fitness)
            print("Initial best fitness in Generation: " + str(min_fitness))
            if min_fitness > 0: #0.00001:
                print("Generation " + str(generation+1))
                # Selecting the best parents in the population for mating.
                parents_PR = select_mating_pool_PR(new_population_PR, fitness, num_parents_mating_PR)
                # Generating next generation using crossover.
                offspring_crossover_PR = crossover(parents_PR, offspring_size=(pop_size[0]-np.array(parents_PR).shape[0], num_par_PR))
                # Adding some variations to the offsrping using mutation.
                offspring_mutation_PR = mutation_PR(offspring_crossover_PR, num_par_PR, -2.0, 2.0)
                # Creating the new population based on the parents and offspring.
                new_population_PR = np.concatenate((parents_PR, offspring_mutation_PR), axis=0)
                # The best result in the current iteration.
            
                fitness = []
                all_ind_arr = []
                for j in range(sol_per_pop_PR):
                    var_cond_test = new_population_PR[j]
                    showit = 0
                    new_ind_list = []
                    new_ind_list.append(best_sol[0])
                    new_ind_list.append(var_cond_base)
                    new_ind_list.append(var_cond_test)
                    new_ind_list.append(showit)
                    all_ind_arr.append(new_ind_list)
                
                fitness = p.map(eval_fitness_PR, all_ind_arr)
                
                pop_with_fitness = []            
                for j in range(sol_per_pop_PR):
                    pop_with_fitness.append([new_population_PR[j], fitness[j]])
        
                np.save("PR_population_Latest_Gen", new_population_PR)
                np.save("PR_fitness_Latest_Gen", fitness)                   
                #np.savetxt("PR_population_Latest_Gen.txt", new_population_PR, delimiter=',')        
                #np.savetxt("PR_fitness_Latest_Gen.txt", fitness, delimiter=',')
                np.savetxt(f"PR_population_Sample_{rep+1}_Gen_{generation+1}.txt", new_population_PR, delimiter=',')        
                np.savetxt(f"PR_fitness_Sample_{rep+1}_Gen_{generation+1}.txt", fitness, delimiter=',')
                print("Best list : " + str(fitness))
                print("Best result : " + str(np.min(fitness)))        
                print("Average Fitness : " + str(np.mean(fitness)))
                sorted_fitness_idx = np.argsort(fitness)
                fit_arr = np.array(fitness)
                sorted_fitness = fit_arr[sorted_fitness_idx]
                sorted_pop = new_population_PR[sorted_fitness_idx]
                bestten_fitness = sorted_fitness[:10] #sorted_fitness[-10 : ]
                bestten_pop = sorted_pop[:10] #sorted_pop[-10 : ]
                print("Average of 10 best fitness : " + str(np.mean(bestten_fitness)))
        
                ##Printing Generational Population
                test = new_population_PR
                for img in range(len(new_population_PR)):
                    pl.figure("Params_comparison")
                    for i in range(0, num_par_PR):
                        pl.axvspan(i+0.8, i+1.2, facecolor='b', alpha=0.1)
                        pl.plot(i+1.05, test[img][i], "xr") 
                        pl.axhline(y=0.0, color='m', linestyle='--')
                        pl.ylim(-3.0, 1.0)
                        pl.ylabel("Current block")                 
                pl.savefig(f"traces/Summary_params_Recov_sample_no{rep+1}_Gen{generation+1}.png")                    
                pl.show()
                pl.close()
        
            else:
                print("Stopping the loop, fitness threshold achieved...")
                break
        
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.where(fitness == np.min(fitness))
        
        print("Best Individual: " + str(best_match_idx[0]+1))
        print("Final solution : " + str(new_population_PR[best_match_idx]))
        print("Final solution fitness : " + str(np.array(fitness)[best_match_idx]))
        
        # Plotting final result
        showit = 1
        var_cond_test = new_population_PR[best_match_idx]
        new_ind_list = []
        new_ind_list.append(best_sol[0])
        new_ind_list.append(var_cond_base)
        new_ind_list.append(var_cond_test[0])
        new_ind_list.append(showit)
        fitness = eval_fitness_PR(new_ind_list)
        
        writer1.writerow(new_population_PR[best_match_idx][0])
