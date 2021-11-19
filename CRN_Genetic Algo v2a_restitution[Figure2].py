# -*- coding: utf-8 -*-

import matplotlib.pyplot as pl
import myokit
import numpy as np
import csv
from multiprocessing import Pool

# Callable functions
def cal_pop_fitness(varcond, bcl, Ko, DI, AF):   
    m, p, s = myokit.load("atrial model/CRN_1998_human_atrial_CM.mmt") 
    p = myokit.Protocol()
    p.schedule(1, 10, 2.0, bcl, 1) #default CRN
    p.schedule(1, DI+10, 2.0, bcl, 1)
    s = myokit.Simulation(m, p, apd_var='membrane.V')
    s.set_constant("standard_ionic_concentrations.K_o", Ko)
    s.set_constant("parameters.INa_block", varcond[0])
    s.set_constant("parameters.Ito_block", varcond[1])
    s.set_constant("parameters.ICaL_block", varcond[2])
    s.set_constant("parameters.IKur_block", varcond[3])
    s.set_constant("parameters.IKAch_block", varcond[4])
    s.set_constant("parameters.IKr_block", varcond[5])
    s.set_constant("parameters.IKs_block", varcond[6])
    s.set_constant("parameters.IK1_block", varcond[7])
    s.set_constant("parameters.INCX_block", varcond[8])
    s.set_constant("parameters.INaK_block", varcond[9])
    s.set_constant("parameters.INaL_block", varcond[10])
    s.set_constant("parameters.AF", AF)
    s.set_tolerance(1e-8, 1e-8)
    s.pre(100*bcl)    
    d = s.run(3*bcl, log_interval = 0.1)
    print("Successfully run the model")
    
    APA = np.max(d['membrane.V']) - np.min(d['membrane.V'])
    RMP = s.state()[m.get('membrane.V').indice()] 
    vt20 = 0.2*s.state()[m.get('membrane.V').indice()] 
    vt30 = 0.3*s.state()[m.get('membrane.V').indice()] 
    vt40 = 0.4*s.state()[m.get('membrane.V').indice()] 
    vt50 = 0.5*s.state()[m.get('membrane.V').indice()] 
    vt60 = 0.6*s.state()[m.get('membrane.V').indice()] 
    vt70 = 0.7*s.state()[m.get('membrane.V').indice()] 
    vt80 = 0.8*s.state()[m.get('membrane.V').indice()]   
    vt90 = -73.54545769621966 #-70.7116320292294 #0.9*s.state()[m.get('membrane.V').indice()] #np.min(d['membrane.V']) + 0.1*APA
    apd20 = d.apd(v='membrane.V', threshold = vt20)
    apd30 = d.apd(v='membrane.V', threshold = vt30)
    apd40 = d.apd(v='membrane.V', threshold = vt40)
    apd50 = d.apd(v='membrane.V', threshold = vt50)
    apd60 = d.apd(v='membrane.V', threshold = vt60)
    apd70 = d.apd(v='membrane.V', threshold = vt70)
    apd80 = d.apd(v='membrane.V', threshold = vt80)   
    apd90 = d.apd(v='membrane.V', threshold = vt90)  
    try:
        return d, APA, apd20['duration'][1], apd30['duration'][1], apd40['duration'][1], apd50['duration'][1], apd60['duration'][1], apd70['duration'][1], apd80['duration'][1], apd90['duration'][1], RMP
    except:
        return d, APA, 0, 0, 0, 0, 0, 0, 0, 9999999, RMP
    # if not apd20['duration'][1] or not apd30['duration'][1] or not apd40['duration'][1] or not apd50['duration'][1] or not apd60['duration'][1] or not apd70['duration'][1] or not apd80['duration'][1] or not apd90['duration'][1]:
    #     return d, APA, 0, 0, 0, 0, 0, 0, 0, 0, RMP
    # else:               
    #     return d, APA, apd20['duration'][1], apd30['duration'][1], apd40['duration'][1], apd50['duration'][1], apd60['duration'][1], apd70['duration'][1], apd80['duration'][1], apd90['duration'][1], RMP

def eval_fitness_PR(ind_vars):
    #var_cond_base = ind_vars[0]
    var_cond_test = ind_vars[1]
    showit = ind_vars[2]
    DI_list = [200, 300, 500, 1100, 1700, 2000] #np.arange(350, 2000.1, 300)
    #Restitution_base = [215.18, 260.25, 290.22, 295.86, 297.34, 298.80, 300.34, 301.80, 303.05, 304.11, 304.96, 305.65, 306.15, 306.61, 306.92, 307.2, 307.34]
    Restitution_base = [114.5360828260113, 142.96124245610696, 163.02886925081515, 176.5246829912594, 179.69531443851542, 180.31486784045433] #[114.05852172664291, 131.72810569182084, 140.55882606165937, 148.97343943928797, 156.32707355695794, 162.3469582107155, 166.96518869142517, 170.50827171339301, 173.3017287117383, 175.48889796007393] #[150.06325382, 169.1969435, 174.90716436, 177.68698673, 179.21990955, 180.06008028] #[215.13291733, 295.87555623, 300.35544721, 304.09742, 306.17733601, 307.17646365]
    Restitution_base = np.array(Restitution_base)
    
    with open('traces/restitution_trace_CRN_AF_6points_tp.csv', newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        resti_200 = []
        resti_300 = []
        resti_500 = []
        resti_1100 = []
        resti_1700 = []
        resti_2000 = []
        for row in reader:
            resti_200.append(float(row[0]))
            resti_300.append(float(row[1]))
            resti_500.append(float(row[2]))
            resti_1100.append(float(row[3]))
            resti_1700.append(float(row[4]))
            resti_2000.append(float(row[5]))
    
    #datalog_PR_base, APA_base, APD20_base, APD30_base, APD40_base, APD50_base, APD60_base, APD70_base, APD80_base, APD90_base, RMP_base = cal_pop_fitness(var_cond_base, 1000, 5.4)
    APV = []
    Restitution_test = []
    for DI in DI_list:
        datalog_PR, APA, APD20, APD30, APD40, APD50, APD60, APD70, APD80, APD90, RMP = cal_pop_fitness(var_cond_test, 1000, 5.4, DI, 0)
        APV.append(datalog_PR['membrane.V'])
        Restitution_test.append(APD90)
    Restitution_test = np.array(Restitution_test)
        
    if datalog_PR == [0]:
        error = 99999999
    else:
        RMSE_resti_200 = np.sqrt(np.mean((APV[0][0:4000] - np.array(resti_200[0:4000])) ** 2))
        RMSE_resti_300 = np.sqrt(np.mean((APV[1][3000:5000] - np.array(resti_300[3000:5000])) ** 2))
        RMSE_resti_500 = np.sqrt(np.mean((APV[2][5000:7000] - np.array(resti_500[5000:7000])) ** 2))
        RMSE_resti_1100 = np.sqrt(np.mean((APV[3][11000:13000] - np.array(resti_1100[11000:13000])) ** 2))
        RMSE_resti_1700 = np.sqrt(np.mean((APV[4][17000:19000] - np.array(resti_1700[17000:19000])) ** 2))
        RMSE_resti_2000 = np.sqrt(np.mean((APV[5][20000:22000] - np.array(resti_2000[20000:22000])) ** 2))
        RMSE_resti_avg = (RMSE_resti_200 + RMSE_resti_300 + RMSE_resti_500 + RMSE_resti_1100 + RMSE_resti_1700 + RMSE_resti_2000) / 6
        RMSE_resti_apd = np.sqrt(np.mean((Restitution_test - Restitution_base) ** 2))
    error = RMSE_resti_avg + RMSE_resti_apd
    print(error)

    if showit == 1:
        pl.figure("Atrial CM model 1Hz")
        pl.subplot(2,1,1)
        pl.title("AP")
        pl.plot(datalog_PR['environment.time'], np.array(resti_2000), '-k')
        pl.plot(datalog_PR['environment.time'], datalog_PR['membrane.V'], '-r')
        pl.subplot(2,1,2)
        pl.title("CaT")
        #pl.plot(datalog_PR_base['environment.time'], datalog_PR_base['intracellular_ion_concentrations.Ca_i'], '-k')
        pl.plot(datalog_PR['environment.time'], datalog_PR['intracellular_ion_concentrations.Ca_i'], '-r')
        pl.savefig(f"traces/Trace_params_Recov_sample_1Hz_no{rep+1}.png")        
        pl.show()
        """
        #fastrate
        pl.figure("Atrial CM model 3Hz")
        pl.subplot(2,1,1)
        pl.title("AP")
        pl.plot(datalog_PR_base_3Hz['environment.time'], datalog_PR_base_3Hz['membrane.V'], '-k')
        pl.plot(datalog_PR_3Hz['environment.time'], datalog_PR_3Hz['membrane.V'], '-r')
        pl.subplot(2,1,2)
        pl.title("CaT")
        pl.plot(datalog_PR_base_3Hz['environment.time'], datalog_PR_base_3Hz['intracellular_ion_concentrations.Ca_i'], '-k')
        pl.plot(datalog_PR_3Hz['environment.time'], datalog_PR_3Hz['intracellular_ion_concentrations.Ca_i'], '-r')
        pl.savefig(f"traces/Trace_params_Recov_sample_3Hz_no{rep+1}.png")        
        pl.show()
        #slowrate
        pl.figure("Atrial CM model 0.25Hz")
        pl.subplot(2,1,1)
        pl.title("AP")
        pl.plot(datalog_PR_base_0p25Hz['environment.time'], datalog_PR_base_0p25Hz['membrane.V'], '-k')
        pl.plot(datalog_PR_0p25Hz['environment.time'], datalog_PR_0p25Hz['membrane.V'], '-r')
        pl.subplot(2,1,2)
        pl.title("CaT")
        pl.plot(datalog_PR_base_0p25Hz['environment.time'], datalog_PR_base_0p25Hz['intracellular_ion_concentrations.Ca_i'], '-k')
        pl.plot(datalog_PR_0p25Hz['environment.time'], datalog_PR_0p25Hz['intracellular_ion_concentrations.Ca_i'], '-r')
        pl.savefig(f"traces/Trace_params_Recov_sample_0p25Hz_no{rep+1}.png")        
        pl.show()
        #low extracell potassium
        pl.figure("Atrial CM model 1Hz with low Ko")
        pl.subplot(2,1,1)
        pl.title("AP")
        pl.plot(datalog_PR_base_lowko['environment.time'], datalog_PR_base_lowko['membrane.V'], '-k')
        pl.plot(datalog_PR_lowko['environment.time'], datalog_PR_lowko['membrane.V'], '-r')
        pl.subplot(2,1,2)
        pl.title("CaT")
        pl.plot(datalog_PR_base_lowko['environment.time'], datalog_PR_base_lowko['intracellular_ion_concentrations.Ca_i'], '-k')
        pl.plot(datalog_PR_lowko['environment.time'], datalog_PR_lowko['intracellular_ion_concentrations.Ca_i'], '-r')
        pl.savefig(f"traces/Trace_params_Recov_sample_1Hz_lowKo_no{rep+1}.png")        
        pl.show()
        """
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
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] * random_value         
            if offspring_crossover[idx, gene_idx] > 1:
                offspring_crossover[idx, gene_idx] = 0.5
            elif offspring_crossover[idx, gene_idx] < -3.0:
                offspring_crossover[idx, gene_idx] = -1.5
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

####################################################################################################################################
# Parameter recovery with GA
####################################################################################################################################
p = Pool()
print("Now conducting parameter recovery simulations...")
var_cond_base = np.zeros(11)

# Genetic Algorithm variables
num_par_PR = len(var_cond_base)
sol_per_pop_PR = 200
Tournament_size_PR = 2
num_parents_mating_PR = int(0.1*sol_per_pop_PR)
num_generations_PR_max = 30 #50
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
            varical = np.random.uniform(-3.0, 1.0, size=1) 
            varito = np.random.uniform(-3.0, 1.0, size=1)
            varikur = np.random.uniform(-3.0, 1.0, size=1) 
            varikach = np.zeros(1) #np.random.uniform(-3.0, 1.0, size=1) 
            varikr = np.random.uniform(-3.0, 1.0, size=1) 
            variks = np.random.uniform(-3.0, 1.0, size=1) 
            varik1 = np.random.uniform(-3.0, 1.0, size=1) 
            varincx = np.random.uniform(-3.0, 1.0, size=1) 
            varinak = np.random.uniform(-3.0, 1.0, size=1) 
            varinal = np.zeros(1) #np.random.uniform(-3.0, 1.0, size=1) 
            vari_tot = [varina[0], varito[0], varical[0], varikur[0], varikach[0], varikr[0], variks[0], varik1[0], varincx[0], varinak[0], varinal[0]]
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
            #print(fit_sum_idx[0])
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
                offspring_mutation_PR = mutation_PR(offspring_crossover_PR, num_par_PR, -3.0, 1.0)
                # Creating the new population based on the parents and offspring.
                new_population_PR = np.concatenate((parents_PR, offspring_mutation_PR), axis=0)
                # The best result in the current iteration.
            
                fitness = []
                all_ind_arr = []
                for j in range(sol_per_pop_PR):
                    var_cond_test = new_population_PR[j]
                    showit = 0
                    new_ind_list = []
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
        new_ind_list.append(var_cond_base)
        new_ind_list.append(var_cond_test[0])
        new_ind_list.append(showit)
        fitness = eval_fitness_PR(new_ind_list)
        
        writer1.writerow(new_population_PR[best_match_idx][0])
