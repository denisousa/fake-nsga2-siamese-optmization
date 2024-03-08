import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import SinglePointCrossover
from mutation import MyMutation
import time
import yaml
import random

with open('parameters.yml', 'r') as file:
    search_space = yaml.safe_load(file)

search_space_values = list(search_space.values())


params_positions = {
    "initial": [0] * 11,
    "final": [20, 10, 9, 9, 9, 9, 11, 11, 11, 11, 8] 
}

def get_all_parameters_combination(all_parameters_position):
    all_combinations = []
    for parameter_position in all_parameters_position:
        all_combinations.append(get_paramerters_by_position(parameter_position))
    
    return all_combinations

def get_paramerters_by_position(parameter_position):
    combination = [search_space['ngram'][parameter_position[0]],
    search_space['minCloneSize'][parameter_position[1]],
    search_space['QRPercentileNorm'][parameter_position[2]],
    search_space['QRPercentileT2'][parameter_position[3]],
    search_space['QRPercentileT1'][parameter_position[4]],
    search_space['QRPercentileOrig'][parameter_position[5]], 
    search_space['normBoost'][parameter_position[6]],
    search_space['t2Boost'][parameter_position[7]],
    search_space['t1Boost'][parameter_position[8]],
    search_space['origBoost'][parameter_position[9]],
    search_space['simThreshold'][parameter_position[10]]]

    return combination

def transform_parameters_list_to_dict(parameter_position):
    return {'ngramSize' : search_space['ngram'][parameter_position[0]],
    'minCloneSize' : search_space['minCloneSize'][parameter_position[1]],
    'QRPercentileNorm' : search_space['QRPercentileNorm'][parameter_position[2]],
    'QRPercentileT2' : search_space['QRPercentileT2'][parameter_position[3]],
    'QRPercentileT1' : search_space['QRPercentileT1'][parameter_position[4]],
    'QRPercentileOrig' : search_space['QRPercentileOrig'][parameter_position[5]], 
    'normBoost': search_space['normBoost'][parameter_position[6]],
    't2Boost': search_space['t2Boost'][parameter_position[7]],
    't1Boost': search_space['t1Boost'][parameter_position[8]],
    'origBoost': search_space['origBoost'][parameter_position[9]],
    'simThreshold': search_space['simThreshold'][parameter_position[10]]}

def execute_siamese(parameters_combination):
    return parameters_combination

def calculate_metrics(all_siamese_results):
    all_precision = []
    all_recall = []
    
    for result in all_siamese_results:
        if result[0] > 22 and result[1] < 8:
            all_precision.append(random.uniform(0.7, 0.97))
            all_recall.append(random.uniform(0.7, 0.97))
            continue

        all_precision.append(random.uniform(0.1, 0.8))
        all_recall.append(random.uniform(0.1, 0.8))
        
    return np.array(all_precision), np.array(all_recall)
    

# Define the optimization problem
class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=11,
                         n_obj=2,
                         n_constr=0,
                         xl=params_positions["initial"],
                         xu=params_positions["final"],
                         vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # Your tool execution logic here to calculate precision and recall
        # Replace the following lines with your tool execution and metric calculation
        
        all_parameters_combination = get_all_parameters_combination(x)
        
        all_siamese_results = []
        for combination in all_parameters_combination:
            all_siamese_results.append(execute_siamese(combination))
            
        all_precisions, all_recalls = calculate_metrics(all_siamese_results)
        
        # Maximizing precision and recall
        fitness = np.column_stack((-all_precisions, -all_recalls))

        out["F"] = fitness

# Qual método de SELEÇÃO, CROSSOVER e MUTAÇÃO usar?

# Configure the NSGA-II algorithm
algorithm = NSGA2(pop_size=10,
                  sampling=IntegerRandomSampling(),
                  # selection=TournamentSelection(),
                  crossover=SinglePointCrossover(),
                  mutation=MyMutation(),
                  eliminate_duplicates=True)

current_time = int(time.time())
termination = get_termination("time", "00:00:05")

# Run the optimizationS
problem = MyProblem()
result = minimize(problem,
               algorithm,
               termination, 
               seed=current_time, # usar o clock da máquina
               verbose=True)

# Display the best solution found
for x in result.X:
    best_solution = x
    print("Best solution found:")
    print("Hyperparameters:", transform_parameters_list_to_dict([int(x) for x in best_solution]))
    print("Hyperparameters Positions:", best_solution)
    #print("Precision:", -result.F[0, 0])  # Precision is maximized, so we invert the sign
    #print("Recall:", -result.F[0, 1])      # Recall is maximized, so we invert the sign
    print("\n")

# n_gen: Este termo se refere ao número de gerações. No contexto de algoritmos genéticos, uma "geração" se refere a uma iteração do algoritmo onde novas soluções são geradas e evoluídas.
# n_eval: n_eval indica o número de avaliações da função objetivo realizadas durante a execução do algoritmo. Em algoritmos de otimização, a função objetivo é a função que o algoritmo tenta minimizar ou maximizar, e cada chamada dessa função é contada como uma avaliação.
# n_nds: Este termo se refere ao número de soluções não dominadas encontradas no conjunto de Pareto. Soluções não dominadas são aquelas que não podem ser melhoradas em relação a uma função objetivo sem piorar em outra.
# eps: eps geralmente significa épsilon, e neste contexto pode se referir à tolerância ou precisão do algoritmo em determinar soluções ótimas. Pode ser usado para definir critérios de parada ou outros aspectos do algoritmo.
# indicator: indicator pode se referir a uma métrica de qualidade de solução ou convergência. Em algoritmos multiobjetivo, como o NSGA-II, há várias métricas que podem ser usadas para avaliar a qualidade do conjunto de soluções não dominadas, como o indicador de distância generacional (GD) ou o hipervolume (HV).


# Aonde está a última frente?

# Vamos diminuir a população

# Operador categórico o de mutação

# Função de mutação que pega um gene aleatório do individuo e muda aleatoriamente

# One point crossover
 
# Italo vai gerar a frente de pareto dos outros algoritmos

# Roda por 2H a População de 10

# Roda pelo tempo completo