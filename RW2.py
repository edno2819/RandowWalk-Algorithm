import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


MEAN_BEST = 5

class Randomwalk:

    def __init__(self, gabarito, iteracoes, population, excluse):
        self.gabarito = gabarito
        self.population = population 
        self.excluse = excluse
        self.iterations = iteracoes
        self.validadores = {'mean': [], 'std':[], 'min':[]}
        self.delta_min = min(self.gabarito)
        self.delta_max = max(self.gabarito)

    def generate_init(self):
        return [np.random.uniform(self.delta_min, self.delta_max, len(self.gabarito))]
    
    def calculate(self, x):
        return x + np.random.uniform(self.delta_min, self.delta_max)

    def calculate_dif(self, population):
        population['result'] = population['array'].apply(lambda x: sum( [abs(self.gabarito[c] - x[c]) for c in range(len(x))] ))
        return population
        
    def sum_step(self, population):
        population['array2'] = population['array'].apply(lambda x: [x[c] + np.random.uniform(-10, 10) for c in range(len(x))])
        population['result2'] = population['array2'].apply(lambda x: sum( [abs(self.gabarito[c] - x[c]) for c in range(len(x))] ))
        for i in range(len(population['array'])):
            population['array'][i] = population['array'][i] if population['result'][i]<= population['result2'][i] else population['array2'][i]

        return population.iloc[:,:2]
    
    def to_df(self, population, df=False, collum=False):
        if type(df)==bool:
            dataset = pd.DataFrame([])
        else:
            dataset = df
        for item in population:
            if not collum:
                dataset=dataset.append(item, ignore_index=True)
            else:
                dataset=dataset.append({collum:item[0]}, ignore_index=True)
        return dataset
    
    def repopule(self, population):
        delta = self.population - len(population)
        population = self.to_df([self.generate_init() for c in range(delta)], population, 'array')
        population = self.calculate_dif(population)
        population = self.sum_step(population)
        population = self.calculate_dif(population)
        population.sort_values(by=['result'], inplace=True)

        return population

    def run(self):
        population = [[self.generate_init()] for c in range(self.population)]
        population = self.to_df(population)
        population. rename(columns={0:'array'}, inplace=True)
        population = self.calculate_dif(population)
        population.sort_values(by=['result'], inplace=True)
        population = population.iloc[:self.population-self.excluse,:]


        for iteration in range(self.iterations):
            population = self.repopule(population)
            population = population.iloc[:self.population-self.excluse,:]

            self.validadores['min'].append(population['result'][:MEAN_BEST].min())
            self.validadores['mean'].append(population['result'][:MEAN_BEST].mean())
            self.validadores['std'].append(population['result'][:MEAN_BEST].std())
        
            if  population['result'][0]<100:
                population.reset_index(inplace=True)        
                return population

        population.reset_index(inplace=True)        
        return population

    def plot_conv(self):
        index=[c+1 for c in range(len(self.validadores['mean']))]
        a=[self.validadores['mean'][c] - self.validadores['std'][c] for c in range(len(index))]
        b=[self.validadores['mean'][c]+self.validadores['std'][c] for c in range(len(index))]
        sns.set_style('dark')
        plt.figure(figsize=(10,6))
        plt.grid()
        plt.plot(index, self.validadores['mean'], label='Mean')
        plt.fill_between(range(len(index)), a, b, alpha=0.5, label='Std. Dev')
        plt.plot(index, self.validadores['min'], label ='Minimum')
        plt.xlabel('Generation')
        plt.xlabel('Fitness Values')
        plt.title('Fitness by Generation')
        plt.legend()
        plt.show()


GABARITO = [52.547,72.154, 53.694, 57.771, 115.88, 105.59, 75.368, 126.02, 52.756, 85.100, 80.525, 111.24, 113.62, 64.95, 89.181, 85.647,
            101.71, 106.75, 110.37, 72.082, 104.38, 102.41, 63.009, 59.52, 89.869, 126.78, 77.231, 96.821, 67.905, 110.1]  

rw = Randomwalk(GABARITO, 50, 100, 90)
population=rw.run()
population

