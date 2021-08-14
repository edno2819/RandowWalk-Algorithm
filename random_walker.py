import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


MEAN_BEST = 5

class Randomwalk:

    def __init__(self, iteracoes, population, excluse):
        self.population = population 
        self.excluse = excluse
        self.iterations = iteracoes
        self.validadores = {'mean': [], 'std':[], 'min':[]}

    def generate_init(self):
        xs=np.random.uniform(0,1,3)
        return {'1':xs[0], '2':xs[1], '3':xs[2], 'round':0}
    
    def calculate(self, x):
        return x + np.random.uniform(0,1)

    def apply_function(self, population):
        x1 = population['x1']
        x2 = population['x2']
        x3 = population['x3']
        population['result']=(10*(x1-1)**2)+(20*(x2-2)**2)+(30*(x3-3)**2)
        return population
    
    def to_df(self, population, df=False):
        if type(df)==bool:
            dataset = pd.DataFrame([])
        else:
            dataset = df
        for item in population:
            dataset=dataset.append(item, ignore_index=True)
        return dataset

    def convert(self, population):
        population['x1'] = population['1'].apply(lambda x: -3 + (3 - (-3))* x)
        population['x2'] = population['2'].apply(lambda x: -2 + (4 - (-2))* x)
        population['x3'] = population['3'].apply(lambda x: -0 + (6 - (-0))* x)
        return population
    
    def sum_step(self, population):
        delta= 0.01
        population['1'] = population['1'].apply(lambda x: x+np.random.uniform(-delta, delta))
        population['2'] = population['2'].apply(lambda x: x+np.random.uniform(-delta, delta))
        population['3'] = population['3'].apply(lambda x: x+np.random.uniform(-delta, delta))
        return population

    def refresh(self, population):
        population['round']+=1
        population = self.sum_step(population)
        population = self.convert(population)
        population = self.apply_function(population)
        population.sort_values(by=['result'], inplace=True)
        return population
    
    def repopule(self, population):
        delta = self.population - len(population)
        population = self.to_df([self.generate_init() for c in range(delta)], population)
        population = self.convert(population)
        population = self.apply_function(population)
        population.sort_values(by=['result'], inplace=True)
        return population


    def run(self):
        population = [self.generate_init() for c in range(self.population)]
        population = self.to_df(population)
        population = self.convert(population)
        population = self.apply_function(population)
        population.sort_values(by=['result'], inplace=True)
        population = population.iloc[:self.population-self.excluse,:]


        for iteration in range(self.iterations):
            population = self.repopule(population)
            population = population.iloc[:self.population-self.excluse,:]
            population = self.refresh(population)

            self.validadores['min'].append(population['result'][:MEAN_BEST].min())
            self.validadores['mean'].append(population['result'][:MEAN_BEST].mean())
            self.validadores['std'].append(population['result'][:MEAN_BEST].std())
        
            if  population.iloc[0,7]<0.03:
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

    

rw = Randomwalk(50,100,90)
population=rw.run()
rw.plot_conv()

