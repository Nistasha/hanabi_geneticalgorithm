# A way to evaluate RuleAgentChromosome
# The objective of this class is that it could easily be extended 
# into a genentic algorithm engine to improve chromosomes.
# M. Fairbank. October 2021.
import sys
from hanabi_learning_environment import rl_env
from rule_agent_chromosome import RuleAgentChromosome
import os, contextlib
import platform
import random
import numpy as np


def run(environment, num_episodes, num_players, chromosome, verbose=False):
    """Run episodes."""
    game_scores = []
    for episode in range(num_episodes):
        observations = environment.reset()# This line shuffles and deals out the cards for a new game.
        agents = [RuleAgentChromosome({'players': num_players},chromosome) for _ in range(num_players)]
        done = False
        episode_reward = 0
        while not done:
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                action = agent.act(observation)
                if observation['current_player'] == agent_id:
                    assert action is not None   
                    current_player_action = action
                    if verbose:
                        print("Player",agent_id,"to play")
                        print("Player",agent_id,"View of cards",observation["observed_hands"])
                        print("Fireworks",observation["fireworks"])
                        print("Player",agent_id,"chose action",action)
                        print()
                else:
                    assert action is None
            # Make an environment step.
            observations, reward, done, unused_info = environment.step(current_player_action)
            if reward<0:
                reward=0 # we're changing the rules so that losing all lives does not result in the score being zeroed.
            episode_reward += reward
            
        if verbose:
            print("Game over.  Fireworks",observation["fireworks"],"Score=",episode_reward)
        game_scores.append(episode_reward)
    return sum(game_scores)/len(game_scores)

if __name__=="__main__":
    num_players=4
    environment=rl_env.make('Hanabi-Full', num_players=num_players)
    if platform.system()=="Windows":
        # We're on a Windows OS.
        # A temporary work-around to fix the problem that it seems on Windows, the random seed always shuffles the deck exactly the same way.
        import random
        for i in range(random.randint(1,100)):
            observations = environment.reset()
            
            
    # TODO you could potentially code a genetic algorithm in here...
    chromosome = [1, 2, 3, 0, 4, 5, 6]
    # Genetic Algorithm Parameters
    population_size = 20
    num_generations = 10
    mutation_rate = 0.05
    elite_size = int(population_size * 0.1)

    # Initialize Population
    population = []
    for i in range(population_size):
        chromosome = [random.randint(0,5) for _ in range(10)]
        print("Chromosome",chromosome)
        population.append(chromosome)
        # Run Genetic Algorithm
        for generation in range(num_generations):
            #Evaluate Fitness
            fitness_scores = []
            for chromosome in population:
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        fitness = run(environment, 25, num_players, chromosome)
                fitness_scores.append(fitness)
        # Select Elite
        elite = [population[i] for i in np.argsort(fitness_scores)[-elite_size:]]
        # Create new population
        new_population = elite.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            child = []
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            # Mutate Child
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    child[i] = random.randint(0, 6)
            new_population.append(child)
        population = new_population
    # Print final population and fitness scores
    for i in range(len(population)):
        print("chromosome:", population[i], "fitness:", fitness_scores[i])

    # chromosome=[1,2,3,0,4,5,6]
    #
    # with open(os.devnull, 'w') as devnull:
    #     with contextlib.redirect_stdout(devnull):
    #         result=run(environment,25,num_players,chromosome)
    # print("chromosome",chromosome,"fitness",result)


