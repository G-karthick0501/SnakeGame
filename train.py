import torch
import random
import numpy as np
from collections import deque
from queue import Queue
from game_ai import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.01


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        # 11 input states -> 256 hidden states -> 3 output states
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        '''
        get_state(): to calculate the 11 input states
        In this function we will get the information needed to process the best prediction 
        for our ai, the state list is a collection of boolean values which we will use to 
        process data
        '''

        # Using the coordinates of the snake head, find the cordinates on the left, right,
        # up and down direction of the head. Take Block size as 20
        head = game.snake[0]

        # make four points with coordinates a Block size around the head block, to help
        # determine possible collisions around the head
        # Use the format Point(x-cords,y-cords) imported from game_ai.py
        point_l =
        point_r =
        point_u =
        point_d =

        # Check the direction the snake is facing, 1 if its facing that direction and 0 if its not.
        # Use game.direction to get the direction being faced by the snake, and compare it with
        # Direction class imported from game_ai.py
        dir_l =
        dir_r =
        dir_u =
        dir_d =

        # The state list containing the 11 boolean input states:
        state = [
            # 1. if Danger straight
            (),

            # 2. if Danger right
            (),

            # 3. if Danger left
            (),

            # 4. if Move direction == left
            (),

            # 5. if Move direction == right
            (),

            # 6. if Move direction == up
            (),

            # 7. if Move direction == down
            (),

            # 8. if Food location towards left of snake
            (),

            # 9. if Food location towards right of snake
            (),

            # 10. if Food location towards up of snake
            (),

            # 11. if Food location towards down of snake
            (),
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_short_memory(self, state, action, reward, next_state, done):
        # soo many params😰.. dont worry, take a look at model at model.py file
        # hint: there is function under QTrainer class with same parameters, u just need to pass these parameters to that function for this specific instance.
        pass

    def train_long_memory(self):
        '''
        train_long_memory(): called after a single game ends, it trains our model by passing the entire 
        state samples acquired so far
        '''
        # Step 1: taking random samples from self.memory if the length of self.memory is greater than the BATCH_SIZE and name it mini_sample, if not, make mini_sample equal to self.memory

        # Step 2: there multiple variables here that need to unpacked using zip(*mini_sample)
        # these variables are states, actions, rewards, next_states, dones, steps_to_food, steps_to_food_new, dangers_old, dangers_new

        # Step 3: pass these as parameters to same function u did with short memory
        pass

    def get_action(self, state, model=None):
        '''
        get_action(): to predict the 3 output boolean states by passing the 11 i/p states and calling
        the RL model on it
        In this function, we will get the action predicted by the ai for the next move of our snake in the game
        '''

        if model is None:
            model = self.model

        # random moves: tradeoff exploration / exploitation
        # set a value for epsilon dependent on the number of games to decide the randomness
        self.epsilon =

        # Initialize a final_move list containing boolean values according to move choice [straight,right,left]
        # Intially the final_move can be [0,0,0] to denote no movement
        final_move = [0, 0, 0]

        # Create an if else statement to move in a random direction dependent on epsilon, and for cases when we don't need randomness
        if random.randint(0, 200) < self.epsilon:
            # make a random move (i.e either straight,right or left)
        else:
            # create a tensor of game state, as neural networks use tensors.
            state0 = torch.tensor(state, dtype=torch.float)

            # use model.py to get the prediction with the maximum argument value and set that as the final move and return it.
            # get prediction by calling the model on state0
            prediction =

            # get the maxium argument in the prediction using torch and set it to the move variable, use argmax().item() of torch module on the prediction
            # move variable will act as the index of final_move list to set the boolean and move in the desired location
            move =
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent =                                     # Instantiate your Agent class here
    game =                                      # Instantiate your SnakeGameAI class here
    while True:
        # get old state
        # Use agent method to get the state from the game
        state_old =

        # get move
        # Use agent method to get action based on the state
        final_move =

        # perform move and get new state
        # Use game method to play a step and get results
        reward, done, score =
        # Use agent method to get the new state from the game
        state_new =

        # train short memory
        agent.train_short_memory()              # Pass the required parameters

        # remember
        agent.remember()                        # Pass the required parameters

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                print("Model.pth updated")
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
