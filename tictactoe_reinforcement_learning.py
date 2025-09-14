import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

class Environment:
    def __init__(self):
        self.grid = [
            0, 0, 0,
            0, 0, 0,
            0, 0, 0
        ]
        self.last_input = [0, 0]

    def check_game_end(self):
        end = 0
        winner = self.last_input[1]
        if self.grid[0] == self.grid[1] == self.grid[2] != 0:
            end = 1
        elif self.grid[3] == self.grid[4] == self.grid[5] != 0:
            end = 1
        elif self.grid[6] == self.grid[7] == self.grid[8] != 0:
            end = 1
        elif self.grid[0] == self.grid[3] == self.grid[6] != 0:
            end = 1
        elif self.grid[1] == self.grid[4] == self.grid[7] != 0:
            end = 1
        elif self.grid[2] == self.grid[5] == self.grid[8] != 0:
            end = 1
        elif self.grid[0] == self.grid[4] == self.grid[8] != 0:
            end = 1
        elif self.grid[2] == self.grid[4] == self.grid[6] != 0:
            end = 1
        
        # Check for draw (all positions filled)
        if end == 0 and 0 not in self.grid:
            end = 1
            winner = 0  # Draw
            
        return [end, winner]
    
    def get_state(self):
        game_end = self.check_game_end()
        return [self.grid.copy(), game_end]
    
    def input(self, input_pla):
        try:
            if self.grid[input_pla[0]] == 0:
                self.grid[input_pla[0]] = input_pla[1]
                self.last_input = input_pla
                return "Success"
            else:
                return "Illegal Move"
        except:
            return "Error"
    
    def reset(self):
        self.grid = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.last_input = [0, 0]
    
    def get_valid_moves(self):
        return [i for i, cell in enumerate(self.grid) if cell == 0]
    
    def print_board(self):
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        print("\n" + "-" * 13)
        for i in range(3):
            row = "| "
            for j in range(3):
                pos = i * 3 + j
                row += symbols[self.grid[pos]] + " | "
            print(row)
            print("-" * 13)

# Q-Network for Deep Q-Learning
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Input: 9 positions (grid state)
        # Output: 9 Q-values (one for each action/position)
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Q-values for each action
        return x

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, lr=0.001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_network = DQN()
        self.target_network = DQN()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        
        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 32
        self.target_update_frequency = 100
        self.step_count = 0
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state, valid_moves, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Exploration: random valid move
            return random.choice(valid_moves)
        else:
            # Exploitation: best Q-value among valid moves
            state_tensor = torch.FloatTensor(state)
            q_values = self.q_network(state_tensor)
            
            # Mask invalid moves
            masked_q_values = q_values.clone()
            for i in range(9):
                if i not in valid_moves:
                    masked_q_values[i] = -float('inf')
            
            return masked_q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the network using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calculate loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_reward(game_end, winner, player):
    """Calculate reward based on game outcome"""
    if not game_end:
        return 0  # Game ongoing
    elif winner == player:
        return 1  # Win
    elif winner == 0:
        return 0.5  # Draw
    else:
        return -1  # Loss

def train_dqn_agent():
    """Train the DQN agent using self-play"""
    agent1 = DQNAgent()
    agent2 = DQNAgent()  # Second agent for self-play
    
    num_episodes = 5000
    wins = {'agent1': 0, 'agent2': 0, 'draws': 0}
    
    print("Training DQN Agent with Self-Play...")
    print("Episodes: 5000 (this may take a few minutes)")
    
    for episode in range(num_episodes):
        env = Environment()
        current_player = 1
        
        # Store experiences for both agents
        experiences = {'agent1': [], 'agent2': []}
        
        while True:
            state = env.get_state()
            grid, (game_end, winner) = state
            
            if game_end:
                # Calculate rewards and store final experiences
                reward1 = get_reward(True, winner, 1)
                reward2 = get_reward(True, winner, 2)
                
                # Store final experiences and train
                for exp_state, action in experiences['agent1']:
                    agent1.store_experience(exp_state, action, reward1, grid, True)
                    agent1.train()
                
                for exp_state, action in experiences['agent2']:
                    agent2.store_experience(exp_state, action, reward2, grid, True)
                    agent2.train()
                
                # Track wins
                if winner == 1:
                    wins['agent1'] += 1
                elif winner == 2:
                    wins['agent2'] += 1
                else:
                    wins['draws'] += 1
                break
            
            # Get valid moves
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                break
            
            # Choose action based on current player
            if current_player == 1:
                action = agent1.get_action(grid, valid_moves)
                experiences['agent1'].append((grid.copy(), action))
            else:
                action = agent2.get_action(grid, valid_moves)
                experiences['agent2'].append((grid.copy(), action))
            
            # Make move
            prev_state = grid.copy()
            env.input([action, current_player])
            
            # Store intermediate experiences (with small negative reward for longer games)
            new_state = env.get_state()[0]
            if current_player == 1 and len(experiences['agent1']) > 1:
                prev_exp_state, prev_action = experiences['agent1'][-2]
                agent1.store_experience(prev_exp_state, prev_action, -0.01, prev_state, False)
                agent1.train()
            elif current_player == 2 and len(experiences['agent2']) > 1:
                prev_exp_state, prev_action = experiences['agent2'][-2]
                agent2.store_experience(prev_exp_state, prev_action, -0.01, prev_state, False)
                agent2.train()
            
            # Switch players
            current_player = 2 if current_player == 1 else 1
        
        # Print progress
        if (episode + 1) % 1000 == 0:
            total_games = episode + 1
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Agent1 wins: {wins['agent1']} ({wins['agent1']/total_games*100:.1f}%)")
            print(f"Agent2 wins: {wins['agent2']} ({wins['agent2']/total_games*100:.1f}%)")
            print(f"Draws: {wins['draws']} ({wins['draws']/total_games*100:.1f}%)")
            print(f"Agent1 epsilon: {agent1.epsilon:.3f}")
            print("-" * 40)
    
    print("Training complete!")
    print(f"Final results after {num_episodes} games:")
    print(f"Agent1 wins: {wins['agent1']} ({wins['agent1']/num_episodes*100:.1f}%)")
    print(f"Agent2 wins: {wins['agent2']} ({wins['agent2']/num_episodes*100:.1f}%)")
    print(f"Draws: {wins['draws']} ({wins['draws']/num_episodes*100:.1f}%)")
    
    # Return the better performing agent
    return agent1 if wins['agent1'] >= wins['agent2'] else agent2

# Function to get AI move using the DQN agent
def get_ai_move(agent, env, player):
    grid = env.get_state()[0]
    valid_moves = env.get_valid_moves()
    
    if not valid_moves:
        return None
    
    # Get action from agent (no exploration during gameplay)
    action = agent.get_action(grid, valid_moves, training=False)
    return action

# Play a game between AI and random player
def play_game_vs_random(agent, ai_player=1):
    env = Environment()
    current_player = 1
    
    print(f"\nNew Game! AI is player {ai_player}")
    env.print_board()
    
    while True:
        state = env.get_state()
        game_end, winner = state[1]
        
        if game_end:
            if winner == 0:
                print("It's a draw!")
            else:
                print(f"Player {winner} wins!")
            break
        
        if current_player == ai_player:
            # AI move
            move = get_ai_move(agent, env, current_player)
            if move is not None:
                result = env.input([move, current_player])
                print(f"AI (Player {current_player}) plays position {move}")
            else:
                print("AI couldn't find a move!")
                break
        else:
            # Random move for opponent
            valid_moves = env.get_valid_moves()
            if valid_moves:
                move = random.choice(valid_moves)
                env.input([move, current_player])
                print(f"Random Player {current_player} plays position {move}")
            else:
                print("No valid moves!")
                break
        
        env.print_board()
        
        # Switch players
        current_player = 2 if current_player == 1 else 1

# Play a game between AI and human player
def play_game_vs_human(agent):
    env = Environment()
    
    # Let user choose who goes first
    while True:
        choice = input("\nDo you want to go first? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            human_player = 1
            ai_player = 2
            break
        elif choice in ['n', 'no']:
            human_player = 2
            ai_player = 1
            break
        else:
            print("Please enter 'y' for yes or 'n' for no")
    
    current_player = 1
    print(f"\nNew Game! You are player {human_player} ({'X' if human_player == 1 else 'O'})")
    print(f"AI is player {ai_player} ({'X' if ai_player == 1 else 'O'})")
    print("\nBoard positions are numbered 0-8:")
    print("0 | 1 | 2")
    print("---------")
    print("3 | 4 | 5")
    print("---------")
    print("6 | 7 | 8")
    
    env.print_board()
    
    while True:
        state = env.get_state()
        game_end, winner = state[1]
        
        if game_end:
            if winner == 0:
                print("It's a draw!")
            elif winner == human_player:
                print("Congratulations! You won! 🎉")
            else:
                print("AI wins! Better luck next time!")
            break
        
        if current_player == ai_player:
            # AI move
            print(f"\nAI's turn...")
            move = get_ai_move(agent, env, current_player)
            if move is not None:
                result = env.input([move, current_player])
                print(f"AI plays position {move}")
            else:
                print("AI couldn't find a move!")
                break
        else:
            # Human move
            print(f"\nYour turn (Player {'X' if current_player == 1 else 'O'})!")
            valid_moves = env.get_valid_moves()
            print(f"Valid moves: {valid_moves}")
            
            while True:
                try:
                    move = int(input("Enter position (0-8): "))
                    if move in valid_moves:
                        result = env.input([move, current_player])
                        if result == "Success":
                            break
                        else:
                            print("Invalid move! Try again.")
                    else:
                        print(f"Position {move} is not available. Choose from: {valid_moves}")
                except ValueError:
                    print("Please enter a number between 0 and 8")
                except KeyboardInterrupt:
                    print("\nGame cancelled!")
                    return
        
        env.print_board()
        
        # Switch players
        current_player = 2 if current_player == 1 else 1

# Interactive menu system
def play_interactive(agent):
    print("\n" + "="*50)
    print("🎮 TIC-TAC-TOE REINFORCEMENT LEARNING AI 🎮")
    print("="*50)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Play against the AI")
        print("2. Watch AI vs Random player")
        print("3. Quit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            play_game_vs_human(agent)
            
        elif choice == '2':
            ai_player = random.choice([1, 2])
            play_game_vs_random(agent, ai_player)
            
        elif choice == '3':
            print("Thanks for playing! Goodbye! 👋")
            break
            
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")
        
        # Ask if they want to play again
        if choice in ['1', '2']:
            while True:
                again = input("\nWould you like to play another game? (y/n): ").lower().strip()
                if again in ['y', 'yes']:
                    break
                elif again in ['n', 'no']:
                    print("Thanks for playing! Goodbye! 👋")
                    return
                else:
                    print("Please enter 'y' for yes or 'n' for no")

# Main function
def main():
    print("🧠 Tic-Tac-Toe Deep Q-Learning Agent 🧠")
    print("=" * 50)
    
    # Train the DQN agent
    agent = train_dqn_agent()
    
    # Show a quick demo game
    print("\nShowing a quick demo game (AI vs Random):")
    play_game_vs_random(agent, ai_player=1)
    
    # Start interactive mode
    play_interactive(agent)

if __name__ == "__main__":
    main()
