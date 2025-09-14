import torch
import torch.nn as nn
import torch.optim as optim
import random

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
        return [self.grid, game_end]
    
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

# Simple Neural Network for Tic-Tac-Toe
class TicTacToeNN(nn.Module):
    def __init__(self):
        super(TicTacToeNN, self).__init__()
        # Input: 9 positions (grid state)
        # Output: 9 positions (move probabilities)
        self.fc1 = nn.Linear(9, 18)  # Hidden layer with 18 neurons
        self.fc2 = nn.Linear(18, 9)  # Output layer (9 possible moves)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = self.fc2(x)  # No activation (raw scores)
        return x

# Training function
def train_network():
    model = TicTacToeNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training the neural network...")
    
    # Simple training data: teach network to prefer center, then corners
    training_data = []
    
    # Generate some basic training examples
    for _ in range(1000):
        # Empty board - prefer center (position 4)
        empty_board = [0] * 9
        target = [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1]  # Prefer center
        training_data.append((empty_board, target))
        
        # Random board states with preferred moves
        board = [0] * 9
        num_moves = random.randint(1, 6)
        for _ in range(num_moves):
            pos = random.randint(0, 8)
            if board[pos] == 0:
                board[pos] = random.choice([1, 2])
        
        # Create target (prefer empty positions, avoid occupied ones)
        target = [0.1 if board[i] == 0 else 0.0 for i in range(9)]
        if sum(target) > 0:
            # Normalize to make it a probability distribution
            total = sum(target)
            target = [t/total for t in target]
            training_data.append((board.copy(), target))
    
    # Train the network
    for epoch in range(100):
        total_loss = 0
        for board, target in training_data:
            # Convert to tensors
            input_tensor = torch.FloatTensor(board)
            target_tensor = torch.FloatTensor(target)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    print("Training complete!")
    return model

# Function to get AI move using the neural network
def get_ai_move(model, env, player):
    # Convert board to tensor
    board_tensor = torch.FloatTensor(env.grid)
    
    # Get neural network prediction
    with torch.no_grad():
        output = model(board_tensor)
    
    # Get valid moves
    valid_moves = env.get_valid_moves()
    
    if not valid_moves:
        return None
    
    # Mask invalid moves (set their scores to very low value)
    masked_output = output.clone()
    for i in range(9):
        if i not in valid_moves:
            masked_output[i] = -999
    
    # Choose move with highest score
    best_move = torch.argmax(masked_output).item()
    return best_move

# Play a game between AI and random player
def play_game(model, ai_player=1):
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
            move = get_ai_move(model, env, current_player)
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

# Main function
def main():
    print("Simple Tic-Tac-Toe Neural Network")
    print("=" * 40)
    
    # Train the network
    model = train_network()
    
    # Show a quick demo game
    print("\nShowing a quick demo game (AI vs Random):")
    play_game_vs_random(model, ai_player=1)
    
    # Start interactive mode
    play_interactive(model)

if __name__ == "__main__":
    main()
