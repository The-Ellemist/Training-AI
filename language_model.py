import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# =============================================================================
# STEP 1: PREPARE THE DATA
# =============================================================================

def prepare_data(text, sequence_length=10):
    """
    Convert text into training examples for our language model.
    
    How it works:
    - We create a vocabulary of unique characters in our text
    - Convert each character to a number (tokenization)
    - Create input-output pairs: given a sequence, predict the next character
    
    Why we do this:
    - Neural networks work with numbers, not text
    - We need input-output pairs to train on
    """
    
    # Create vocabulary: all unique characters in our text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings between characters and numbers
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    
    print(f"Vocabulary: {chars}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Convert entire text to numbers
    data = [char_to_idx[ch] for ch in text]
    
    # Create training examples
    # If sequence_length=3 and we have "hello", we create:
    # Input: [h,e,l] -> Output: l
    # Input: [e,l,l] -> Output: o
    inputs = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        # Input: sequence of characters
        inputs.append(data[i:i + sequence_length])
        # Target: the next character
        targets.append(data[i + sequence_length])
    
    return torch.tensor(inputs), torch.tensor(targets), char_to_idx, idx_to_char, vocab_size

# =============================================================================
# STEP 2: DEFINE THE MODEL
# =============================================================================

class SimpleLanguageModel(nn.Module):
    """
    A very simple language model using:
    1. Embedding layer: converts character IDs to vectors
    2. LSTM layer: processes sequences and maintains memory
    3. Linear layer: converts LSTM output to vocabulary predictions
    
    Why this architecture:
    - Embeddings give each character a learnable representation
    - LSTM can remember patterns across the sequence
    - Linear layer maps to probabilities over our vocabulary
    """
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        
        # EMBEDDING LAYER
        # Converts each character ID (0, 1, 2, ...) into a dense vector
        # Why: gives the model a learnable way to represent each character
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM LAYER  
        # Processes sequences and maintains memory of what it has seen
        # Why: can learn patterns like "th" often followed by "e", etc.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # OUTPUT LAYER
        # Maps LSTM output to probabilities over vocabulary
        # Why: we need to predict which character comes next
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        
        # Step 1: Convert character IDs to embeddings
        # Shape: (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Step 2: Process through LSTM
        # LSTM output: (batch_size, sequence_length, hidden_dim)
        # We only care about the last output for next character prediction
        lstm_out, _ = self.lstm(embedded)
        
        # Step 3: Get prediction for next character
        # Take the last time step: lstm_out[:, -1, :]
        # Shape: (batch_size, vocab_size)
        output = self.linear(lstm_out[:, -1, :])
        
        return output

# =============================================================================
# STEP 3: TRAINING FUNCTION
# =============================================================================

def train_model(model, inputs, targets, epochs=100, learning_rate=0.001):
    """
    Train the model to predict the next character.
    
    How training works:
    1. Forward pass: model makes predictions
    2. Calculate loss: how wrong were the predictions?
    3. Backward pass: calculate gradients (which way to adjust weights)
    4. Update weights: make small improvements
    5. Repeat until model gets better
    """
    
    # OPTIMIZER: decides how to update model weights
    # Adam is a popular, robust choice that adapts learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # LOSS FUNCTION: measures how wrong our predictions are
    # CrossEntropyLoss is standard for classification (picking next character)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    
    for epoch in range(epochs):
        # FORWARD PASS: get model predictions
        predictions = model(inputs)
        
        # CALCULATE LOSS: how wrong are we?
        loss = criterion(predictions, targets)
        
        # BACKWARD PASS: calculate gradients
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Calculate new gradients
        optimizer.step()       # Update weights using gradients
        
        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    print("Training completed!")

# =============================================================================
# STEP 4: TEXT GENERATION
# =============================================================================

def generate_text(model, char_to_idx, idx_to_char, seed_text, length=100, temperature=1.0):
    """
    Generate new text using our trained model.
    
    How it works:
    1. Start with seed text
    2. Predict most likely next character
    3. Add that character to our text
    4. Use the updated text to predict the next character
    5. Repeat until we have desired length
    
    Temperature controls randomness:
    - Low (0.1): very predictable, repetitive
    - High (2.0): very random, chaotic
    - Medium (1.0): balanced
    """
    
    model.eval()  # Put model in evaluation mode
    
    # Convert seed text to numbers
    current_seq = [char_to_idx[ch] for ch in seed_text]
    generated_text = seed_text
    
    print(f"Generating text starting with: '{seed_text}'")
    
    for _ in range(length):
        # Take last sequence_length characters as input
        input_seq = torch.tensor([current_seq[-10:]]).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():  # Don't calculate gradients during generation
            output = model(input_seq)
            
            # Apply temperature to control randomness
            output = output / temperature
            
            # Convert to probabilities
            probabilities = F.softmax(output, dim=-1)
            
            # Sample next character based on probabilities
            next_char_idx = torch.multinomial(probabilities, 1).item()
        
        # Convert back to character and add to our text
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char
        
        # Update our sequence for next prediction
        current_seq.append(next_char_idx)
    
    return generated_text

# =============================================================================
# STEP 5: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Our training text - the model will learn patterns from this
    training_text = """
    Hello world! This is a simple example of training a language model.
    The model will learn to predict the next character in a sequence.
    With enough training data and time, it can generate text that looks
    similar to what it was trained on. Machine learning is amazing!
    The quick brown fox jumps over the lazy dog. This sentence contains
    every letter of the alphabet at least once.
    """ * 5  # Repeat to have more training data
    
    print("=" * 60)
    print("SIMPLE LANGUAGE MODEL TUTORIAL")
    print("=" * 60)
    
    # STEP 1: Prepare the data
    print("\nStep 1: Preparing data...")
    sequence_length = 10  # How many characters to look at to predict next one
    inputs, targets, char_to_idx, idx_to_char, vocab_size = prepare_data(training_text, sequence_length)
    
    print(f"Created {len(inputs)} training examples")
    print(f"Example input: {inputs[0]} -> target: {targets[0]}")
    
    # STEP 2: Create the model
    print("\nStep 2: Creating model...")
    model = SimpleLanguageModel(vocab_size, embedding_dim=32, hidden_dim=64)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # STEP 3: Train the model
    print("\nStep 3: Training model...")
    train_model(model, inputs, targets, epochs=50, learning_rate=0.01)
    
    # STEP 4: Generate some text
    print("\nStep 4: Generating text...")
    print("-" * 40)
    
    # Generate with different temperatures to show the effect
    seed = "Hello"
    
    print("Low temperature (predictable):")
    generated = generate_text(model, char_to_idx, idx_to_char, seed, length=100, temperature=0.5)
    print(generated)
    print()
    
    print("High temperature (creative):")
    generated = generate_text(model, char_to_idx, idx_to_char, seed, length=100, temperature=1.5)
    print(generated)
    
    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE!")
    print("=" * 60)
    print("Key takeaways:")
    print("1. We converted text to numbers (tokenization)")
    print("2. Created input-output pairs for training") 
    print("3. Used embeddings + LSTM + linear layer")
    print("4. Trained by predicting next characters")
    print("5. Generated new text by sampling predictions")
    print("=" * 60)