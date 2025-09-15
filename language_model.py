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
    
    # Handle both string and list inputs
    if isinstance(text, list):
        text = ' '.join(text)
    
    # Create vocabulary: all unique characters in our text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings between characters and numbers
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    
    print(f"Vocabulary: {chars[:20]}{'...' if len(chars) > 20 else ''}")  # Show first 20 chars
    print(f"Vocabulary size: {vocab_size}")
    print(f"Text length: {len(text)} characters")
    
    # Convert entire text to numbers
    data = [char_to_idx[ch] for ch in text]
    
    # Create training examples
    inputs = []
    targets = []
    
    # Limit the amount of training data to prevent memory issues
    max_examples = min(10000, len(data) - sequence_length)  # Limit to 10k examples max
    
    for i in range(0, max_examples, 2):  # Skip every other example to reduce memory usage
        # Input: sequence of characters
        inputs.append(data[i:i + sequence_length])
        # Target: the next character
        targets.append(data[i + sequence_length])
    
    print(f"Created {len(inputs)} training examples")
    
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
    """
    
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):  # Smaller default sizes
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.linear(lstm_out[:, -1, :])
        return output

# =============================================================================
# STEP 3: TRAINING FUNCTION
# =============================================================================

def train_model(model, inputs, targets, epochs=50, learning_rate=0.01, batch_size=32):
    """
    Train the model using mini-batches to reduce memory usage.
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training...")
    
    # Calculate number of batches
    num_batches = len(inputs) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle the data each epoch
        perm = torch.randperm(len(inputs))
        inputs_shuffled = inputs[perm]
        targets_shuffled = targets[perm]
        
        # Process data in batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = inputs_shuffled[start_idx:end_idx]
            batch_targets = targets_shuffled[start_idx:end_idx]
            
            # Forward pass
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        avg_loss = total_loss / num_batches
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Training completed!")

# =============================================================================
# STEP 4: TEXT GENERATION
# =============================================================================

def generate_text(model, char_to_idx, idx_to_char, seed_text, length=100, temperature=1.0, sequence_length=10):
    """
    Generate new text using our trained model.
    """
    
    model.eval()
    
    # Handle case where seed_text contains characters not in vocabulary
    filtered_seed = ''.join([ch for ch in seed_text if ch in char_to_idx])
    if not filtered_seed:
        # If no valid characters, use a random character from vocabulary
        filtered_seed = idx_to_char[0]
    
    # Convert seed text to numbers
    current_seq = [char_to_idx[ch] for ch in filtered_seed[-sequence_length:]]  # Take last sequence_length chars
    
    # Pad if seed is shorter than sequence_length
    while len(current_seq) < sequence_length:
        current_seq.insert(0, 0)  # Pad with first character
    
    generated_text = filtered_seed
    
    print(f"Generating text starting with: '{filtered_seed}'")
    
    for _ in range(length):
        # Take last sequence_length characters as input
        input_seq = torch.tensor([current_seq[-sequence_length:]])
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_seq)
            
            # Apply temperature
            output = output / temperature
            probabilities = F.softmax(output, dim=-1)
            
            # Sample next character
            next_char_idx = torch.multinomial(probabilities, 1).item()
        
        # Convert back to character and add to our text
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char
        
        # Update sequence for next prediction
        current_seq.append(next_char_idx)
    
    return generated_text

# =============================================================================
# STEP 5: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE LANGUAGE MODEL TUTORIAL")
    print("=" * 60)
    
    # Load training text with error handling
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            training_text = f.read()
        print(f"Loaded text from input.txt ({len(training_text)} characters)")
    except FileNotFoundError:
        print("input.txt not found, using sample text instead")
        training_text = """
        Hello world! This is a simple example of training a language model.
        The model will learn to predict the next character in a sequence.
        With enough training data and time, it can generate text that looks
        similar to what it was trained on. Machine learning is amazing!
        The quick brown fox jumps over the lazy dog. This sentence contains
        every letter of the alphabet at least once.
        """ * 10  # Repeat to have more training data
    
    # STEP 1: Prepare the data
    print("\nStep 1: Preparing data...")
    sequence_length = 30
    inputs, targets, char_to_idx, idx_to_char, vocab_size = prepare_data(training_text, sequence_length)
    
    print(f"Example input: {inputs[0]} -> target: {targets[0]}")
    
    # STEP 2: Create the model
    print("\nStep 2: Creating model...")
    model = SimpleLanguageModel(vocab_size, embedding_dim=32, hidden_dim=64)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # STEP 3: Train the model
    print("\nStep 3: Training model...")
    train_model(model, inputs, targets, epochs=100, learning_rate=0.01, batch_size=64)
    
    # STEP 4: Generate some text
    print("\nStep 4: Generating text...")
    print("-" * 40)
    
    # Generate with different temperatures
    seed = "Hello"
    
    print("Low temperature (predictable):")
    try:
        generated = generate_text(model, char_to_idx, idx_to_char, seed, length=100, temperature=0.5, sequence_length=sequence_length)
        print(generated)
    except Exception as e:
        print(f"Generation failed: {e}")
    print()
    
    print("High temperature (creative):")
    try:
        generated = generate_text(model, char_to_idx, idx_to_char, seed, length=100, temperature=1.5, sequence_length=sequence_length)
        print(generated)
    except Exception as e:
        print(f"Generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("TUTORIAL COMPLETE!")
    print("=" * 60)