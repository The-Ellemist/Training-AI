import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# =============================================================================
# STEP 1: PREPARE THE DATA (IMPROVED)
# =============================================================================

def prepare_data(text, sequence_length=20):  # Increased sequence length
    """
    Convert text into training examples for our language model.
    
    Improvements:
    - Use more training data (removed artificial limits)
    - Longer sequences for better context
    - Better data preprocessing
    """
    
    # Handle both string and list inputs
    if isinstance(text, list):
        text = ' '.join(text)
    
    # Clean the text a bit
    text = text.replace('\n\n', '\n').replace('  ', ' ')  # Remove excessive whitespace
    
    # Create vocabulary: all unique characters in our text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings between characters and numbers
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    
    print(f"Vocabulary: {chars[:20]}{'...' if len(chars) > 20 else ''}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Text length: {len(text)} characters")
    
    # Convert entire text to numbers
    data = [char_to_idx[ch] for ch in text]
    
    # Create training examples - USE ALL AVAILABLE DATA
    inputs = []
    targets = []
    
    # Use sliding window approach for maximum data utilization
    for i in range(int(len(data) / 50) - sequence_length):
        inputs.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    
    print(f"Created {len(inputs)} training examples")
    
    return torch.tensor(inputs), torch.tensor(targets), char_to_idx, idx_to_char, vocab_size

# =============================================================================
# STEP 2: IMPROVED MODEL
# =============================================================================

class ImprovedLanguageModel(nn.Module):
    """
    Improved language model with:
    1. Larger embedding and hidden dimensions
    2. Dropout for regularization
    3. Multiple LSTM layers
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # Use dropout before final layer
        dropped = self.dropout(lstm_out[:, -1, :])
        output = self.linear(dropped)
        return output

# =============================================================================
# STEP 3: IMPROVED TRAINING FUNCTION
# =============================================================================

def train_model(model, inputs, targets, epochs=200, learning_rate=0.002, batch_size=32):
    """
    Improved training with:
    - More epochs
    - Better learning rate
    - Learning rate scheduling
    - Better progress reporting
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    print("Starting training...")
    
    # Calculate number of batches
    num_batches = len(inputs) // batch_size
    
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress more frequently
        if epoch % 5 == 0 or patience_counter >= patience:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print("Training completed!")

# =============================================================================
# STEP 4: IMPROVED TEXT GENERATION
# =============================================================================

def generate_text(model, char_to_idx, idx_to_char, seed_text, length=200, temperature=1.0, sequence_length=20):
    """
    Improved text generation with better seed handling and sampling.
    """
    
    model.eval()
    
    # Handle case where seed_text contains characters not in vocabulary
    filtered_seed = ''.join([ch for ch in seed_text if ch in char_to_idx])
    if not filtered_seed:
        # If no valid characters, use a space or common character
        common_chars = [' ', 'e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r']
        for ch in common_chars:
            if ch in char_to_idx:
                filtered_seed = ch
                break
        else:
            filtered_seed = idx_to_char[0]
    
    # Convert seed text to numbers
    current_seq = [char_to_idx[ch] for ch in filtered_seed[-sequence_length:]]
    
    # Pad if seed is shorter than sequence_length
    while len(current_seq) < sequence_length:
        # Pad with space character if available, otherwise first character
        pad_char = char_to_idx.get(' ', 0)
        current_seq.insert(0, pad_char)
    
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
            
            # Sample next character (with some randomness even at low temperature)
            if temperature < 0.1:
                next_char_idx = torch.argmax(probabilities, dim=-1).item()
            else:
                next_char_idx = torch.multinomial(probabilities, 1).item()
        
        # Convert back to character and add to our text
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char
        
        # Update sequence for next prediction
        current_seq.append(next_char_idx)
    
    return generated_text

# =============================================================================
# STEP 5: MAIN EXECUTION WITH BETTER SAMPLE DATA
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LANGUAGE MODEL TRAINING")
    print("=" * 60)
    
    # Load training text with error handling
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            training_text = f.read()
        print(f"Loaded text from input.txt ({len(training_text)} characters)")
    except FileNotFoundError:
        print("input.txt not found, using expanded sample text instead")
        # Much more extensive sample text for better training
        training_text = """
        The art of machine learning lies in the delicate balance between complexity and simplicity. 
        When we train neural networks, we are essentially teaching machines to recognize patterns 
        in data that might be invisible to the human eye. This process requires patience, 
        computational resources, and a deep understanding of the underlying mathematics.
        
        Language models, in particular, represent one of the most fascinating applications of 
        artificial intelligence. They learn to predict the next word or character in a sequence 
        by analyzing vast amounts of text data. The transformer architecture, introduced in 
        recent years, has revolutionized this field by enabling models to understand context 
        and relationships between words across much longer distances.
        
        Training a language model begins with preprocessing the data. We convert text into 
        numerical representations that the neural network can process. Each character or word 
        becomes a token, and these tokens are mapped to high-dimensional vectors through an 
        embedding layer. These embeddings capture semantic relationships between different 
        elements of language.
        
        The learning process itself involves showing the model millions of examples of text 
        and gradually adjusting the network's parameters to minimize prediction errors. This 
        optimization process, typically using variants of gradient descent, slowly shapes the 
        model's internal representations to capture the statistical patterns of natural language.
        
        What emerges from this training process is often surprising. The model begins to 
        understand grammar, syntax, and even some aspects of meaning, despite never being 
        explicitly programmed with these rules. This emergent behavior is one of the most 
        remarkable aspects of deep learning and artificial intelligence.
        
        As we continue to scale these models and improve our training techniques, we edge 
        closer to artificial intelligence systems that can understand and generate human 
        language with unprecedented fluency and accuracy. The implications for communication, 
        education, and creativity are profound and far-reaching.
        
        However, with great power comes great responsibility. We must ensure that these 
        powerful language models are developed and deployed ethically, with careful 
        consideration of their potential impacts on society. The future of artificial 
        intelligence depends not just on technical advances, but on our wisdom in guiding 
        their development and application.
        """ * 5  # Repeat 5 times for more training data
    
    # STEP 1: Prepare the data with improved parameters
    print("\nStep 1: Preparing data...")
    sequence_length = 20  # Increased for better context
    inputs, targets, char_to_idx, idx_to_char, vocab_size = prepare_data(training_text, sequence_length)
    
    print(f"Example input sequence length: {len(inputs[0])}")
    example_input_text = ''.join([idx_to_char[i.item()] for i in inputs[0]])
    example_target_text = idx_to_char[targets[0].item()]
    print(f"Example: '{example_input_text}' -> '{example_target_text}'")
    
    # STEP 2: Create the improved model
    print("\nStep 2: Creating improved model...")
    model = ImprovedLanguageModel(vocab_size, embedding_dim=16, hidden_dim=32, num_layers=3)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # STEP 3: Train the model with better parameters
    print("\nStep 3: Training model...")
    train_model(model, inputs, targets, epochs=30, learning_rate=0.002, batch_size=32)
    
    # STEP 4: Generate some text
    print("\nStep 4: Generating text...")
    print("-" * 60)
    
    # Generate with different seeds and temperatures
    seeds = ["The art of", "Machine learning", "Language models"]
    
    for seed in seeds:
        print(f"\nSeed: '{seed}'")
        print("-" * 40)
        
        print("Conservative (temperature=0.7):")
        try:
            generated = generate_text(model, char_to_idx, idx_to_char, seed, 
                                    length=150, temperature=0.7, sequence_length=sequence_length)
            print(generated)
        except Exception as e:
            print(f"Generation failed: {e}")
        
        print("\nCreative (temperature=1.2):")
        try:
            generated = generate_text(model, char_to_idx, idx_to_char, seed, 
                                    length=150, temperature=1.2, sequence_length=sequence_length)
            print(generated)
        except Exception as e:
            print(f"Generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)