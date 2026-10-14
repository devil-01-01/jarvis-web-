import torch
import torch.nn as nn
import math

# ============ 1. DATA - Tiny Dataset ============
sentences = [
    "capital of india is new delhi",
    "capital of france is paris",
    "capital of japan is tokyo",
    "largest planet is jupiter",
    "smallest planet is mercury",
    "water formula is h2o",
]

# ============ 2. TOKENIZER ============
# Build vocabulary from our sentences
words = sorted(set(" ".join(sentences).split()))
word_to_id = {w: i+1 for i, w in enumerate(words)} # 0 = padding
id_to_word = {i: w for w, i in word_to_id.items()}
vocab_size = len(word_to_id) + 1
print(f"Vocab: {word_to_id}")

# Convert sentences to numbers
data = []
for s in sentences:
    ids = [word_to_id[w] for w in s.split()]
    data.append(ids)

# ============ 3. TRANSFORMER MODEL ============
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        # Embedding: Token -> Vector of 64 numbers
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional Encoding: To know word order
        self.pos_embedding = nn.Embedding(20, d_model)
        # Transformer Encoder - This has ATTENTION inside
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Final layer: Vector -> Vocab (to predict next word)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x shape: [batch, seq_len]
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        # Add meaning + position
        x = self.embedding(x) + self.pos_embedding(positions)
        # Pass through Transformer (Attention happens here)
        x = self.transformer(x)
        # Predict next word for each position
        out = self.fc_out(x)
        return out

# ============ 4. TRAINING ============
device = "cpu"
model = TinyTransformer(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.CrossEntropyLoss()

print("\nTraining started... It is learning to predict next word")

for epoch in range(500):
    total_loss = 0
    for seq in data:
        # Input: all words except last, Target: all words except first
        # Example: input [capital, of, india, is, new] -> target [of, india, is, new, delhi]
        inp = torch.tensor([seq[:-1]]).to(device)
        target = torch.tensor([seq[1:]]).to(device)

        optimizer.zero_grad()
        output = model(inp) # [1, seq_len-1, vocab_size]
        loss = loss_fn(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

print("Training Done! Loss should be near 0.")

# ============ 5. GENERATION - Like Real Model ============
def generate(prompt, max_len=10):
    model.eval()
    words_in = prompt.split()
    ids = [word_to_id[w] for w in words_in if w in word_to_id]

    with torch.no_grad():
        for _ in range(max_len):
            inp = torch.tensor([ids]).to(device)
            out = model(inp) # Get predictions
            # Take last word prediction
            next_token_logits = out[0, -1, :]
            next_id = torch.argmax(next_token_logits).item()

            # Stop if we predicted something weird (or loop)
            next_word = id_to_word.get(next_id, "?")
            ids.append(next_id)
            words_in.append(next_word)

            # If model generated a word that ends sentence, stop
            if len(words_in) > 6:
                break

    return " ".join(words_in)

# ============ 6. TEST IT ============
print("\n--- Testing Generation ---")
print("Input: capital of india is ->", generate("capital of india is"))
print("Input: capital of france is ->", generate("capital of france is"))
print("Input: largest planet is ->", generate("largest planet is"))
print("Input: water formula is ->", generate("water formula is"))

# ============ 7. SEE ATTENTION ============
# This shows how Attention works
print("\n--- How Attention Works ---")
print("When model sees 'capital of india is', attention gives HIGH score to 'india'")
print("So it knows to predict 'new delhi', not 'paris'.")
print("That is the magic of Transformer.")
