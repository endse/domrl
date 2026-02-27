import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import os
import numpy as np

from domrl.env.user_simulator import UserDynamicsNet
from domrl.config import cfg
from domrl.utils.data_loader import load_user_sequences

class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

def collate_fn(batch):
    # Pad sequences to max length in batch
    padded = pad_sequence(batch, batch_first=True, padding_value=0) # 0 is also a category, but for padding we might need masking?
    # Our categories are 0-9. Padding with 0 adds noise if we don't mask.
    # Ideally use a dedicated PAD token, but our architecture expects 0-9.
    # Let's simple truncation or fixed length? 
    # Or just pad with 0 and rely on the model to handle it (UserDynamicsNet doesn't have mask input).
    # Correct way: Add masking support to UserDynamicsNet GRU.
    # Hack for now: Training on individual steps is safer if we don't have masking.
    # But this is a sequence model training script.
    
    # Let's just return padded and lengths
    lengths = torch.tensor([len(x) for x in batch])
    return padded, lengths

def train_simulator(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Load Data
    sequences = load_user_sequences(args.dataset_path, max_rows=args.max_rows)
    
    # Create DataLoader
    # We can simple iterate sequences and train step-by-step or use a Custom Collate
    # Given the GRU, efficient batch training requires padding + masking.
    # UserDynamicsNet is a Cell wrapper (GRUCell), not full GRU.
    # So we must loop manually over time steps.
    
    dataset = SequenceDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Initialize Model
    model = UserDynamicsNet(action_dim=cfg.NUM_CATEGORIES, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        batch_count = 0
        
        for batch_seqs, lengths in dataloader:
            batch_seqs = batch_seqs.to(device) # (B, T)
            B, T = batch_seqs.shape
            
            # Initial Hidden State
            h = torch.zeros(B, args.hidden_dim).to(device)
            persona_id = torch.zeros(B, dtype=torch.long).to(device) # Default persona
            
            loss = 0
            
            # Sequence Loop
            # Input: x_t, Target: x_{t+1}
            for t in range(T - 1):
                input_action = batch_seqs[:, t]
                target_action = batch_seqs[:, t+1]
                
                # Check for padding (naive: if target and input are 0, maybe padding? 
                # but 0 is valid category Action. Risk of training on padding.
                # Use lengths to mask loss.
                
                # Forward Transition
                # We need to predict NEXT item preferences based on CURRENT item interaction + prev state
                # Wait, UserDynamicsNet: (Action, h_prev) -> h_next, target_logits (for choice at h_next)
                # So if we feed Action_t, we get h_{t+1} and logits for Action_{t+1}.
                
                h_next, target_logits, _, _ = model.forward_transition(input_action, h, persona_id)
                
                # Loss
                step_loss = criterion(target_logits, target_action)
                
                # Masking using lengths
                mask = (t < lengths - 1).float().to(device)
                step_loss = (step_loss * mask).mean()
                
                loss += step_loss
                
                h = h_next
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
    # Save
    os.makedirs("domrl/checkpoints", exist_ok=True)
    save_path = "domrl/checkpoints/user_model_v2.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved simulator model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=cfg.MOVIE_LENS_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_rows", type=int, default=100000)
    
    args = parser.parse_args()
    train_simulator(args)
