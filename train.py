import torch
from src.model.gpt import GPT
from src.model.config import GPTConfig
from src.data.dataloader import DataLoaderLite
from src.utils.device import get_device

def train():
    device = get_device()
    print(f"using device: {device}")

    # Set seeds
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Initialize model and move to device
    model = GPT(GPTConfig())
    print("Model Architecture:")
    print(model)
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")
    
    model.to(device)

    # Initialize data loader
    train_loader = DataLoaderLite(B=4, T=32)
    num_epochs = 10  # Define number of epochs
    steps_per_epoch = len(train_loader.tokens) // (train_loader.B * train_loader.T)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for step in range(steps_per_epoch):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if step % 10 == 0:  # Print every 10 steps
                print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, Loss: {loss.item():.4f}')
        
        # Print epoch summary
        avg_loss = epoch_loss / steps_per_epoch
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Average Loss: {avg_loss:.4f}\n')

if __name__ == "__main__":
    train() 