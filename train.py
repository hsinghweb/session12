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
    model.to(device)

    # Initialize data loader
    train_loader = DataLoaderLite(B=4, T=32)

    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(50):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f'step {i}, loss: {loss.item()}')

if __name__ == "__main__":
    train() 