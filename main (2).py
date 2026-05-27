import torch
import torch.optim as optim
from renderer_engine import NeuroJackSplatRenderer

def train_neurojack(img_tensor):
    # 1. Init (Këtu do të importosh NeuroForge-in tënd më vonë)
    renderer = NeuroJackSplatRenderer().cuda()
    N = 50000
    means = torch.randn((N, 3), device="cuda", requires_grad=True)
    # ... (inicializimi i pjesës tjetër të parametrave)

    optimizer = optim.Adam([means], lr=0.01)

    # 2. Loop-i i pastër
    for step in range(500):
        optimizer.zero_grad()
        # Thirrja e thjeshtuar
        render = renderer(means, colors, opacities, scales, rots, 
                          torch.eye(4).cuda(), torch.eye(4).cuda(), 
                          torch.tensor([0.,0.,-3.]).cuda())
        
        loss = torch.nn.functional.l1_loss(render[:3], img_tensor)
        loss.backward()
        optimizer.step()
        
        if step % 50 == 0:
            print(f"NeuroJackSplat | Step {step} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    print("Duke inicializuar NeuroJackSplat...")
    # train_neurojack(img_data)