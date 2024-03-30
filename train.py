
import torch
import time
import wandb
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine

def train_model(model, train_loader, val_loader, config_train, wandb_api_key):
    device = next(model.parameters()).device

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), config_train["lrate"], weight_decay=config_train["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_train["max_epochs"])

    # WandB initialization
    wandb.login(key=wandb_api_key)
    wandb.init(project="SegResNet-1", job_type="train", config=config_train)

    best_metric = float('inf')
    for epoch in range(config_train["max_epochs"]):
        torch.cuda.empty_cache()
        epoch_start = time.time()
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{config_train['max_epochs']}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            print(
                f"{step}/{len(train_loader)}"
                f", Train Loss: {loss.item():.4f}"
                f", Step Time: {(time.time() - step_start):.4f}"
            )

            wandb.log({"Loss": loss.item(), "Epoch": epoch})

            torch.cuda.empty_cache()
        
        lr_scheduler.step()
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

        wandb.log({"Epoch_Average_Loss": epoch_loss})

        if epoch_loss < best_metric:
            best_metric = epoch_loss
            print("Saving new best metric model...")
            torch.save(model.state_dict(), "best_metric_model.pth")

        print(f"Time consuming of Epoch {epoch + 1}: {(time.time() - epoch_start):.4f}")

    wandb.finish()
    print(f"Training completed, Best Metric: {best_metric:.4f}")
