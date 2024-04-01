import torch
import time
import os
import wandb
import logging
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine

def train_model(model, train_loader, val_loader, config_train, wandb_api_key, data_path):
    best_metric = 1
    best_metric_epoch = -1
    epoch_loss_values = []


    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    device = next(model.parameters()).device

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), config_train["lrate"], weight_decay=config_train["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_train["max_epochs"])

    # WandB initialization
    logging.info("Logging in WandB")
    wandb.login(key=wandb_api_key)
    wandb.init(project="SegResNet-1", job_type="train", config=config_train)
    total_start = time.time()
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
            if config_train["use_scaler"]:
                # with autocast
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # without autocast
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

            wandb.log({"Loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "Epoch": epoch})

            torch.cuda.empty_cache()
        
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")

        wandb.log({"Epoch_Average_Loss": epoch_loss})

        if epoch_loss < best_metric:
            best_metric = epoch_loss
            best_metric_epoch = epoch + 1
            best_model_file = os.path.join(data_path, "best_metric_model.pth")
            torch.save(model.state_dict(), best_model_file)
            torch.cuda.empty_cache()
            print("Saved new best metric model")


        print(f"Time consuming of Epoch {epoch + 1}: {(time.time() - epoch_start):.4f}")
    # wandb save the best model
    artifact_name = f"{wandb.run.id}_best_model"
    at = wandb.Artifact(artifact_name, type="model")
    at.add_file(best_model_file)
    wandb.log_artifact(at, aliases=[f"epoch_{epoch}"])

    total_time = time.time() - total_start
    logging.info(f"Total time: {total_time}")
    wandb.finish()
    print(f"Training completed, Best Metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
