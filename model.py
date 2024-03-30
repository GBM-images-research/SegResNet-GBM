import torch
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference

def get_model_and_train_components(config_train, device):
    max_epochs = config_train["max_epochs"]
    val_interval = 1
    VAL_AMP = True

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=config_train["init_filters"],
        in_channels=4,
        out_channels=3,
        dropout_prob=config_train["dropout_prob"],
    ).to(device)

    loss_function = DiceLoss(
        smooth_nr=0,
        smooth_dr=1e-5,
        squared_pred=True,
        to_onehot_y=False,
        sigmoid=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        config_train["lrate"],
        weight_decay=config_train["weight_decay"]
    )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_epochs
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose([
        Activations(sigmoid=True),
        AsDiscrete(threshold=config_train["threshold"])
    ])

    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    return model
