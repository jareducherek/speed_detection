from .detr import build

class Args(object):
    lr=1e-4
    lr_backbone=1e-5
    batch_size=2
    weight_decay=1e-4
    epochs=300
    lr_drop=200
    clip_max_norm=0.1 #gradient clipping max norm

    # Model parameters
    frozen_weights=None
    
    # * Backbone
    backbone='resnet50' #"Name of the convolutional backbone to use")
    dilation=False #"If true, we replace stride with dilation in the last convolutional block (DC5)")
    position_embedding='sine' #choices=('sine', 'learned'), "Type of positional embedding to use on top of the image features")

    # * Transformer
    enc_layers=6
    dec_layers=6
    dim_feedforward=2048
    hidden_dim=64 #Size of the embeddings (dimension of the transformer)
    dropout=0.1
    nheads=8 #Number of attention heads inside the transformer's attentions
    num_queries=50 #Number of query slots
    pred_dim=1 #Dimension of each prediction per timestep
    pre_norm=False

    # * Segmentation
    masks=False #"Train segmentation head if the flag is provided")
    
    # Loss
    aux_loss=False #Disables auxiliary decoding losses (loss at each layer)
    
    # * Loss coefficients
    mse_loss_coef=1
    mae_loss_coef=0

    # dataset parameters
    output_dir='' #path where to save, empty for no saving
    device='cuda' #device to use for training / testing
    seed=42
    resume='' #resume from checkpoint
    start_epoch=0 #start epoch
    eval=False
    num_workers = 2

    # distributed training parameters
    world_size=1, #number of distributed processes
    dist_url='env://' #url used to set up distributed training
    
def build_args():
    return Args()

def build_model(args):
    return build(args)