import argparse
import random
import sys
import os
import numpy as np
np.__config__.show()
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.w2v2_aasist import Model as w2v2_aasist
from networks.aasist import Model as aasist
from networks.beats_aasist import Model as beats_aasist 
from data_utils import genSpoof_list, ADD_Dataset, eval_to_score_file
import config
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)

def ensure_bt(x: torch.Tensor) -> torch.Tensor:
    # Ensure shape is [B, T] for AASIST
    if x.dim() == 4:      # [B, 1, 1, T]
        x = x.squeeze(1).squeeze(1)
    elif x.dim() == 3:    # [B, 1, T]
        x = x.squeeze(1)
    # if x.dim() == 2: already [B, T]
    return x  # ← ADD THIS LINE

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_x = ensure_bt(batch_x)

            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x)
            
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
        
    val_loss /= num_total
   
    return val_loss


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
    model.eval()
    with open(save_path, 'w') as fh:
        with torch.no_grad():
            for batch_x, utt_id in tqdm(data_loader, total=len(data_loader), desc='Generating scores'):
                batch_x = batch_x.to(device)
                batch_x = ensure_bt(batch_x)

                batch_out = model(batch_x)
                batch_score = batch_out[:, 0].data.cpu().numpy().ravel()  # Probability of class 0 (real)
                for f, cm in zip(utt_id, batch_score):
                    fh.write(f'{f}|{cm}\n')
    print(f'Scores saved to {save_path}')


def evaluate_eer_on_validation(model, device, args, epoch, model_save_path):
    """Evaluate EER on validation set"""
    print(f'\n===== Evaluating EER on Validation Set (Epoch {epoch}) =====')
    
    # Generate validation data list
    d_label_val, file_val = genSpoof_list(
        dir_meta=f'{config.metadata_json_file}/{args.dev_meta_json}',
        is_train=False,
        is_eval=False
    )
    
    # Create validation dataset for evaluation
    val_eval_set = ADD_Dataset(args, list_IDs=file_val, labels=d_label_val, is_eval=True)
    
    # Generate score file
    val_score_path = os.path.join(model_save_path, f'val_scores_epoch_{epoch}.txt')
    produce_evaluation_file(val_eval_set, model, device, val_score_path)
    
    # Calculate EER
    eer = eval_to_score_file(val_score_path, f'{config.metadata_json_file}/{args.dev_meta_json}')
    
    print(f'Validation EER at Epoch {epoch}: {eer:.4f}%')
    return eer


def train_epoch(train_loader, model, optimizer, device, scaler, accumulation_steps=1, grad_clip=0.0, amp_enabled=False):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss, num_total = 0.0, 0.0
    step_nums = 0

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Training'):
        batch_size = batch_x.size(0)
        num_total += batch_size

        # Move tensors to GPU (device)
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.view(-1).long().to(device, non_blocking=True)

        batch_x = ensure_bt(batch_x)

        optimizer.zero_grad(set_to_none=True)
        
        batch_x = torch.nan_to_num(batch_x, nan=0.0, posinf=1.0, neginf=-1.0)

        if amp_enabled:
            with torch.cuda.amp.autocast():
                logits = model(batch_x)
                loss = criterion(logits, batch_y) / accumulation_steps
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y) / accumulation_steps

        # Guard rails: skip bad batches
        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected. Skipping batch.")
            continue
        if not torch.isfinite(logits).all():
            print("[WARN] Non-finite logits detected. Skipping batch.")
            continue

        if amp_enabled:
            scaler.scale(loss).backward()
            if (step_nums + 1) % accumulation_steps == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (step_nums + 1) % accumulation_steps == 0:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        running_loss += loss.item() * batch_size
        step_nums += 1

    return running_loss / max(num_total, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='baseline system')
    # Dataset
    parser.add_argument('--train_meta_json', type=str, 
                        default='tta/dev/train.json', 
                        help='metadata of training data.') 
    parser.add_argument('--dev_meta_json', type=str, 
                        default='tta/dev/valid.json', 
                        help='metadata of validation data.')
    parser.add_argument('--test_meta_json', type=str, 
                        default='tta/test/test_01.json', 
                        help='metadata of test data.')
    parser.add_argument('--protocols_path', type=str, 
                        default='./', 
                        help='Change with path to user\'s database protocols directory address.')
    # Hyperparameters
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=5)  # Changed default to 5
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--num_workers', type=int, default=8)
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    parser.add_argument('--model', type=str, default='aasist',)
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--exp_id', type=str, default=0, 
                        help='Experiment id.')
    parser.add_argument('--amp', action='store_true', help='use mixed precision')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='max grad norm (0 = off)')

    
    args = parser.parse_args()
    print('====== Begin ======')
    output_folder = os.path.join(args.protocols_path, f'exps/exp_{args.exp_id}')
    os.makedirs(f'{output_folder}/ckpts', exist_ok=True)

    #make experiment reproducible
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    #define model saving path
    model_tag = '{}_{}_{}_{}_{}_{}'.format(
        args.model, args.loss, args.num_epochs, 
        args.batch_size, args.accumulation_steps, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join(f'{output_folder}/ckpts', model_tag)
    if args.eval_output is None:
        args.eval_output = os.path.join(model_save_path, 'eval_scores.txt')
    #set model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
        
    #GPU device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    amp_enabled = args.amp
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and device.type == "cuda"))
    
    if args.model == 'w2v2_aasist':
        model = w2v2_aasist(args,device)
    elif args.model == 'aasist':
        aasist_config = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
        model = aasist(aasist_config)
    elif args.model == 'beats_aasist':
        model = beats_aasist(args,device)
    else:
        print('Model not found'); sys.exit()

    model.to(device)

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('nb_params:',nb_params)
    
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad(set_to_none=True)
    
    # Learning rate scheduler — reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,     # halve LR on plateau
        patience=3,     # wait 3 epochs before reducing
        verbose=True,
        min_lr=1e-6
    )

    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    #evaluation 
    if args.eval:
        print('====== Evaluation ======')
        d_label, file_eval = genSpoof_list(
            dir_meta = f'{config.metadata_json_file}/{args.test_meta_json}',
            is_train=False,
            is_eval=False)
        print('test data path: ', args.test_meta_json)
        print('no. of test trials',len(file_eval))
        eval_set=ADD_Dataset(args, list_IDs = file_eval, labels=d_label, is_eval=True)
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        eer = eval_to_score_file(args.eval_output, f'{config.metadata_json_file}/{args.test_meta_json}' )
        sys.exit()
   
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta = f'{config.metadata_json_file}/{args.train_meta_json}',is_train=True,is_eval=False)
    print('train data path: ', args.train_meta_json)
    print('no. of training trials',len(file_train))
    
    train_set=ADD_Dataset(args,list_IDs = file_train,labels = d_label_trn)
    
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )
    
    del train_set,d_label_trn

    # define dev (validation) dataloader
    d_label_dev,file_dev = genSpoof_list( dir_meta =  f'{config.metadata_json_file}/{args.dev_meta_json}',is_train=False,is_eval=False)
    print('validation data path: ', args.dev_meta_json)
    print('no. of validation trials',len(file_dev))
    
    dev_set = ADD_Dataset(args,list_IDs = file_dev,labels = d_label_dev)

    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )

    del dev_set,d_label_dev
    
    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/exp_{}'.format(args.exp_id))
    
    print('====== Train ======')
    print(f'Training for {num_epochs} epochs\n')
    
    val_not_decrease_epochs = 0
    min_val_loss = 1e3
    best_eer = 100.0
    
    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'{"="*60}')
        
        ## early stop
        if val_not_decrease_epochs == args.patience:
            print(f'Early stopping triggered at epoch {epoch}')
            break
            
        # Training
        running_loss = train_epoch(
            train_loader, model, optimizer, device,
            scaler, accumulation_steps=args.accumulation_steps,
            grad_clip=args.grad_clip, amp_enabled=amp_enabled
        )
        
        # Validation loss
        val_loss = evaluate_accuracy(dev_loader, model, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, epoch)

        # Validation EER
        val_eer = evaluate_eer_on_validation(model, device, args, epoch, model_save_path)
        
        # Log to tensorboard
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('train_loss', running_loss, epoch)
        writer.add_scalar('val_eer', val_eer, epoch)
        
        # Print epoch summary
        print(f'\n{"="*60}')
        print(f'Epoch {epoch} Summary:')
        print(f'  Train Loss: {running_loss:.4f}')
        print(f'  Val Loss:   {val_loss:.4f}')
        print(f'  Val EER:    {val_eer:.4f}%')
        print(f'{"="*60}\n')
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
        
        # Track best model based on validation loss
        if val_loss < min_val_loss:
            val_not_decrease_epochs = 0
            min_val_loss = val_loss
            print(f'✓ New best validation loss: {min_val_loss:.4f}')
            torch.save(model.state_dict(), os.path.join(
                model_save_path, 'best_model.pth'))
        else:
            val_not_decrease_epochs += 1
            print(f'✗ Validation loss did not improve ({val_not_decrease_epochs}/{args.patience})')
        
        # Track best EER
        if val_eer < best_eer:
            best_eer = val_eer
            print(f'✓ New best EER: {best_eer:.4f}%')
            torch.save(model.state_dict(), os.path.join(
                model_save_path, 'best_eer_model.pth'))
    
    print(f'\n{"="*60}')
    print('Training Completed!')
    print(f'Best Validation Loss: {min_val_loss:.4f}')
    print(f'Best Validation EER: {best_eer:.4f}%')
    print(f'{"="*60}\n')
    
    writer.close()