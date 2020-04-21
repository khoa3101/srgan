from dataset import TrainData, ValData
from models import Generator, Discriminator
from loss import GeneratorLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils
import os
import argparse
import pytorch_ssim
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Super Resolution Model')
parser.add_argument('-pt', '--path_train', default='data/train', type=str, help='Path to training data')
parser.add_argument('-pv', '--path_val', default='data/val', type=str, help='Path to validating data')
parser.add_argument('-n', '--n_epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('-nb', '--begin_epoch', default=1, type=int, help='Which number epoch starts with')
parser.add_argument('-s', '--size', default=88, type=int, help='Cropsize of HR image')
parser.add_argument('-u', '--up', default=4, type=int, help='Upsampling factor')
parser.add_argument('-b', '--batch', default=64, type=int, help='Batch size')
parser.add_argument('-rb', '--resblock', default=5, type=int, help='Number of Residual Blocks')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate used in Adam')
parser.add_argument('-bt', '--beta', default=0.9, type=float, help='First beta used in Adam')
parser.add_argument('-m', '--mode', default='vgg19', type=str, help='Type of feature extractor in Generator Loss')
opt = parser.parse_args()

def create_folders():
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir('checkpoints/' + opt.mode):
        os.mkdir('checkpoints/' + opt.mode)
    if not os.path.isdir('scores'):
        os.mkdir('scores')
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/' + opt.mode):
        os.mkdir('results/' + opt.mode)


def train(epoch, models, train_loader, optimizers, loss_fn):
    train_bar = tqdm(train_loader)
    train_result = {'batch_size': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
    models['G'].train()
    models['D'].train()

    for lr, hr in train_bar:
        # Update batch_size
        train_result['batch_size'] += opt.batch

        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        
        with torch.enable_grad():
            # Train discriminator
            models['D'].zero_grad()

            sr = models['G'](lr)
            fake_labels = models['D'](sr).mean()
            real_labels = models['D'](hr).mean()

            d_loss = 1-real_labels + fake_labels
            d_loss.backward(retain_graph=True)
            optimizers['D'].step()

            # Train generator
            models['G'].zero_grad()

            g_loss = loss_fn(sr, hr, fake_labels)
            g_loss.backward()

            sr = models['G'](lr)
            fake_labels = models['D'](sr).mean()

            optimizers['G'].step()

        # Update loss and score
        train_result['g_loss'] += g_loss.item() * opt.batch
        train_result['d_loss'] += d_loss.item() * opt.batch
        train_result['g_score'] += fake_labels.item() * opt.batch
        train_result['d_score'] += real_labels.item() * opt.batch

        # Write results on the training bar
        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, opt.n_epochs,
            train_result['d_loss'] / train_result['batch_size'],
            train_result['g_loss'] / train_result['batch_size'],
            train_result['d_score'] / train_result['batch_size'],
            train_result['g_score'] / train_result['batch_size']
        ))

    # Take average after finish an epoch
    train_result['d_loss'] /= train_result['batch_size']
    train_result['g_loss'] /=train_result['batch_size']
    train_result['d_score'] /= train_result['batch_size']
    train_result['g_score'] /= train_result['batch_size']

    return train_result


def val(epoch, generator, val_loader):
    val_bar = tqdm(val_loader)
    val_result = {'batch_size': 0, 'mse': 0, 'psnr': 0, 'ssim': 0}
    val_images = []
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(200),
        transforms.CenterCrop(200),
        transforms.ToTensor()
    ])

    generator.eval()

    for lr, hr, hr_lr in val_bar:
        # Update batch_size
        val_result['batch_size'] += opt.batch

        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()

        with torch.no_grad():
            sr = generator(lr)

        # Update results
        batch_mse = ((sr - hr)**2).mean().cpu().numpy()
        batch_ssim = pytorch_ssim.ssim(sr, hr).item()
        val_result['mse'] += batch_mse * opt.batch
        val_result['ssim'] += batch_ssim * opt.batch
        val_result['psnr'] = 10 * np.log10(1 / (val_result['mse']/val_result['batch_size']))

        # Write result on validating bar
        val_bar.set_description(desc='[LR to SR] PSNR: %.4f dB SSIM: %.4f' % (
            val_result['psnr'],
            val_result['ssim'] / val_result['batch_size']
        ))

        # Transform images
        val_images.extend([
            trans(sr.cpu().squeeze(0)).unsqueeze(0), 
            trans(hr.cpu().squeeze(0)).unsqueeze(0), 
            trans(hr_lr.squeeze(0)).unsqueeze(0)
        ])

    # Save images
    val_images = torch.cat(val_images)
    val_images = torch.chunk(val_images, val_images.size(0) // 15)  # Display 15 images once
    save_bar = tqdm(val_images, desc='[Saving]')
    
    for idx, image in enumerate(save_bar):
        grid = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(grid, 'results/%s/%dx_%s_%s.jpg' % (opt.mode, opt.up, str(epoch).zfill(2), str(idx).zfill(2)), padding=5)

    # Take average after finish
    val_result['ssim'] /= val_result['batch_size']

    return val_result


if __name__ == '__main__':
    # Prepare data
    train_data = TrainData(path=opt.path_train, size=opt.size, up_factor=opt.up)
    val_data = ValData(path=opt.path_val, up_factor=opt.up)
    train_loader = DataLoader(train_data, batch_size=opt.batch, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=4, shuffle=False)

    # Prepare loss and models
    loss_fn = GeneratorLoss(mode=opt.mode)
    models = {'G': Generator(opt.resblock, opt.up), 'D': Discriminator()}
    if torch.cuda.is_available():
        loss_fn.cuda()
        for model in models.values():
            model.cuda()

    # Prepare optimizers
    optimizers = {
        'G': Adam(models['G'].parameters(), lr=opt.learning_rate, betas=(opt.beta, 0.999)),
        'D': Adam(models['D'].parameters(), lr=opt.learning_rate, betas=(opt.beta, 0.999))
    }

    # Create folders for storing training results
    create_folders()

    result = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(opt.begin_epoch, opt.n_epochs + opt.begin_epoch):
        print('Epoch %d' % epoch)
        train_result = train(epoch, models, train_loader, optimizers, loss_fn)
        val_result = val(epoch, models['G'], val_loader)

        # Update results
        result['d_loss'].append(train_result['d_loss'])
        result['g_loss'].append(train_result['g_loss'])
        result['d_score'].append(train_result['d_score'])
        result['g_score'].append(train_result['g_score'])
        result['psnr'].append(val_result['psnr'])
        result['ssim'].append(val_result['ssim'])

        # Save models
        torch.save(models['G'].state_dict(), 'checkpoints/%s/%dx_%s_G.pth' % (opt.mode, opt.up, str(epoch).zfill(2)))
        torch.save(models['D'].state_dict(), 'checkpoints/%s/%dx_%s_D.pth' % (opt.mode, opt.up, str(epoch).zfill(2)))

        # Save score
        if epoch % 2 == 0:
            data_frame = pd.DataFrame(
                data={
                    'Loss_D': result['d_loss'], 'Loss_G': result['g_loss'],
                    'Score_D': result['d_score'], 'Score_G': result['g_score'],
                    'PSNR': result['psnr'], 'SSIM': result['ssim']
                },
                index=range(1, epoch + 1)
            )
            data_frame.to_csv('scores/%dx_training_%s.csv' % (opt.up, opt.mode), index_label='Epoch')
