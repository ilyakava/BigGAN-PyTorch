''' Sample
   This script loads a pretrained net and a weightsfile and sample '''
import functools
import math
import numpy as np
from tqdm import tqdm, trange

import os
from collections import defaultdict


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision
from train_fns import get_n_correct_from_D, get_error_on_dataset, get_self_error

# Import my stuff
import inception_utils
import utils
import losses


import inception as iscore
import fid

import pdb

def run(config):
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                'best_IS': 0, 'best_FID': 999999, 'config': config}
                
  # Optionally, get the configuration from the state dict. This allows for
  # recovery of the config provided only a state dict and experiment name,
  # and can be convenient for writing less verbose sample shell scripts.
  if config['config_from_name']:
    utils.load_weights(None, None, state_dict, config['weights_root'], 
                       config['experiment_name'], config['load_weights'], None,
                       strict=False, load_optim=False)
    # Ignore items which we might want to overwrite from the command line
    for item in state_dict['config']:
      if item not in ['z_var', 'base_root', 'batch_size', 'G_batch_size', 'use_ema', 'G_eval_mode']:
        config[item] = state_dict['config'][item]
  
  # update config (see train.py for explanation)
  config['resolution'] = utils.imsize_dict[config['dataset']]
  config['n_classes'] = utils.nclass_dict[config['dataset']]
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
  config = utils.update_config_roots(config)
  config['skip_init'] = True
  config['no_optim'] = True
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])
   
  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True
  
  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  
  G = model.Generator(**config).cuda()
  utils.count_parameters(G)
  
  # In some cases we need to load D
  if True or config['get_test_error'] or config['get_train_error'] or config['get_self_error']or config['get_generator_error']:
    disc_config = config.copy()
    if config['mh_csc_loss'] or config['mh_loss']:
      disc_config['output_dim'] = disc_config['n_classes'] + 1
    D = model.Discriminator(**disc_config).to(device)
    
    
  
  # Load weights
  print('Loading weights...')
  # Here is where we deal with the ema--load ema weights or load normal weights
  utils.load_weights(G if not (config['use_ema']) else None, D, state_dict, 
                     config['weights_root'], experiment_name, config['load_weights'],
                     G if config['ema'] and config['use_ema'] else None,
                     strict=False, load_optim=False)
  # Update batch size setting used for G
  G_batch_size = max(config['G_batch_size'], config['batch_size']) 
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'], 
                             z_var=config['z_var'])
  
  if config['G_eval_mode']:
    print('Putting G in eval mode..')
    G.eval()
  else:
    print('G is in %s mode...' % ('training' if G.training else 'eval'))
  
  sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)  
  brief_expt_name = config['experiment_name'][-30:]
  
  # load results dict always
  HIST_FNAME = 'scoring_hist.npy'
  def load_or_make_hist(d):
    """make/load history files in each
    """
    if not os.path.isdir(d):
      raise Exception('%s is not a valid directory' % d)
    f = os.path.join(d, HIST_FNAME)
    if os.path.isfile(f):
      return np.load(f, allow_pickle=True).item()
    else:
      return defaultdict(dict)
  hist_dir = os.path.join(config['weights_root'], config['experiment_name'])
  hist = load_or_make_hist(hist_dir)
    
  if config['get_test_error'] or config['get_train_error']:
    loaders = utils.get_data_loaders(**{**config, 'batch_size': config['batch_size'],
                                      'start_itr': state_dict['itr'], 'use_test_set': config['get_test_error']})
    acc_type = 'Test' if config['get_test_error'] else 'Train'
    
    
    accuracy = get_error_on_dataset(D, loaders[0], config, device)
    
    hist = load_or_make_hist(hist_dir)
    hist[state_dict['itr']][acc_type] = accuracy
    np.save(os.path.join(hist_dir, HIST_FNAME), hist)
    
    print('[%s][%06d] %s accuracy: %f.' % (brief_expt_name, state_dict['itr'], acc_type, accuracy * 100))

  if config['get_self_error']:
    accuracy = get_self_error(D, sample, G_batch_size, config, device)
    print('[%s][%06d] %s accuracy: %f.' % (brief_expt_name, state_dict['itr'], 'Self', accuracy * 100))
    hist = load_or_make_hist(hist_dir)
    hist[state_dict['itr']]['Self'] = accuracy
    np.save(os.path.join(hist_dir, HIST_FNAME), hist)
    
  if config['get_generator_error']:
    
    if config['dataset'] == 'C10':
      from classification.models.densenet import DenseNet121
      from torchvision import transforms
      compnet = DenseNet121()
      compnet = torch.nn.DataParallel(compnet)
      #checkpoint = torch.load(os.path.join('/scratch0/ilya/locDoc/classifiers/densenet121','ckpt_47.t7'))
      checkpoint = torch.load(os.path.join('/fs/vulcan-scratch/ilyak/locDoc/experiments/classifiers/cifar/densenet121','ckpt_47.t7'))
      compnet.load_state_dict(checkpoint['net'])
      compnet = compnet.to(device)
      compnet.eval();
      minimal_trans = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    elif config['dataset'] == 'C100':
      from classification.models.densenet import DenseNet121
      from torchvision import transforms
      compnet = DenseNet121(num_classes=100)
      compnet = torch.nn.DataParallel(compnet)
      #checkpoint = torch.load(os.path.join('/scratch0/ilya/locDoc/classifiers/cifar100/densenet121','ckpt.copy.t7'))
      checkpoint = torch.load(os.path.join('/fs/vulcan-scratch/ilyak/locDoc/experiments/classifiers/cifar100/densenet121','ckpt.copy.t7'))
      compnet.load_state_dict(checkpoint['net'])
      compnet = compnet.to(device)
      compnet.eval();
      minimal_trans = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)), ])
    elif config['dataset'] == 'STL48':
      from classification.models.wideresnet import WideResNet48
      from torchvision import transforms
      checkpoint = torch.load(os.path.join('/fs/vulcan-scratch/ilyak/locDoc/experiments/classifiers/stl/mixmatch_48','model_best.pth.tar'))
      compnet = WideResNet48(num_classes=10)
      compnet = compnet.to(device)
      for param in compnet.parameters():
          param.detach_()
      compnet.load_state_dict(checkpoint['ema_state_dict'])
      compnet.eval()
      minimal_trans = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    else:
      raise ValueError('Dataset %s has no comparison network.' % config['dataset'])
    
    n_used_imgs = 10000
    correct = 0
    mean_label = np.zeros(config['n_classes'])
    imageSize = config['resolution']
    x = np.empty((n_used_imgs,imageSize,imageSize,3), dtype=np.uint8)
    for l in  tqdm(range(n_used_imgs // G_batch_size), desc='Generating [%s][%06d]' % (brief_expt_name, state_dict['itr'])):
      with torch.no_grad():
        images, y = sample()
        fake = images.data.cpu().numpy()
        fake = np.floor((fake + 1) * 255/2.0).astype(np.uint8)
        fake_input = np.zeros(fake.shape)
        for bi in range(fake.shape[0]):
          fake_input[bi] = minimal_trans(np.moveaxis(fake[bi],0,-1))
        images.data.copy_(torch.from_numpy(fake_input));
        lab = compnet(images).max(1)[1]
        mean_label +=np.bincount(lab.data.cpu(),minlength=config['n_classes'])
        correct += int((lab == y).sum().cpu())
        
    accuracy = float(correct) / float(n_used_imgs) 
    mean_label_normalized = mean_label / float(n_used_imgs)
    
    print('[%s][%06d] %s accuracy: %f.' % (brief_expt_name, state_dict['itr'], 'Generator', accuracy * 100))
    hist = load_or_make_hist(hist_dir)
    hist[state_dict['itr']]['Generator'] = accuracy
    hist[state_dict['itr']]['Mean_Label'] = mean_label_normalized
    np.save(os.path.join(hist_dir, HIST_FNAME), hist)
    
  if config['accumulate_stats']:
    print('Accumulating standing stats across %d accumulations...' % config['num_standing_accumulations'])
    utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                    config['num_standing_accumulations'])
    
  
  # Sample a number of images and save them to an NPZ, for use with TF-Inception
  if config['sample_npz']:
    # Lists to hold images and labels for images
    x, y = [], []
    print('Sampling %d images and saving them to npz...' % config['sample_num_npz'])
    for i in trange(int(np.ceil(config['sample_num_npz'] / float(G_batch_size)))):
      with torch.no_grad():
        images, labels = sample()
      x += [np.uint8(255 * (images.cpu().numpy() + 1) / 2.)]
      y += [labels.cpu().numpy()]
    x = np.concatenate(x, 0)[:config['sample_num_npz']]
    y = np.concatenate(y, 0)[:config['sample_num_npz']]    
    print('Images shape: %s, Labels shape: %s' % (x.shape, y.shape))
    npz_filename = '%s/%s/samples.npz' % (config['samples_root'], experiment_name)
    print('Saving npz to %s...' % npz_filename)
    np.savez(npz_filename, **{'x' : x, 'y' : y})
  
  
  if config['official_FID']:
    f = np.load(config['dataset_is_fid'])
    # this is for using the downloaded one from
    # https://github.com/bioinf-jku/TTUR
    #mdata, sdata = f['mu'][:], f['sigma'][:]
    
    # this one is for my format files
    mdata, sdata = f['mfid'], f['sfid']

  # Sample a number of images and stick them in memory, for use with TF-Inception official_IS and official_FID
  data_gen_necessary = False
  if config['sample_np_mem']:
    is_saved = int('IS' in hist[state_dict['itr']])
    is_todo = int(config['official_IS'])
    fid_saved = int('FID' in hist[state_dict['itr']])
    fid_todo = int(config['official_FID'])
    data_gen_necessary = config['overwrite'] or (is_todo > is_saved) or (fid_todo > fid_saved)
  if config['sample_np_mem'] and data_gen_necessary:
    n_used_imgs = 50000
    imageSize = config['resolution']
    x = np.empty((n_used_imgs,imageSize,imageSize,3), dtype=np.uint8)
    for l in  tqdm(range(n_used_imgs // G_batch_size), desc='Generating [%s][%06d]' % (brief_expt_name, state_dict['itr'])):
      start = l * G_batch_size
      end = start + G_batch_size
      
      with torch.no_grad():
        images, labels = sample()
      fake = np.uint8(255 * (images.cpu().numpy() + 1) / 2.)
      x[start:end] = np.moveaxis(fake,1,-1)
      #y += [labels.cpu().numpy()]

  
  if config['official_IS']:
    if (not ('IS' in hist[state_dict['itr']])) or config['overwrite']:
      mis, sis = iscore.get_inception_score(x)
      print('[%s][%06d] IS mu: %f. IS sigma: %f.' % (brief_expt_name, state_dict['itr'], mis, sis))
      hist = load_or_make_hist(hist_dir)
      hist[state_dict['itr']]['IS'] = [mis, sis]
      np.save(os.path.join(hist_dir, HIST_FNAME), hist)
    else:
      mis, sis = hist[state_dict['itr']]['IS']
      print('[%s][%06d] Already done (skipping...): IS mu: %f. IS sigma: %f.' % (brief_expt_name, state_dict['itr'], mis, sis))
      
  
  if config['official_FID']:
    import tensorflow as tf
    def fid_ms_for_imgs(images, mem_fraction=0.5):
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
      inception_path = fid.check_or_download_inception(None)
      fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
      with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)
      return mu_gen, sigma_gen
    if (not ('FID' in hist[state_dict['itr']])) or config['overwrite']: 
      m1, s1 = fid_ms_for_imgs(x)
      fid_value = fid.calculate_frechet_distance(m1, s1, mdata, sdata)
      print('[%s][%06d] FID: %f' % (brief_expt_name, state_dict['itr'], fid_value))
      hist = load_or_make_hist(hist_dir)
      hist[state_dict['itr']]['FID'] = fid_value
      np.save(os.path.join(hist_dir, HIST_FNAME), hist)
    else:
      fid_value = hist[state_dict['itr']]['FID']
      print('[%s][%06d] Already done (skipping...): FID: %f' % (brief_expt_name, state_dict['itr'], fid_value))
      
        
      
  
  # Prepare sample sheets
  if config['sample_sheets']:
    print('Preparing conditional sample sheets...')
    folder_number=config['sample_sheet_folder_num']
    if folder_number == -1:
      folder_number = config['load_weights']
    utils.sample_sheet(G, classes_per_sheet=utils.classes_per_sheet_dict[config['dataset']], 
                         num_classes=config['n_classes'], 
                         samples_per_class=10, parallel=config['parallel'],
                         samples_root=config['samples_root'], 
                         experiment_name=experiment_name,
                         folder_number=folder_number,
                         z_=z_,)
  # Sample interp sheets
  if config['sample_interps']:
    print('Preparing interp sheets...')
    folder_number=config['sample_sheet_folder_num']
    if folder_number == -1:
      folder_number = config['load_weights']
    for fix_z, fix_y in zip([False, False, True], [False, True, False]):
      utils.interp_sheet(G, num_per_sheet=16, num_midpoints=8,
                         num_classes=config['n_classes'], 
                         parallel=config['parallel'], 
                         samples_root=config['samples_root'], 
                         experiment_name=experiment_name,
                         folder_number=int(folder_number), 
                         sheet_number=0,
                         fix_z=fix_z, fix_y=fix_y, device='cuda')
  # Sample random sheet
  if config['sample_random']:
    print('Preparing random sample sheet...')
    images, labels = sample()    
    torchvision.utils.save_image(images.float(),
                                 '%s/%s/%s.jpg' % (config['samples_root'], experiment_name, config['load_weights']),
                                 nrow=int(G_batch_size**0.5),
                                 normalize=True)

  
  # Prepare a simple function get metrics that we use for trunc curves
  def get_metrics():
    # Get Inception Score and FID
    get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['parallel'], config['no_fid'])
    sample = functools.partial(utils.sample, G=G, z_=z_, y_=y_, config=config)    
    IS_mean, IS_std, FID = get_inception_metrics(sample, config['num_inception_images'], num_splits=10, prints=False)
    # Prepare output string
    outstring = 'Using %s weights ' % ('ema' if config['use_ema'] else 'non-ema')
    outstring += 'in %s mode, ' % ('eval' if config['G_eval_mode'] else 'training')
    outstring += 'with noise variance %3.3f, ' % z_.var
    outstring += 'over %d images, ' % config['num_inception_images']
    if config['accumulate_stats'] or not config['G_eval_mode']:
      outstring += 'with batch size %d, ' % G_batch_size
    if config['accumulate_stats']:
      outstring += 'using %d standing stat accumulations, ' % config['num_standing_accumulations']
    outstring += 'Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f' % (state_dict['itr'], IS_mean, IS_std, FID)
    print(outstring)
  if config['sample_inception_metrics']: 
    print('Calculating Inception metrics...')
    get_metrics()
    
  # Sample truncation curve stuff. This is basically the same as the inception metrics code
  if config['sample_trunc_curves']:
    start, step, end = [float(item) for item in config['sample_trunc_curves'].split('_')]
    print('Getting truncation values for variance in range (%3.3f:%3.3f:%3.3f)...' % (start, step, end))
    for var in np.arange(start, end + step, step):     
      z_.var = var
      # Optionally comment this out if you want to run with standing stats
      # accumulated at one z variance setting
      if config['accumulate_stats']:
        utils.accumulate_standing_stats(G, z_, y_, config['n_classes'],
                                    config['num_standing_accumulations'])
      get_metrics()
  
  
  
      
def main():
  # parse command line and run    
  parser = utils.prepare_parser()
  parser = utils.add_sample_parser(parser)
  config = vars(parser.parse_args())
  print(config)
  if config['sample_multiple']:
    suffixes = config['load_weights'].split(',')
    for suffix in suffixes:
      config['load_weights'] = suffix
      run(config)
  else:
    run(config)
  
if __name__ == '__main__':    
  main()