import graph_tool as gt
import hydra
import numpy as np
import torch
import pickle
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import torch.distributed as dist
import os
from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo', rank=0, world_size=1)

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
    from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
    from analysis.visualization import NonMolecularVisualization

    datamodule = SpectreGraphDataModule(cfg)
    if dataset_config['name'] == 'sbm':
        sampling_metrics = SBMSamplingMetrics(datamodule)
    elif dataset_config['name'] == 'comm20':
        sampling_metrics = Comm20SamplingMetrics(datamodule)
    else:
        sampling_metrics = PlanarSamplingMetrics(datamodule)

    dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
    train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
    visualization_tools = NonMolecularVisualization()

    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    checkpoint_path = '/mnt/c/repo/watermark-graph-diffusion/model/sbm-v1.ckpt'
    checkpoint_path_weightonly = '/mnt/c/repo/watermark-graph-diffusion/model/sbm-v1-weightonly.ckpt'
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
                      accelerator='gpu',
                      devices=cfg.general.gpus,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      enable_progress_bar=False,
                      log_every_n_steps=50,
                      logger = [])
    model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    '''
    # trainer.save_checkpoint(checkpoint_path_weightonly, weights_only=True)
    model_kw = trainer.model.hparams
    print(model_kw)
    # save model_kw as args
    with open('/mnt/c/repo/watermark-graph-diffusion/model/sbm-v1-args.pkl', 'wb') as f:
        pickle.dump(model_kwargs, f)
    '''

def get_model_sbm():
    argpath = '/mnt/c/repo/watermark-graph-diffusion/model/sbm-v1-args.pkl'
    modelpath = '/mnt/c/repo/watermark-graph-diffusion/model/sbm-v1-weightonly.ckpt'
    args = pickle.load(open(argpath, 'rb'))
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(modelpath, **args).to('cuda')
    model.eval()
    return model


def get_model_facebook():
    argpath = '/mnt/c/repo/watermark-graph-diffusion/model/facebook.pkl'
    modelpath = '/mnt/c/repo/watermark-graph-diffusion/model/facebook-epoch=699.ckpt'
    args = pickle.load(open(argpath, 'rb'))
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(modelpath, map_location=torch.device('cuda'), **args).to('cuda')
    model.eval()

    return model

def get_model_flickr():
    argpath = '/mnt/c/repo/watermark-graph-diffusion/model/flickr.pkl'
    modelpath = '/mnt/c/repo/watermark-graph-diffusion/model/flickr-epoch=4399.ckpt'
    args = pickle.load(open(argpath, 'rb'))
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(modelpath, map_location=torch.device('cuda'), **args).to('cuda')
    model.eval()

    return model

if __name__ == '__main__':


    argpath = '/mnt/c/repo/watermark-graph-diffusion/model/facebook.pkl'
    modelpath = '/mnt/c/repo/watermark-graph-diffusion/model/facebook-epoch=699.ckpt'
    args = pickle.load(open(argpath, 'rb'))
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(modelpath, map_location=torch.device('cuda'), **args).to('cuda')
    model.eval()

    samples = model.sample_batch_simplified(1)
    print(samples[0].size(), samples[1].size())
    #model_perfs = model.sampling_metrics.test_result(samples)
    #print(model_perfs)


    #prob = model.get_node_prob()
    #print(prob)

    #arr = np.array(prob)
    #np.save('node_dist.npy', arr)


    #samples = model.sample_batch_simplified(1)
    #model_perfs = model.sampling_metrics.test_result(samples)


