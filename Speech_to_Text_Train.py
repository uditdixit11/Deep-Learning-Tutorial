import pytorch_lightning as pl
from omegaconf import DictConfig
import sys
sys.path.append('.')
from nemo.collections.asr.models import EncDecCTCModel

import copy
from ruamel.yaml import YAML

def main():
    trainer = pl.Trainer(gpus=1, max_epochs=10, amp_level='O1', precision=16, val_check_interval=1)
    config_path = r'C:\Users\e5610521\Desktop\NeMo-Dev\config.yaml'
    train_manifest = r'D:\azure_data\udit\incomm_train.json'
    val_manifest = r'D:\azure_data\udit\incomm_val.json'
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = val_manifest
    asr_model = EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    # asr_model = EncDecCTCModel.load_from_checkpoint(r"C:\Users\e5602894\Desktop\NeMo-Dev\models\checkpoints\epoch=3.ckpt")
    new_opt = copy.deepcopy(params['model']['optim'])
    new_opt['lr'] = 0.001
    asr_model.setup_optimization(optim_config=DictConfig(new_opt))
    asr_model.setup_training_data(train_data_config=params['model']['train_ds'])
    asr_model.setup_validation_data(val_data_config=params['model']['validation_ds'])
    trainer.fit(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
