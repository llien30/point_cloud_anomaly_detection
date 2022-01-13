# TOWARD UNSUPERVISED 3D POINT CLOUD ANOMALY DETECTION USING VARIATIONAL AUTOENCODER
(accepted ICIP2021)

### Usage
Before start training, please setup earth mover's distance module (See libs/emd/README.md)

#### Make environment
```.sh
conda env create -f=emd_env.yml
```

#### Compile
```.sh
cd libs/emd
python setup.py install
```
to check the compile is success, please run
```.sh
python emd_module.py
```
after the last line of `emd_module.py`(`test_emd()`).


#### Training
```.sh
python train.py ./config/[config file name] (--no_wandb)
```



