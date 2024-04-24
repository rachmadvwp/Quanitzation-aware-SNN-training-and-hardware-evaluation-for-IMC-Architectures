# SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks (Updated)
### This repository contains the Pytorch-based evaluation codes for [SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks]. https://arxiv.org/pdf/2210.12899.pdf
 
The repository consists of two hardware evaluation tools: 1) Non-ideality Computation Engine (NICE) and 2) Energy-Latency-Area (ELA) Tool. It also contains the code for quantization-aware SNN training. For reference, we have also provided a pre-trained model path for a 4-bit quantized VGG9 SNN on CIFAR10 dataset. 

## Package Installation
```shell
pip install attrs==19.3.0 beautifulsoup4==4.9.1 bottleneck brotlipy==0.7.0  certifi==2020.12.5 cffi chardet==3.0.4 cryptography==2.9.2 cycler==0.10.0 cymem cython dataclasses==0.6 eagerpy==0.29.0 einops==0.6.0 fastai==1.0.61 fastprogress==0.2.2 fonttools==4.29.1 foolbox==3.2.1 future==0.18.3 gitdb==4.0.5 gitpython==3.1.12 glib icu idna==2.10 importlib-metadata==1.7.0 intel-openmp joblib==1.1.0 jsonschema==3.2.0 kiwisolver==1.2.0 libtiff matplotlib==3.5.1 mkl mkl-service mkl_fft mkl_random murmurhash==1.0.2 ninja numexpr numpy nvidia-ml-py3==7.352.0 olefile==0.46 packaging==20.4 pandas pillow pip==20.1.1 plac==0.9.6 preshed protobuf pycparser==2.20 pyopenssl==19.1.0 pyparsing==2.4.7 pyrsistent==0.16.0 pysocks==1.7.1 python-dateutil>=2.8.2 torch 

pip install pytz==2020.1 pyyaml readline requests==2.24.0 scipy setuptools==47.3.1 sip six smmap==3.0.5 soupsieve==2.0.1 spikingjelly==0.0.0.0.12 srsly tensorboardx==2.1 threadpoolctl==3.1.0 tk torchattacks==3.0.0 torchvision tornado==6.0.4 tqdm==4.47.0 typing-extensions urllib3==1.25.9 wasabi==0.2.2 wheel==0.34.2 zipp==3.1.0 zstd==1.4.4
```

## Quantization-aware (weights only) SNN Training
```shell
cd SNN_train_infer_quantization_ela
python train.py --lr 0.001 --encode 'd' --arch 'vgg9' --T 5 --quant 4
```
## Hardware-realistic Inference using the NICE
```shell
cd NICE_Evaluation
python hw_inference.py --num_steps 5 --arch 'vgg9' --batch_size 128 --b_size 4 --ADC_precision 4 --quant 4 --xbar_size 64
```
## Hardware-realistic energy-latency-area evaluation
```shell
cd SNN_train_infer_quantization_ela
python ela_spikesim.py 
```
```
## Variable Description 
________________________________________________________________________________________
| Variable     | Type | Length            | Description                                |
|--------------|------|-------------------|--------------------------------------------|
| in_ch_list   | list | No. of SNN Layers | Layer-wise input channel count             |
| out_ch_list  | list | No. of SNN Layers | Layer-wise output channel count            |
| in_dim_list  | list | No. of SNN Layers | Layer-wise input feature size              |
| out_dim_list | list | No. of SNN Layers | Layer-wise output feature size             |
| xbar_size    | int  | -                 | Crossbar Size                              |
| kernel_size  | int  | -                 | SNN Kernel Size                            | 
| pe_per_tile  | int  | -                 | No. of Processing Engines (PE) in one tile |
| time_steps   | int  | -                 | No. of Time Steps                          |
| clk_freq     | int  | -                 | Clock Frequency in MHz                     | 
----------------------------------------------------------------------------------------
```
## Citation
Please consider citing our paper:

```
@article{moitra2022spikesim,
  title={SpikeSim: An end-to-end Compute-in-Memory Hardware Evaluation Tool for Benchmarking Spiking Neural Networks},
  author={Moitra, Abhishek and Bhattacharjee, Abhiroop and Kuang, Runcong and Krishnan, Gokul and Cao, Yu and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2210.12899},
  year={2022}
}
```
## Ackonwledgements

Code for SNN training with quantization has been adapted from https://github.com/jiecaoyu/XNOR-Net-PyTorch 
