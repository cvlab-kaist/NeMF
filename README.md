## Neural Matching Fields: Implicit Representation of Matching Fields for Visual Correspondence  (NeurIPS'22)
This is the implementation of the paper "Neural Matching Fields: Implicit Representation of Matching Fields for Visual Correspondence" by Sunghwan Hong, Jisu Nam, Seokju Cho, Susung Hong, Sangryul Jeon, Dongbo Min and Seungryong Kim. \
\
For more information, check out the paper on [[arXiv](https://arxiv.org/pdf/2210.02689.pdf)] and the [[project page](https://ku-cvlab.github.io/NeMF)]. \
Training code will be updated soon...

# Overall Architecture

Our model NeMF is illustrated below:


![alt text](/images/Overall_Architecture.png)


# Environment Settings

```
git clone https://github.com/KU-CVLAB/NeMF.git 
cd NeMF

conda create -n NeMF python=3.8
conda activate NeMF

conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -U scikit-image
pip install git+https://github.com/albumentations-team/albumentations
pip install tensorboardX termcolor timm tqdm requests pandas einops matplotlib
```

<!-- 
# Train

![alt text](/images/Train.png) -->


# Inference

![alt text](/images/Inference.png)


- Download pre-trained weights on [Link](https://drive.google.com/drive/folders/11kP1z0AmAl-Cb_MTLG7ViC3EHVoPZgHd?usp=sharing)


Result on SPair-71k :

      CUDA_VISIBLE_DEVICES=0 python test.py --pretrained ./SPAIR-NEMF --pretrained_file_name model_best.pth --benchmark spair

Result on PF-Pascal :

      CUDA_VISIBLE_DEVICES=0 python test.py --pretrained ./PF-PASCAL-NEMF --pretrained_file_name model_best.pth --benchmark pfpascal

Result on PF-Willow :

      CUDA_VISIBLE_DEVICES=0 python test.py --pretrained ./PF-PASCAL-NEMF --pretrained_file_name model_best.pth --benchmark pfwillow

# Visualization

![alt text](/images/Visualization.png)
![alt text](/images/Qual_Pascal.png)


# Acknowledgement <a name="Acknowledgement"></a>

We borrow code from public projects (huge thanks to all the projects). We mainly borrow code from  [DHPF](https://github.com/juhongm999/dhpf) and [CATs](https://github.com/SunghwanHong/Cost-Aggregation-transformers). 
### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@inproceedings{hong2022neural,
  title={Neural Matching Fields: Implicit Representation of Matching Fields for Visual Correspondence},
  author={Sunghwan Hong and Jisu Nam and Seokju Cho and Susung Hong and Sangryul Jeon and Dongbo Min and Seungryong Kim},
  year={2022}
}
````
