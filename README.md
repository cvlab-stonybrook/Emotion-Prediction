# Using Large-scale Web Data for Visual Emotion Prediction
This repository contains the source code for [**Yevheniia Soroka's Master's Thesis**](https://drive.google.com/open?id=1cKoZaWBElCL3B3tlXT0guLtwvEjRup8y) on emotion prediction.

In this project, we:
- Explored dataset bias in existing visual emotion datasets
- Compared several techniques of combining datasets
- Learned new word embeddings trained on tags from new large-scale dataset
- Incorporated multiple datasets in a single emotion prediction system
- Achieved state-of-the-art performance on all single-label datasets
- Utilized emotional text representations for label distribution prediction


### Prerequisites
Identical conda environment can be installed using `specs.txt`. At the bare minimum, Python 3, pytorch, tqdm, PIL, numpy are needed to run the code.


### Folder Structure
`main/ckpts/` is where model checkpoints are saved by default during training.

`main/PublicEmotionDatasets/` contains folders with scripts for processing each of the existing datasets we worked with. Datasets are available for download from [1]-[3].

`main/CNNs/` has all the source code for running the models.


### Training
You must prepare your own `pkl` files containing the list of image-label pairs of the format (image_relative_path, image_label) -- separate `pkl` files for training, validation and testing. The paths to these files and other parameters must be specified in a config file, and some are included in `main/CNNs/config_files/`. `main/CNNs/mains_emotions/` contains scripts for different learning methods: single- or multi-dataset, sequential or mixed learning. 

Navigate to `main/CNNs/mains_emotions/`, choose the appropriate script and run the command:
```
python [script_name] --config_file [config_file_path]
```
e.g. for joint multi-dataset learning, run:
```
python MultiDS_Mix.py --config_file ../config_files/MultiDS/MultiDS_Mix.json
```

### Testing
For testing the trained model, simply change parameters in the corresponding config file to:
```
"resume": "[model_path_folder]/[model_file_name].pth.tar",
"evaluate": true,
...
"data_dir": "[images_folder_path]",
"test_file": "[test_file].pkl"
```
and run the same script as for training.


### Word Embedding
Pre-trained word embeddings can be dowloaded from [Google Drive](https://drive.google.com/drive/folders/1pjg5BkNPfKtzI_8cCIVxrESLVD_d6TbC?usp=sharing). Emotional embeddings trained on Adobe dataset [4] can be used as-is, and are suitable for emotion-related NLP tasks. These were used to weakly label images from Adobe dataset using scripts in `main/PublicEmotionDatasets/Adobe/process/`.


### Copyright and attribution
Created by [Yevheniia Soroka](https://www3.cs.stonybrook.edu/~ysoroka/), Research Assistant in [Computer Vision Lab](https://www3.cs.stonybrook.edu/~cvl/index.html), Stony Brook University, NY. Work conducted under the supervision of [Dimitris Samaras](https://www3.cs.stonybrook.edu/~samaras/), SUNY Empire Innovation Professor, Computer Science Department, Stony Brook University, NY.

#### Acknowledgement:
Special thanks to [Zijun Wei](http://www.zijunwei.org/) for introducing this problem to me, providing Adobe dataset, core code and initial support for this project.

#### References:
[1] Quanzeng You, Jiebo Luo, Hailin Jin and Jianchao Yang. ["Building a Large Scale Dataset for Image Emotion Recognition: The Fine Print and The Benchmark"](https://www.cs.rochester.edu/u/qyou/deepemotion/index.html), the Thirtieth AAAI Conference on Artificial Intelligence (AAAI), 2016.

[2] Jufeng Yang, Ming Sun, and Xiaoxiao Sun. [Learning visual sentiment distributions via augmented conditional probability neural network](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14506). AAAI Conference on Artificial Intelligence, 2017.

[3] Rameswar Panda, Jianming Zhang, Haoxiang Li, Joon-Young Lee, Xin Lu, and Amit K Roy-Chowdhury. [Contemplating visual emotions: Understanding and overcoming dataset bias](https://arxiv.org/abs/1808.02212). In European Conference on Computer Vision, 2018.

[4] [Adobe Research](https://research.adobe.com/)
