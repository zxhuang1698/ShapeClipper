# ShapeClipper

[[Project Page]](https://zixuanh.com/projects/shapeclipper.html)  [[Paper]](https://zixuanh.com/projects/cvpr2023-shapeclipper/paper.pdf) [[Video]](https://www.youtube.com/watch?v=BxTGVjXoXu8&t=9s)

![](teaser.gif)

The repository currently includes training and evaluation code for Pix3D experiments.

## Dependencies

If you are using CUDA 10, please install the dependencies by running
```bash
conda env create --file requirements.yaml
pip install git+https://github.com/openai/CLIP.git
export CUDA_HOME=/usr/local/cuda-10.2 (change this to your cuda path)
cd external/chamfer3D && python3 setup.py install
cd ../..
```

If you are using newer CUDA versions, then mostly the complication will not work and you will need to install the dependencies manually. For example, you can try the following commands:
```bash
conda create -n shapeclipper -c conda-forge -c pytorch tqdm termcolor pyyaml pip matplotlib scikit-learn seaborn trimesh vigra tensorboard python=3 pytorch::pytorch=1.12 pytorch::torchvision=0.13 cudatoolkit=11.3 (change this to your version)
pip install pymcubes
pip install git+https://github.com/openai/CLIP.git
export CUDA_HOME=/usr/local/cuda-11.3 (change this to your cuda path)
cd external/chamfer3D && python3 setup.py install
cd ../..
```

## Dataset

### Pix3D

Please download our processed Pix3D from this [URL](https://www.dropbox.com/s/6pkehu0xrh3s7q6/Pix3D.tar?dl=0) and extract it under the `data` folder.
Note this processed data already includes the pre-computed CLIP nearest neighbor annotations. If you wish to re-compute it by yourself, please run:
```bash
python CLIP_anno.py --yaml=options/clip/pix3d.yaml
```
The new CLIP annotations will overwrite the old ones at `data/Pix3D/CLIP_NN` by default.

## Training

Please first pretrain the model so it reconstructs a sphere (for stable initialization) if you want to train from scratch:
```bash
python pretrain.py --yaml=options/pix3d/config.yaml --pretrain
```
The pretrained weights will be saved at `output/pix3d_output/pretrain/latest.ckpt` by default.
Then you can run the following command to train the reconstruction model:

```bash
python train.py --yaml=options/pix3d/config.yaml --name=pix3d_exp --load=output/pix3d_output/pretrain/latest.ckpt
```
The training visualizations and checkpoints will be saved at the output directory.

## Evaluating

To evaluate the model with Chamfer Distance and F-score, Please run

```bash
python evaluate.py --yaml=options/pix3d/config.yaml --name=pix3d_exp --resume --eval.vox_res=100
```

The evaluation results will be printed and saved at the output directory.

## References

If you find our work helpful, please consider citing our paper.
```
@inproceedings{huang2023shapeclipper,
  author    = {Huang, Zixuan and Jampani, Varun and Thai, Anh and Li, Yuanzhen and Stojanov, Stefan and Rehg, James M},
  title     = {ShapeClipper: Scalable 3D Shape Learning from Single-View Images via Geometric and CLIP-based Consistency},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2023},
}
```

This codebase borrows code from [SDF-SRN](https://github.com/chenhsuanlin/signed-distance-SRN) ([MIT License](https://github.com/chenhsuanlin/signed-distance-SRN/blob/main/LICENSE)) - Copyright (c) 2020 Chen-Hsuan Lin, and [Cat-3D](https://github.com/zxhuang1698/cat-3d). Please also cite these works if you use this codebase.
```
@inproceedings{huang2022planes,
  title={Planes vs. Chairs: Category-guided 3D shape learning without any 3D cues},
  author={Huang, Zixuan and Stojanov, Stefan and Thai, Anh and Jampani, Varun and Rehg, James M},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part I},
  pages={727--744},
  year={2022},
  organization={Springer}
}
```
```
@inproceedings{lin2020sdfsrn,
  title={SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images},
  author={Lin, Chen-Hsuan and Wang, Chaoyang and Lucey, Simon},
  booktitle={Advances in Neural Information Processing Systems ({NeurIPS})},
  year={2020}
}
```

If you use the Pix3D dataset, please also cite it.
```
@inproceedings{pix3d,
  title={Pix3D: Dataset and Methods for Single-Image 3D Shape Modeling},
  author={Sun, Xingyuan and Wu, Jiajun and Zhang, Xiuming and Zhang, Zhoutong and Zhang, Chengkai and Xue, Tianfan and Tenenbaum, Joshua B and Freeman, William T},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```