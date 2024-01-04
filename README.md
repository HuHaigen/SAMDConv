# [Title （SAMDConv: Spatially Adaptive Multi-scale Dilated Convolution）](https://github.com/HuHaigen/SAMDConv)

> **The Paper Links**: [PRCV](https://link.springer.com/content/pdf/10.1007/978-981-99-8543-2_37.pdf?pdf=inline%20link).

## Abstract （摘要）
Dilated convolutions have received a widespread attention
in popular segmentation networks owing to the ability to enlarge the
receptive field without introducing additional parameters. However, it is
unsatisfactory to process multi-scale objects from different spatial positions in an image only by using multiple fixed dilation rates based on the
structure of multiple parallel branches. In this work, a novel spatiallyadaptive multi-scale dilated convolution (SAMDConv) is proposed to
adaptively adjust the size of the receptive field for different scale targets.
Specifically, a Spatial-Separable Attention (SSA) module is firstly proposed to personally select a reasonable combination of sampling scales
for each spatial location. Then a recombination module is proposed to
combine the output features of the four dilated convolution branches
according to the attention maps generated by SSA. Finally, a series of
experiments are conducted to verify the effectiveness of the proposed
method based on various segmentation networks on various datasets,
such as Cityscapes, ADE20K and Pascal VOC. The results show that
the proposed SAMDConv can obtain competitive performance compared
with normal dilated convolutions and depformable convolutions, and can
effectively improve the ability to extract multi-scale information by adaptively regulating the dilation rate.

**Keywords: SAMDConv · Image Segmentation · Receptive Field**

## Citation（引用）

Please cite our paper if you find the work useful: 

	@inproceedings{hu2023samdconv,
  	title={SAMDConv: Spatially Adaptive Multi-scale Dilated Convolution},
  	author={Hu, Haigen and Yu, Chenghan and Zhou, Qianwei and Guan, Qiu and Chen, Qi},
  	booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  	pages={460--472},
  	year={2023},
  	organization={Springer}
	}

**[⬆ back to top](#0-preface)**
