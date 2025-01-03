# Generative Style Transfer for MR Image Segmentation: A Case of Glioma Segmentation in Sub-Saharan Africa
Abstract. In Sub-SaharanAfrica (SSA), the utilization of lower-quality Magnetic Resonance Imaging (MRI) technology raises questions about the applicability of machine learning (ML) methods for clinical tasks. This study aims to provide a robust deep learning-based brain tumor segmentation (BraTS) method tailored for the SSA population using a threefold approach. Firstly, the impact of domain shift from the SSA training data on model efficacy was examined, revealing no significant effect.Secondly, a comparative analysis of 3D and 2D full-resolution models using the nnU-Net framework indicates similar performance of both the models trained for 300 epochs achieving a five-fold cross-validation score of 0.93. Lastly, addressing the performance gap observed in SSA validation as opposed to the relatively larger BraTS glioma (GLI) validation set, two strategies are proposed: fine-tuning SSA cases using the GLI + SSA best-pretrained 2D fullres model at 300 epochs, and introducing a novel neural style transfer-based data augmentation technique for the SSA cases. This investigation underscores the potential of enhancing brain tumor prediction within SSA’s unique healthcare landscape.

Keywords: Brain Tumor Segmentation · Neural style transfer · nnU-Net

<div align="center">
  <img src="https://github.com/BTSKifaru/SPARK_BTS_KIFARU/blob/d27b026f7d41c4818e4b1cea4604df7a520276fc/Img/GenStyleTrans_BRaTS-SSA.png" 
  alt="Alt text" width="75%">
</div>

## Citation

If you use this research and/or software, please cite it using the following:

```bibtex
@article{Chepchirchir2025GenerativeStyleTransfer,
  author       = {Chepchirchir, Rancy and Sunday, Jill and Confidence, Raymond and Zhang, Dong and Chaudhry, Talha and Annazodo, Udunna and Muchungi, Kendi and Zou, Yujing},
  title        = {Generative Style Transfer for MR Image Segmentation: A Case of Glioma Segmentation in Sub-Saharan Africa},
  year         = {2025},
  month        = {January},
  version      = {1.0.0},
  doi          = {XXX}, 
  url          = {https://github.com/BTSKifaru/SPARK_BTS_KIFARU.git},
}
