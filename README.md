# Correntropy-Induced Wasserstein GCN: Learning Graph Embedding via Domain Adaptation


## Introduction
-----------------------------------------
Graph embedding aims at learning vertex representations in a low-dimensional space by distilling information from a complex structured graph. Recent efforts in graph embedding have been devoted to generalizing the representations from the trained graph in a source domain to the new graph in a different target domain based on information transfer. However, when the graphs are contaminated by unpredictable and complex noise in practice, this transfer problem is quite challenging because of the need to extract helpful knowledge from the source graph and to reliably transfer knowledge to the target graph. This paper puts forward a two-step correntropy-induced Wasserstein GCN (graph convolutional network, or CW-GCN for short) architecture to facilitate the robustness in cross-graph embedding. In the first step, CW-GCN originally investigates correntropy-induced loss in GCN, which places bounded and smooth losses on the noisy nodes with incorrect edges or attributes. Consequently, helpful information are extracted only from clean nodes in the source graph. In the second step, a novel Wasserstein distance is introduced to measure the difference in marginal distributions between graphs, avoiding the negative influence of noise. Afterwards, CW-GCN maps the target graph to the same embedding space as the source graph by minimizing the Wasserstein distance, and thus the knowledge preserved in the first step is expected to be reliably transferred to assist the target graph analysis tasks. Extensive experiments demonstrate the significant superiority of CW-GCN over state-of-the-art methods
in different noisy environments.



## Environment

1. Linux Ubuntu 18.04   

2. python=3.6  

3. tensorflow-gpu=1.14.0  

4. networkx=2.5

5. CUDA = 9.2

6. GPU = RTX 2080Ti

## How to run?

1. Run train_F1_source.py.
2. Run train_F1.py.


## Citation

If you find the code useful in your research, please consider citing:

 

```

@ARTICLE{10179964,
  author={Wang, Wei and Zhang, Gaowei and Han, Hongyong and Zhang, Chi},
  journal={IEEE Transactions on Image Processing}, 
  title={Correntropy-Induced Wasserstein GCN: Learning Graph Embedding via Domain Adaptation}, 
  year={2023},
  volume={32},
  number={},
  pages={3980-3993},
  keywords={Noise measurement;Knowledge transfer;Task analysis;Pollution measurement;Data mining;Proteins;Convolutional neural networks;Graph embedding;graph convolutional network (GCN);cross-graph embedding;domain adaptation;correntropy},
  doi={10.1109/TIP.2023.3293774}}


```



-------------------------------------------
### FAQ
Please create a new issue

