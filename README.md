# HGCNSI-model
___
## Introduction of .py files
___
*
    `get_data.py` files are used to load data from public datasets and datasets collated from `VGG19`, and divide the data into `train`/`test`/`val.pt` in the form of `.pt` for `training`, `testing`, and `validation`, respectively.
    
*    `metrics.py` file provides the calculation method of the evaluation index (ADE/FDE), the calculation of Gaussian distribution loss and other related functions.
    
*    `model.py` file provides the design details of the overall architecture of the model in this paper, including: the design of `H-GCN`, the design of `SCA` extracted features, and the construction of `TCNs`.
    
*    `utils.py` file provides the construction of hyperedges in hypergraphs and the construction of interaction modules between pedestrians.
    
*    `train.py` file is used for model training.
    
*    `test.py` file for testing purposes, where the particle filter module is used for deterministic pedestrian trajectory prediction.

## The instruction runs
___
`Pre-process datasets`
```Bash
python get_data.py  --dataset zara1
```

`Training` process at `ZARA1` dataset.
```Bash
python train.py --lr 0.01 --n_ssagcn 2 --n_txpcnn 7  --dataset zara1 --tag ssagcn-zara1 --use_lrschd --num_epochs 400
```

`Testing` process
```Bash
python test.py
```

## Kernal code explanation
___
### Social interaction module
---
Here shows the code segment of `social_soft_attention` function appeared in `utils.py`.
```Python
if ((0 <= cosine_a <= 1) and (-cosine_a <= cosine_b <= 0) and (
                        0 <= cosine_theta <= 1)) or ((0 <= cosine_a <= 1) and (0 <= cosine_b <= sine_a) and (
                        0 <= cosine_theta <= sine_a)) or ((0 <= cosine_a <= 1) and (sine_a <= cosine_b <= 1) and (
                        -cosine_a <= cosine_theta <= 0)) or ((0 <= cosine_a <= 1) and (cosine_a <= cosine_b <= 1) and (
                        -1 <= cosine_theta <= -cosine_a)):
                    if (0 <= cosine_a <= 1) and (cosine_a <= cosine_b <= 1) and (-1 <= cosine_theta <= -cosine_a):  # 3
                        fssa_weight[s, k, h] = 3.0 / l2_norm_distance
                    if (0 <= cosine_a <= 1) and (sine_a <= cosine_b <= 1) and (-cosine_a <= cosine_theta <= 0):  # 4
                        fssa_weight[s, k, h] = 4.0 / l2_norm_distance
                    if (0 <= cosine_a <= 1) and (0 <= cosine_b <= sine_a) and (0 <= cosine_theta <= sine_a):  # 2
                        fssa_weight[s, k, h] = 2.0 / l2_norm_distance
                    if (0 <= cosine_a <= 1) and (-cosine_a <= cosine_b <= 0) and (0 <= cosine_theta <= 1):  # 1
                        fssa_weight[s, k, h] = 1.0 / l2_norm_distance
```
The different number of values `1`,`2`,`3` and `4` denote different `collision probability`. The larger the number, and the larger the value, the more likely the collision is to occur.

*We have also designed some other methods to compute the weights of hyper-edges, and some strategies on the construction of association matrices in hypergraphs, please refer to the utils.py code for detailed design ideas. Future work focuses on exploring variable hyperedge collision probability models.*

### The construction of hypergraph
---
Compute `hyperedges'` degree matrix
```Python
hyper_degree = np.array(fssa_weight[s].sum(0))
ni_hyperedge_degree_matrix = np.diag(np.power(hyper_degree, -1).flatten())
```
Compute `hypernodes'` degree matrix
```Python
hyper_node = np.array(fssa_weight[s].sum(1))
ni_hypernodes_degree_matrix = np.diag(np.power(hyper_node, -1).flatten())
```
Compute `adjacent matrix`
```Python
def compute_adjacent_matrix(test_matrix, weight_matrix, ni_degree_matrix, test_matrix_T):
    res = test_matrix.dot(weight_matrix).dot(ni_degree_matrix).dot(test_matrix_T)
    return res
adjacent_matrix = compute_adjacent_matrix(fssa_weight[s], weight_matrix, ni_hyperedge_degree_matrix,fssa_weight[s].T)
```

### The design of social interaction module
---
```Python
npeds = end_pos.size(1)
end_pos = end_pos[0, :, :]
T_length = end_pos.shape[1]
self.spatial_embedding = nn.Linear(2, self.embedding_dim)  # (2,10)
curr_rel_embedding = self.spatial_embedding(end_pos)  # curr_rel_embedding.shape:[npeds,10]
curr_rel_embedding = curr_rel_embedding.view(curr_rel_embedding.shape[0], curr_rel_embedding.shape[1], 1,self.embedding_dim).repeat(1, 1, self.L, 1)
```
`npeds` represents the number of peddestrians, the end_pos means the last observed time step `t`.  `T_length` means the length of observation time.

```Python
vgg = vgg.repeat(end_pos.shape[1] * npeds, 1, 1, 1)
vgg = vgg.view(-1, self.D)
features_proj = self.pre_att_proj(vgg)
features_proj = features_proj.view(-1, self.L, self.D_down)
features_proj = features_proj.view(-1, T_length, self.L, self.D_down)
```
The shape of `VGG` is [1,14,14,512].

```Python
xn = nn.functional.adaptive_avg_pool2d(features_proj, (1, 1))
xn = self.channel_linear(xn)
xn = features_proj * self.sigmoid(xn)
```
