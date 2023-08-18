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

## The introduction of data files
___
*
    The dataset named `dataset` contains the dataset used in the experiment. In a `dataset` dataset, the dataset named `raw` holds the original dataset, which contains five         categories: `ETH`,`HOTEL`,`UNIV`,`ZARA1` and `ZARA2`. Each subclass is divided into a `training set`, a `validation set`, and a `test set`, and consists of a `.pkl` and a `.txt` file. For example, a dataset named ETH has a `.txt` file named `biwi_eth`, and the data indexes in that file are: `frame_id`, `ped_id`, `x-coordinates` and `y-coordinates`.

*    In the folder named data, there are many files with the `.pt` suffix, these `.pt` files are processed from the original data set, and the processing process goes through some key functions, such as: `social_soft_attention`.
*    The dataset named `checkpoint` store pre-trained model files, with several files ending in `.pkl` and `.pth` in each dataset category.

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

### The design of SCA module
---
```Python
npeds = end_pos.size(1)
end_pos = end_pos[0, :, :]
T_length = end_pos.shape[1]
self.spatial_embedding = nn.Linear(2, self.embedding_dim)  # (2,10)
curr_rel_embedding = self.spatial_embedding(end_pos)  # curr_rel_embedding.shape:[npeds,10]
curr_rel_embedding = curr_rel_embedding.view(curr_rel_embedding.shape[0], curr_rel_embedding.shape[1], 1,self.embedding_dim).repeat(1, 1, self.L, 1)
```
`npeds` represents the number of peddestrians, the `end_pos` means the last observed time step `t`.  `T_length` means the length of observation time.

```Python
vgg = vgg.repeat(end_pos.shape[1] * npeds, 1, 1, 1)
vgg = vgg.view(-1, self.D)
self.pre_att_proj = nn.Linear(self.D, self.D_down)
features_proj = self.pre_att_proj(vgg)
features_proj = features_proj.view(-1, self.L, self.D_down)
features_proj = features_proj.view(-1, T_length, self.L, self.D_down)
```
The shape of `VGG` is [1,14,14,512], and the value of `self.D` and `self.D_down` is 512 and 16, respectively. 

```Python
xn = nn.functional.adaptive_avg_pool2d(features_proj, (1, 1))
self.channel_linear = nn.Linear(1, 1)
xn = self.channel_linear(xn)
xn = features_proj * self.sigmoid(xn)
```
The output shape of channel attention is [npeds, T_length, self.L, self.D_down], `self.L` is 196 and `self.D_down` is 16.

```Python
num_channels = T_length
group_norm = nn.GroupNorm(1, num_channels).cuda()
xs = group_norm(curr_rel_embedding)
self.spatial_linear = nn.Linear(embedding_dim, embedding_dim)
xs = self.spatial_linear(xs)
xs = curr_rel_embedding * self.sigmoid(xs)
out = torch.cat([xn, xs], dim=3)
```
The output shape of spatial attention is [npeds, T_length, self.L, self.D_down], and the value of `self.L` and `self.D_down` is 196 and 16, respectively. The value of embedding_dim is 10.

```Python
out = torch.cat([xn, xs], dim=3)
```
The shape of `out` is [npeds, T_length, self.L, self.D_down+self.embedding_dim], and the value of `self.embedding_dim` is 10. 

```Python
groups = 2
bs, chnls, h, w = out.data.size()
if chnls % groups:    # 2 denotes the number of groups
sequential_scene_attention = out
else:
chnls_per_group = chnls // groups
sequential_scene_attention = out.view(bs, groups, chnls_per_group, h, w)
sequential_scene_attention = torch.transpose(sequential_scene_attention, 1, 2).contiguous()
sequential_scene_attention = sequential_scene_attention.view(bs, -1, h, w)
sequential_scene_attention = sequential_scene_attention.sum(axis=2)

# dimension exchange process
sequential_scene_attention = sequential_scene_attention.permute(0, 2, 1)

# use 1x1 convolution
dimentional1_conv = nn.Conv2d(self.D_down + self.embedding_dim, self.bottleneck_dim, kernel_size=1, stride=1)
dimentional1_conv = dimentional1_conv.cuda()
sequential_scene_attention = dimentional1_conv(sequential_scene_attention.unsqueeze(-1))
sequential_scene_attention = sequential_scene_attention.squeeze(-1)

# dimension exchange process
sequential_scene_attention = sequential_scene_attention.permute(2, 0, 1)

# expanding on dimensions
sequential_scene_attention = sequential_scene_attention.unsqueeze(0)
```
The above process realizes information exchange and dimensional compression between channels.

### The design of SIRF module
___
```Python
def resampling_process(listname, n):
    ran_w = np.random.rand(n)  
    dd = [0 for i in range(n)]
    for i in range(len(ran_w)):
        j = 0
        while ran_w[i] > listname[j]:  # If the random number is within the interval, the subscript (j+1) is stored in the dd array
            if j < n - 1:
                if ran_w[i] <= listname[j + 1]:
                    break
                else:
                    j += 1
            else:
                j = j - 1
                break
        dd[i] = j + 1
    return dd
```
The code ```ran_w = np.random.rand(n)``` aims to generate `N` random numbers, and stored in the array `listname` are the weights of the particles. 

In our paper, the design of our SIRF module is illustrated in the following python code:
```Python
# take particle filter measure!
particle_number = 20  # represents the number of particles
filter_value = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2)
weight_cluster_ = torch.zeros(particle_number, V_pred.shape[1], 2)
cum_ = torch.zeros(particle_number, V_pred.shape[1], 2)
for pred_len in range(filter_value.shape[0]):
    o_mux = mux_deepcopy[pred_len, :].data.cpu()
    o_muy = muy_deepcopy[pred_len, :].data.cpu()
    o_sx = sx_deepcopy[pred_len, :].data.cpu()
    o_sy = sy_deepcopy[pred_len, :].data.cpu()
    o_corr = corr_deepcopy[pred_len, :].data.cpu()
    predicted_x = torch.zeros(filter_value.shape[1])  # represents the npeds
    predicted_y = torch.zeros(filter_value.shape[1])
    next_values_cluster = torch.zeros(filter_value.shape[1], particle_number, 2)  # the first is npeds
    for node in range(filter_value.shape[1]):
        mmean = [o_mux[node], o_muy[node]]
        ccov = [[o_sx[node] * o_sx[node], o_corr[node] * o_sx[node] * o_sy[node]],
                            [o_corr[node] * o_sx[node] * o_sy[node], o_sy[node] * o_sy[node]]]
        mmean = np.array(mmean, dtype='float')  # tensor -> numpy
        ccov = np.array(ccov, dtype='float')
        predicted_value = np.random.multivariate_normal(mmean, ccov, particle_number)
        predicted_x[node] = predicted_value[0][0]
        predicted_y[node] = predicted_value[0][1]
        next_values_cluster[node, :, 0] = torch.from_numpy(predicted_value[:, 0])
        next_values_cluster[node, :, 1] = torch.from_numpy(predicted_value[:, 1])
    next_values_cluster_copy = next_values_cluster.clone()
    particle_main = torch.zeros(filter_value.shape[1], 2)
    for i in range(filter_value.shape[1]):
        particle_main_i = torch.from_numpy(np.mean(next_values_cluster.clone().numpy()[i, :], axis=0))
        particle_main[i] = particle_main_i
    for i in range(next_values_cluster.shape[1]):
        particle_i = next_values_cluster[:, i]
        weight_cluster_[i] = 1 / (np.sqrt(2 * np.pi * (1 / particle_number))) * np.exp(
                        -(particle_main - particle_i) ** 2 / (2 * (1 / particle_number)))
    weight_cluster_ = weight_cluster_ / (sum(weight_cluster_) + 1e-20)
    for j in range(next_values_cluster.shape[1]):
        cum_[j] = functools.reduce(lambda x, y: x + y, weight_cluster_[:j + 1])
    for i in range(filter_value.shape[1]):
        i_x = resampling_process(cum_[:, i, 0], particle_number)  # the second number is particle number
        i_y = resampling_process(cum_[:, i, 1], particle_number)
        next_values_cluster_copy[i, [k for k in range(particle_number)], 0] = next_values_cluster[i, i_x, 0]
        next_values_cluster_copy[i, [k for k in range(particle_number)], 1] = next_values_cluster[i, i_y, 1]
    next_values_cluster_copy = next_values_cluster_copy.mean(axis=1, keepdim=False)
    new_x = next_values_cluster_copy[:, 0]
    new_y = next_values_cluster_copy[:, 1]
    filter_value[pred_len, :, 0] = new_x
    filter_value[pred_len, :, 1] = new_y
V_pred_rel_to_abs = nodes_rel_to_nodes_abs(filter_value.data.cpu().numpy().squeeze().copy(),V_x[-1, :, :].copy())
```
The `normalization` operation is showed in the following code:
```Python
weight_cluster_ = weight_cluster_ / (sum(weight_cluster_) + 1e-20)
```
The array named `cum` store weights, and the summation function shows below:
```Python
cum_[j] = functools.reduce(lambda x, y: x + y, weight_cluster_[:j + 1])
```
The following code denotes Sampling Importance Re-sampling (`SIR`) operation:
```Python
for i in range(filter_value.shape[1]):
    i_x = resampling_process(cum_[:, i, 0], particle_number)  # the second number is particle number
    i_y = resampling_process(cum_[:, i, 1], particle_number)
    next_values_cluster_copy[i, [k for k in range(particle_number)], 0] = next_values_cluster[i, i_x, 0]
    next_values_cluster_copy[i, [k for k in range(particle_number)], 1] = next_values_cluster[i, i_y, 1]
```
```Python
filter_value[pred_len, :, 0] = new_x
filter_value[pred_len, :, 1] = new_y
```
The output results `new_x` and `new_y` represent the predicted trajectories of a person. 

The following code shows the situation assuming that each pedestrian interacts with all the pedestrians around him:
```Python
for s in range(seq_len):
    step_ = seq_[:, :, s]  
    step_rel = seq_rel[:, :, s]
    # initialize a adjacent matrix
    for h in range(len(step_)):
        V[s, h, :] = step_rel[h]
        fssa_weight[s, h, h] = 1.0
        for k in range(len(step_)):
            fssa_weight[s, k, h] = 1.0
    original_weight_list = np.array([1.0 for i in range(max_nodes)])
    weight_matrix = np.diag(original_weight_list)
    # compute hypernodes' degree matrix
    hyper_node = np.array(fssa_weight[s].sum(1))
    ni_hypernodes_degree_matrix = np.diag(np.power(hyper_node, -1).flatten())
    # compute adjacent matrix
    adjacent_matrix = fssa_weight[s].dot(weight_matrix).dot(fssa_weight[s].T)
    fssa_weight[s] = softmax_operation(adjacent_matrix)
```
```Python
for h in range(len(step_)):
    V[s, h, :] = step_rel[h]
    fssa_weight[s, h, h] = 1.0
    for k in range(len(step_)):
        fssa_weight[s, k, h] = 1.0
```
where this code snippet assigns a weight value of `1` to every two pedestrians that interact with each other.

The following code shows that when the interaction between pedestrians is reasonably divided and formed to form a hypergraph where all hyper-edge weights are always `1`, a `static` inter-pedestrian interaction diagram is used to model the `high-order interaction` relationship between pedestrians.
```Python
for s in range(seq_len):
    step_ = seq_[:, :, s]
    step_rel = seq_rel[:, :, s]
    
    # initialize a incidence matrix
    for h in range(len(step_)):
        V[s, h, :] = step_rel[h]
        fssa_weight[s, h, h] = 1.0
        for k in range(len(step_)):
            if k == h:
                continue
            cosine_a = get_cosine(step_[h], step_[k], step_rel[h])
            cosine_b = get_cosine(step_[k], step_[h], step_rel[k])
            cosine_theta = get_cosine_angle(step_rel[h], step_rel[k])
            sine_a = get_sin(step_[h], step_[k], step_rel[h])
    
            if ((0 <= cosine_a <= 1) and (-cosine_a <= cosine_b <= 0) and (
                0 <= cosine_theta <= 1)) or ((0 <= cosine_a <= 1) and (0 <= cosine_b <= sine_a) and (
                0 <= cosine_theta <= sine_a)) or ((0 <= cosine_a <= 1) and (sine_a <= cosine_b <= 1) and (
                -cosine_a <= cosine_theta <= 0)) or ((0 <= cosine_a <= 1) and (cosine_a <= cosine_b <= 1) and (
                -1 <= cosine_theta <= -cosine_a)):
                if (0 <= cosine_a <= 1) and (cosine_a <= cosine_b <= 1) and (-1 <= cosine_theta <= -cosine_a):
                    fssa_weight[s, k, h] = 1.0
                if (0 <= cosine_a <= 1) and (sine_a <= cosine_b <= 1) and (-cosine_a <= cosine_theta <= 0):
                    fssa_weight[s, k, h] = 1.0
                if (0 <= cosine_a <= 1) and (0 <= cosine_b <= sine_a) and (0 <= cosine_theta <= sine_a):
                    fssa_weight[s, k, h] = 1.0
                if (0 <= cosine_a <= 1) and (-cosine_a <= cosine_b <= 0) and (0 <= cosine_theta <= 1):
                    fssa_weight[s, k, h] = 1.0
    
    original_weight_list = np.array([1.0 for i in range(max_nodes)])
    weight_matrix = np.diag(original_weight_list)
    
    # compute hyperedges' degree matrix
    hyper_degree = np.array(fssa_weight[s].sum(0))
    ni_hyperedge_degree_matrix = np.diag(np.power(hyper_degree, -1).flatten())
    
    # compute hypernodes' degree matrix
    hyper_node = np.array(fssa_weight[s].sum(1))
    ni_hypernodes_degree_matrix = np.diag(np.power(hyper_node, -1).flatten())
    
    # compute adjacent matrix
    adjacent_matrix = compute_adjacent_matrix(fssa_weight[s], weight_matrix, ni_hyperedge_degree_matrix,fssa_weight[s].T)
    adjacent_matrix_softmax = softmax_operation(adjacent_matrix)
    fssa_weight[s] = ni_hypernodes_degree_matrix.dot(adjacent_matrix_softmax).dot(ni_hypernodes_degree_matrix)
```

```Python
 if ((0 <= cosine_a <= 1) and (-cosine_a <= cosine_b <= 0) and (
    0 <= cosine_theta <= 1)) or ((0 <= cosine_a <= 1) and (0 <= cosine_b <= sine_a) and (
    0 <= cosine_theta <= sine_a)) or ((0 <= cosine_a <= 1) and (sine_a <= cosine_b <= 1) and (
    -cosine_a <= cosine_theta <= 0)) or ((0 <= cosine_a <= 1) and (cosine_a <= cosine_b <= 1) and (
    -1 <= cosine_theta <= -cosine_a)):
    if (0 <= cosine_a <= 1) and (cosine_a <= cosine_b <= 1) and (-1 <= cosine_theta <= -cosine_a):
        fssa_weight[s, k, h] = 1.0
    if (0 <= cosine_a <= 1) and (sine_a <= cosine_b <= 1) and (-cosine_a <= cosine_theta <= 0):
        fssa_weight[s, k, h] = 1.0
    if (0 <= cosine_a <= 1) and (0 <= cosine_b <= sine_a) and (0 <= cosine_theta <= sine_a):
        fssa_weight[s, k, h] = 1.0
    if (0 <= cosine_a <= 1) and (-cosine_a <= cosine_b <= 0) and (0 <= cosine_theta <= 1):
        fssa_weight[s, k, h] = 1.0
```
The above code shows the details of the hypergraph partition.







