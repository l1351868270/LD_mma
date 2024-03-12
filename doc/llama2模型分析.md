# 代码
Adapted from https://github.com/facebookresearch/llama/blob/main/llama/model.py

# 模型参数量
## self-attention
### attention_norm
只有一个参数self.weight，参数量为 $dim$
### attention
attention：有四个参数self.wq, self.wk, self.wv, self.wo，每个权重的参数相同

存在bias时  $4dim^2+4dim$  

不存在bias时 $4dim^2$ 

### self-attention total
存在bias时 $4dim^2+5dim$ 

不存在bias时 $4dim^2+dim$
## mlp
### ffn_norm
只有一个参数self.weight，参数量为 $dim$
### ffn
有三个参数self.w1, self.w2, self.w3，参数 $`hiddenDim=2*4*dim/3=8*dim/3`$ 

存在bias时 

$$dim * hiddenDim + dim + hiddenDim * dim + hiddenDim + dim * hiddenDim + dim = 3*hiddenDim*dim + 2*dim + hiddenDim = 8*dim/3 * dim *3+ 2*dim + 8*dim/3 = 8*dim^2 + 2*dim + 8*dim/3$$

不存在bias时 $ 8*dim/3 * dim * 3 = 8*dim^2$

### ffn total
存在bias时 $8*dim/3 * dim * 3 + 3*dim + 8*dim/3 = 8*dim^2 + 3*dim + 8*dim/3$

不存在bias时 $ 8*dim/3 * dim * 3 + dim = 8*dim^2 + dim $

## transformer/per
存在bias时 
$4*dim^2+5*dim + 8*dim/3 * dim * 3 + 3*dim + 8*dim/3 = 4*dim^2+8*dim/3*dim*3+8*dim + 8*dim/3 = 12*dim^2 +8*dim + 8*dim/3$

不存在bias时 $ 4*dim^2+dim + 8*dim/3 * dim * 3 + dim = 12*dim^2+2*dim $







