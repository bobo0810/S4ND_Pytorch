# S4ND: Single-Shot Single-Scale Lung Nodule Detection
#### 该仓库收录于[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)
# 说明
- 非官方实现
- 3D数据输入+Dense Block结构，造成 显存占用严重,可通过修改growth_rate、block_conv_num参数简化网络

----------

# 环境

| python版本 | pytorch版本 | 系统   |
|------------|-------------|--------|
| 3.5        | 0.4.1       | Ubuntu |

----------

# 网络结构

![](https://github.com/bobo0810/S4ND_Pytorch/blob/master/imgs/network.png)

----------
 # 参考文献

```
@inproceedings{khosravan2018s4nd,
  title={S4ND: Single-Shot Single-Scale Lung Nodule Detection},
  author={Khosravan, Naji and Bagci, Ulas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={794--802},
  year={2018},
  organization={Springer}
}
```