# Insonesia

The dataset I use is from https://commonvoice.mozilla.org/zh-CN/datasets

Results on cv-corpus-8.0-2022-01-19-Indonesia datasets.


## BLSTM

* SP: 3-way speed perturbation
* SA: SpecAugment
* SDL: SchedulerEarlyStop、SchedulerTransformerEarlyStop、SchedulerWarmupMileStone
* AM: BLSTM with 6.19M parameters.
* Hyper-parameters of AM training: `lamb=0.01, n_layers=3, hdim=320, lr=0.001`
#

* Model size/M: 6.19
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 3080 Ti

具体 config.json, monitor.png, readme.md, scripts.tar.gz 分别保存在 exp/ 对应目录下

| Unit  | LM     | Test  | SP | SA | model     | SDL                  | loss_fn | n_layers | idim | hdim | num_classes | dropout |
| ----- | ------ | ----- | -- | -- | --------- | -------------------- | ------- | -------- | ---- | ---- | ----------- | ------- | 
| phone | 3_gram | 11.49 | Y  | N  | BLSTM     | EarlyStop            | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 10.78 | Y  | N  | BLSTM     | EarlyStop            | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 3-gram | 10.05 | Y  | Y  | BLSTM     | EarlyStop            | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 9.65  | Y  | Y  | BLSTM     | EarlyStop            | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 3-gram | 10.50 | Y  | Y  | BLSTM     | TransformerEarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 9.98  | Y  | Y  | BLSTM     | TransformerEarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 3-gram | 10.29 | Y  | Y  | BLSTM     | WarmupMileStone      | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 9.67  | Y  | Y  | BLSTM     | WarmupMileStone      | crf     | 3        | 120  | 320  | 42          | 0.5     |




## VGG-BLSTM

* Model size/M: 9.31
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 3080 Ti


| Unit  | LM     | Test  | SP | SA | model     | SDL                  | loss_fn | n_layers | idim | hdim | num_classes | dropout |
| ----- | ------ | ----- | -- | -- | --------- | -------------------- | ------- | -------- | ---- | ---- | ----------- | ------- | 
| phone | 3_gram | 8.51  | Y  | Y  | VGG_BLSTM | EarlyStop            | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 8.14  | Y  | Y  | VGG-BLSTM | EarlyStop            | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 3-gram | 8.50  | Y  | Y  | VGG-BLSTM | WarmupMileStone      | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 8.09  | Y  | Y  | VGG-BLSTM | WarmupMileStone      | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 3-gram | 8.70  | Y  | Y  | VGG-BLSTM | TransformerEarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 8.25  | Y  | Y  | VGG-BLSTM | TransformerEarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |



## Conformer

* Model size/M: 7.83
* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 3080 Ti



| Unit  | LM     | Test | SP  | SA | model     | SDL             | num_cells | idim | hdim | num_classes | conv_multiplier | delta_feats |
| ----- | ------ | ---- | --- | -- | --------- | --------------- | --------- | ---- | ---- | ----------- | --------------- | ----------- | 
| phone | 3_gram | 6.38 | N   | Y  | Conformer | WarmupMileStone | 16        | 80   | 128  | 42          | 256             | false       |
| phone | 4_gram | 6.05 | N   | Y  | Conformer | WarmupMileStone | 16        | 80   | 128  | 42          | 256             | false       |



## Crosslingual

- Chengrui Zhu, Keyu An, Huahuan Zheng and Zhijian Ou, "Multilingual and crosslingual speech recognition using phonological-vector based phone embeddings", IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), 2021. [pdf](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf)
- [THU-SPMI@ASRU2021: 基于音位矢量的多语言与跨语言语音识别，促进多语言信息共享与迁移](https://mp.weixin.qq.com/s?__biz=MzU3MzgyNDMzMQ==&mid=2247484519&idx=1&sn=492cc4e098df0077fc51ecb163d8c8a4&chksm=fd3a8843ca4d015560d9cb3fcfc9e0741c0cd898ad69c7b94b6e092f60ee3e6db3c1f9ccf54d&mpshare=1&scene=1&srcid=0612RqU7DGRZG5XQqg0L2Le1&sharer_sharetime=1655005703359&sharer_shareid=96a0960dd6af6941d3216dad8f2d3a50&key=311fd5318431ff9c5328351edecbba7c5d812fe2ebfc0df6c234172e3cd3b056a5dc35c3c9476a894d7828f7932113f61f420f11bd98bd9f19a18dbbce60d74810202a96eb262756df24294667730f65015d74e3b84a12d358110afd52a3e26cd7bfd692bf4322094d61d031aab32954e42b0043521ae4d7a3ba8b52f177429f&ascene=1&uin=MjI2OTIxNjcxMA%3D%3D&devicetype=Windows+10+x64&version=6209051a&lang=zh_CN&exportkey=AxSPQ4EqXRXSVFCXOPz3zSc%3D&acctmode=0&pass_ticket=5FeYTkI0JWlQDdwbOw%2B90azniyK49b4eF6G1m7lzzoG4aLbog8BRp8ZMiC%2BnfXI5&wx_header=0)

#
* mode : 三种模型类型 [ "flat_phone" , "joinap_linear" , "joinap_nonlinear"]
* usg : [ "finetune" , "zero-shot-eval" , "few-shot-eval" , "multi-eval" , "multi-finetune-eval"]
#

* Model : `VGG-BLSTM`
* Model size/M: 16.71 (joinap_linear)
* Model size/M: 17.03 (joinap_nonlinear)

* GPU info \[1\]
  * \[1\] NVIDIA GeForce RTX 3080 Ti

| Unit  | SDL    | Test  | SP | SA | mode              | usg            | SDL       | loss_fn | n_layers | idim | hdim | num_classes | dropout |
| ----- | ------ | ----- | -- | -- | ----------------- | -------------- | --------- | ------- | -------- | ---- | ---- | ----------- | ------- |
| phone | 3_gram |       | Y  | Y  | joinap_linear     | zero_shot_eval | EarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram |       | Y  | Y  | joinap_linear     | zero-shot-eval | EarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 3-gram | 6.26  | Y  | Y  | joinap_linear     | few-shot-eval  | EarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 5.97  | Y  | Y  | joinap_linear     | few-shot-eval  | EarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 3-gram |       | Y  | Y  | joinap_nonlinear  | zero-shot-eval | EarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram |       | Y  | Y  | joinap_nonlinear  | zero-shot-eval | EarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 3-gram | 7.19  | Y  | Y  | joinap_nonlinear  | few-shot-eval  | EarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |
| phone | 4-gram | 6.87  | Y  | Y  | joinap_nonlinear  | few-shot-eval  | EarlyStop | crf     | 3        | 120  | 320  | 42          | 0.5     |


