# 基于多语言/跨语言ASR(joinAP)
**本文档介绍如何使用joinAP模型进行多语言/跨语言ASR的研究，开始前推荐先阅读以下参考资料了解理论知识以及相关细节**
- Chengrui Zhu, Keyu An, Huahuan Zheng and Zhijian Ou, "Multilingual and crosslingual speech recognition using phonological-vector based phone embeddings", IEEE Workshop on Automatic Speech Recognition and Understanding (ASRU), 2021. [pdf](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ASRU21_JoinAP.pdf)

**本文档将细化说明实验的每一步过程，包括数据获取，数据预处理，发音词典的生成(G2P),音位矢量生成，模型训练评估等。**

* [数据获取及预处理](#数据获取及预处理)

* [发音词典](#发音词典)

* [音位矢量](#音位矢量)

* [训练及评估](#训练及评估)

## 数据获取及预处理

本次实验我们选择开源Common Voice[数据](https://commonvoice.mozilla.org/zh-CN/datasets)作为我们原始训练语料，我们针对CommonVoiceCorpus5.1中德语(750小时)，法语(604小时)，西班牙语(521小时)，意大利语(167小时)，波兰语(119小时)进行多语言以及跨语言的实验，这些开源数据我们可以直接下载得到。下载好的数据由我们音频以及训练，验证，测试文本组成。

数据预处理阶段我们仿照kaldi脚本处理[CAT-commonvoice](https://github.com/thu-spmi/CAT/tree/master/egs/commonvoice),其中 **local**下的脚本文件无需任何处理，我们只需修改**run_mc.sh** 脚本文件即可,脚本中**stage7**开始为joinAP模型的训练，我们目前只说明前6步

```
lang=(de it fr es)
datadir=/path/to/cv-corpus-5.1-2020-06-22/

saved_dict="saved_dict"
dict_tmp=data/local/dict_tmp
 
```
**lang**决定我们训练的语言种类de(德语)，it(意大利语)，fr(法语)，es(西班牙语)可以根据自己硬件要求进行实验。**datadir**存放我们训练的数据
**saved_dict**存放我们训练的完整发音词典**dict_tmp**存放我们从文本数据中切分下的未注音的词典(注音部分我们后续说明)



```
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

```
这部分代码仿照kaldi处理，主要生成**train,dev,test**下的**wav.scp,text,utt2spk,spk2utt**文件



```
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then

```
这部分主要是针对我们Multilingual训练词典进行加噪并排序并用数字编号的声学单元units.txt，以及用数字标号的词典，lexicon_numbers.txt。以及生成德语，法语，西班牙以及意大利语的TLG.fst。




```
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then

```
这一部分我们利用FBank进行特征提取和特征归一化由于joinAP模型基于VGGBLSTM系列模型所以我们`conf`目录下fbank.conf设置16K和40维进行提取，默认使用三倍数据增广。





```
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then

```
单词序列转换为标签序列





```
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then

```
这一部分我们将训练以及测试数据加一阶和二阶差分以便于模型训练






```
if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then

```
生成den_lm.fst。最后，由den_lm.fst和标签文件出发，计算出标签序列$l$的对数概率 $logp(l)，称为path weight整合到data/pickle下。

## 发音词典

由于Common Voice数据没有提供相应的词典，所以我们需要自己手动生成。这里我们在**stage1**步骤中：

`cat data/${train_set}/text | awk '{$1="";print $0}' | sed 's/ /\n/g' | sort -u >$dict_tmp/wordlist_${x}` 这一命令脚本在data/local/dict_tmp目录中生成de,fr,es,it没有注音的词典(wordlist_de,wordlist_it,wordlist_es,wordlist_fr),我们需要利用G2P去将目前没有注音的词典去注音。

**以下说明G2P的安装以及使用。**

**Phonetisaurus G2P**

**安装**：[G2P](https://github.com/AdolfVonKleist/Phonetisaurus)

创建一个目录用于我们G2P的安装
```
$ mkdir g2p
$ cd g2p/
```
下载并安装 OpenFst-1.7.2
```
$ wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz
$ tar -xvzf openfst-1.7.2.tar.gz
$ cd openfst-1.7.2
$ ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts
$ make -j
$ sudo make install
$ echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib:/usr/local/lib/fst' \
     >> ~/.bashrc
$ source ~/.bashrc
$ cd ..
```
从最新github-master git下最新的Phonetisaurus 并使用 python3 绑定进行编译：
```
$ git clone https://github.com/AdolfVonKleist/Phonetisaurus.git
$ cd Phonetisaurus
$ sudo pip3 install pybindgen
$ PYTHON=python3 ./configure --enable-python
$ make
$ sudo make install
$ cd python
$ cp ../.libs/Phonetisaurus.so .
$ sudo python3 setup.py install
$ cd ../..
```
获取并安装 mitlm 
```
$ git clone https://github.com/mitlm/mitlm.git
$ cd mitlm/
$ ./autogen.sh
$ make
$ sudo make install
$ cd ..
```
获取最新版本 CMUdict 的副本并清理一下：
```
$ mkdir example
$ cd example
$ wget https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
$ cat cmudict.dict \
  | perl -pe 's/\([0-9]+\)//;
              s/\s+/ /g; s/^\s+//;
              s/\s+$//; @_ = split (/\s+/);
              $w = shift (@_);
              $_ = $w."\t".join (" ", @_)."\n";' \
  > cmudict.formatted.dict
```
使用包装的脚本训练具有默认参数的完整模型。注意：默认 python3 绑定编译：
```
$ phonetisaurus-train --lexicon cmudict.formatted.dict --seq2_del
```
**至此我们完成了G2P工具安装**

测试：(test.wlist就是我们需要注音的文件)

```
$ phonetisaurus-apply --model train/model.fst --word_list test.wlist
```
test  T EH1 S T

jumbotron  JH AH1 M B OW0 T R AA0 N

excellent  EH1 K S AH0 L AH0 N T

eggselent  EH1 G S L AH0 N T

**安装好后的G2P需要训练好的fst各种语言的模型对我们德语，法语，西班牙，意大利语进行注音**

**LanguageNet Grapheme-to-Phoneme Transducers**：[FST](https://github.com/uiuc-sst/g2ps)

git下已经训练好的FST，并测试
```
$ git clone https://github.com/uiuc-sst/g2ps
$ phonetisaurus-g2pfst --model=g2ps/models/akan.fst --word=ahyiakwa
```
**注意**：models下的fst文件需要解压

**运行以下脚本可以生成每种语言发音词典**
```
g2ps=/mnt/workspace/liziwei/g2ps/models/ # g2p model 的路径
dict_tmp=/mnt/workspace/liziwei/data/local/dict_tmp/ # 未被注音的词典
    phonetisaurus-apply --model $g2ps/french_8_4_2.fst --word_list $dict_tmp/wordlist_fr > $dict_tmp/lexicon_fr
    phonetisaurus-apply --model $g2ps/german_8_4_2.fst --word_list $dict_tmp/wordlist_de > $dict_tmp/lexicon_de
    phonetisaurus-apply --model $g2ps/spanish_4_3_2.fst --word_list $dict_tmp/wordlist_es > $dict_tmp/lexicon_es
    phonetisaurus-apply --model $g2ps/italian_8_2_3.fst --word_list $dict_tmp/wordlist_it > $dict_tmp/lexicon_it
```
**dict_tmp目录下已经生成我们所需发音词典**

## 音位矢量

在多语言声学模型训练时我们希望能用一个矢量表示每个音素，这个矢量包含音素的发音信息。由此我们引入音位矢量对我们最后输出线性层进行修改。音位矢量的形成我们用到了panphon工具包，panphon定义了全部 IPA 音素符号到区别特征的映射，我们可以直接根据 IPA 音素得到它的区别特征表达。

**panphon工具包**[panphon](https://github.com/dmort27/panphon)

我们需要对每个音素进行手动标记，panphon一共提供24个区别特征，每种特征又被用“+”、“-”、“0”三种符号表示，由此我们将**“+”被编码“10”“-”被编码为“01”，“00”则表示“0”**。这样以来，24 维的区别特征被编码为了 48 维的音位矢量。再加上三个特殊 token：blk（空）、spn（说话噪音）、nsn（自然噪音）一共51维音位矢量。

**注意**：映射表中未出现的集外音素，我们可以将其映射到与其它声学上最相似的音素。

**音素映射关系表**[IPA](https://github.com/dmort27/panphon/blob/master/panphon/data/ipa_all.csv)

我们可以通过映射表将每个音素进行编码 `以下展示我们以德语为例`

|    |   token | IPA   |   syl+ |   syl- |   son+ |   son- |   cons+ |   cons- |   cont+ |   cont- |   delrel+ |   delrel- |   lat+ |   lat- |   nas+ |   nas- |   srtid+ |   strid- |   voi+ |   voi- |   sg+ |   sg- |   cg+ |   cg- |   ant+ |   ant- |   cor+ |   cor- |   distr+ |   distr- |   lab+ |   lab- |   hi+ |   hi- |   lo+ |   lo- |   back+ |   back- |   round+ |   round- |   velaric+ |   velaric- |   tense+ |   tense- |   long+ |   long- |   hitone+ |   hitone- |   hireg+ |   hireg- |   blk |   nsn |   spn |
|---:|--------:|:------|-------:|-------:|-------:|-------:|--------:|--------:|--------:|--------:|----------:|----------:|-------:|-------:|-------:|-------:|---------:|---------:|-------:|-------:|------:|------:|------:|------:|-------:|-------:|-------:|-------:|---------:|---------:|-------:|-------:|------:|------:|------:|------:|--------:|--------:|---------:|---------:|-----------:|-----------:|---------:|---------:|--------:|--------:|----------:|----------:|---------:|---------:|------:|------:|------:|
|  0 |       1 | blk |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |         0 |         0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |       0 |       0 |        0 |        0 |          0 |          0 |        0 |        0 |       0 |       0 |         0 |         0 |        0 |        0 |     1 |     0 |     0 |
|  1 |       2 | NSN |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |         0 |         0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |       0 |       0 |        0 |        0 |          0 |          0 |        0 |        0 |       0 |       0 |         0 |         0 |        0 |        0 |     0 |     1 |     0 |
|  2 |       3 | SPN |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |         0 |         0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |       0 |       0 |        0 |        0 |          0 |          0 |        0 |        0 |       0 |       0 |         0 |         0 |        0 |        0 |     0 |     0 |     1 |
|  3 |       4 | #     |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |         0 |         0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |       0 |       0 |        0 |        0 |          0 |          0 |        0 |        0 |       0 |       0 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
|  4 |       5 | 1     |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |         0 |         0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |       0 |       0 |        0 |        0 |          0 |          0 |        0 |        0 |       0 |       0 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
|  5 |       6 | 7     |      0 |      0 |      0 |      0 |       0 |       0 |       0 |       0 |         0 |         0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |      0 |      0 |      0 |      0 |        0 |        0 |      0 |      0 |     0 |     0 |     0 |     0 |       0 |       0 |        0 |        0 |          0 |          0 |        0 |        0 |       0 |       0 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
|  6 |       7 | a     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     1 |     0 |       0 |       1 |        0 |        1 |          0 |          1 |        1 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
|  7 |       8 | b     |      0 |      1 |      0 |      1 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      1 |      0 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
|  8 |       9 | d     |      0 |      1 |      0 |      1 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      1 |      0 |      1 |      0 |        0 |        1 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
|  9 |      10 | e     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        1 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 10 |      11 | f     |      0 |      1 |      0 |      1 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        1 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      1 |      0 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 11 |      12 | g     |      0 |      1 |      0 |      1 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       1 |       0 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 12 |      13 | h     |      0 |      1 |      1 |      0 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 13 |      14 | i     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        1 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 14 |      15 | j     |      0 |      1 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 15 |      16 | k     |      0 |      1 |      0 |      1 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       1 |       0 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 16 |      17 | l     |      0 |      1 |      1 |      0 |       1 |       0 |       1 |       0 |         0 |         1 |      1 |      0 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      1 |      0 |      1 |      0 |        0 |        1 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 17 |      18 | m     |      0 |      1 |      1 |      0 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      1 |      0 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      1 |      0 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 18 |      19 | n     |      0 |      1 |      1 |      0 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      1 |      0 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      1 |      0 |      1 |      0 |        0 |        1 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 19 |      20 | o     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       1 |       0 |        1 |        0 |          0 |          1 |        1 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 20 |      21 | p     |      0 |      1 |      0 |      1 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      1 |      0 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 21 |      22 | r     |      0 |      1 |      1 |      0 |       1 |       0 |       1 |       0 |         0 |         0 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      1 |      0 |      1 |      0 |        0 |        1 |      0 |      1 |     0 |     0 |     0 |     0 |       0 |       0 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 22 |      23 | s     |      0 |      1 |      0 |      1 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        1 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      1 |      0 |      1 |      0 |        0 |        1 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 23 |      24 | t     |      0 |      1 |      0 |      1 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      1 |      0 |      1 |      0 |        0 |        1 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 24 |      25 | ts    |      0 |      1 |      0 |      1 |       1 |       0 |       0 |       1 |         1 |         0 |      0 |      1 |      0 |      1 |        1 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      1 |      0 |      1 |      0 |        0 |        1 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 25 |      26 | u     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      1 |      0 |     1 |     0 |     0 |     1 |       1 |       0 |        1 |        0 |          0 |          1 |        1 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 26 |      27 | v     |      0 |      1 |      0 |      1 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        1 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      1 |      0 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 27 |      28 | w     |      0 |      1 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     1 |     0 |     0 |     1 |       1 |       0 |        1 |        0 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 28 |      29 | y     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      1 |      0 |     1 |     0 |     0 |     1 |       0 |       1 |        1 |        0 |          0 |          1 |        1 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 29 |      30 | z     |      0 |      1 |      0 |      1 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        1 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      1 |      0 |      1 |      0 |        0 |        1 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 30 |      31 | ç     |      0 |      1 |      0 |      1 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 31 |      32 | ø     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        1 |        0 |          0 |          1 |        1 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 32 |      33 | ŋ     |      0 |      1 |      1 |      0 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      1 |      0 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       1 |       0 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 33 |      34 | œ     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        1 |        0 |          0 |          1 |        0 |        1 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 34 |      35 | ɔ     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       1 |       0 |        1 |        0 |          0 |          1 |        0 |        1 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 35 |      36 | ɛ     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        1 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 36 |      37 | ɡ     |      0 |      1 |      0 |      1 |       1 |       0 |       0 |       1 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       1 |       0 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 37 |      38 | ɪ     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        1 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 38 |      39 | ʁ     |      0 |      1 |      0 |      1 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      1 |      0 |      1 |        0 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       1 |       0 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 39 |      40 | ʃ     |      0 |      1 |      0 |      1 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        1 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |      0 |      1 |      1 |      0 |        1 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 40 |      41 | ʊ     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       1 |       0 |        1 |        0 |          0 |          1 |        0 |        1 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 41 |      42 | ʏ     |      1 |      0 |      1 |      0 |       0 |       1 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        0 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      0 |      0 |      1 |        0 |        0 |      0 |      1 |     1 |     0 |     0 |     1 |       0 |       1 |        1 |        0 |          0 |          1 |        0 |        1 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |
| 42 |      43 | ʒ     |      0 |      1 |      0 |      1 |       1 |       0 |       1 |       0 |         0 |         1 |      0 |      1 |      0 |      1 |        1 |        0 |      1 |      0 |     0 |     1 |     0 |     1 |      0 |      1 |      1 |      0 |        1 |        0 |      0 |      1 |     0 |     1 |     0 |     1 |       0 |       1 |        0 |        1 |          0 |          1 |        0 |        0 |       0 |       1 |         0 |         0 |        0 |        0 |     0 |     0 |     0 |


**手动编码完成后我们还需将其转换成`numpy`格式文件以便于模型训练**

```
$ pip install numpy
$ import numpy as np
$ np.save('de.npy',path)

```
使用numpy读取音位矢量：
```
$import numpy as np
$de=np.load('de.npy')
$de

array( [[0, 0, 0, ..., 1, 0, 0],

     [0, 0, 0, ..., 0, 1, 0],
     
     [0, 0, 0, ..., 0, 0, 1],
           ...,
     [1, 0, 1, ..., 0, 0, 0],
       
     [1, 0, 1, ..., 0, 0, 0],
     
     [0, 1, 0, ..., 0, 0, 0]], dtype=int64)
```

**至此我们完成了音位矢量的构建**

## 训练及评估

训练及评估部分具体可以参考[CAT-joinAP](https://github.com/thu-spmi/CAT/blob/master/joinap.md)官方说明，这里我们演示**JoinAP-Linear**

**训练部分代码**

```
PARENTDIR='.'
dir="exp/mc_linear/"
DATAPATH=$PARENTDIR/data/
mkdir -p $dir

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    unset CUDA_VISIBLE_DEVICES
    
    if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
        echo ""
        tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
    elif [ $NODE == 0 ]; then
        echo ""
        echo "'$dir/scripts.tar.gz' already exists."
        echo "If you want to update it, please manually rm it then re-run this script."
    fi

  # uncomment the following line if you want to use specified GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3,4                    \
    python3 ctc-crf/train.py --seed=0               \
        --world-size 1 --rank $NODE                 \
        --mc-train-pv=./embedding/mul.npy            \
        --batch_size=128                            \
        --dir=$dir                                  \
        --config=$dir/config.json         \
        --trset=data/pickle/train.pickle            \
        --devset=data/pickle/dev.pickle             \
        --data=$DATAPATH                            \
        || exit 1
fi

```
训练的这部和我们单语种相同，只是我们添加了`--mc-train-pv`这个参数，指定路径为我们构建的多语种的音位矢量`numpy`文件。


**Finetune部分代码**


```
finetune_dir="exp/mc_linear_finetune_de/"
if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    # finetune
    unset CUDA_VISIBLE_DEVICES
    
    if [[ $NODE == 0 && ! -f $dir/scripts.tar.gz ]]; then
        echo ""
        tar -zcf $dir/scripts.tar.gz $(readlink ctc-crf) $0
    elif [ $NODE == 0 ]; then
        echo ""
        echo "'$dir/scripts.tar.gz' already exists."
        echo "If you want to update it, please manually rm it then re-run this script."
    fi

    CUDA_VISIBLE_DEVICES=0,1,2,3,4                    \
    python3 ctc-crf/train.py --seed=0               \
        --world-size 1 --rank $NODE                 \
        --batch_size=128                            \
	     --grad-accum-fold=2                           \
        --mc-train-pv=./embedding/mul.npy            \
        --resume=$dir/ckpt/bestckpt.pt              \
        --den-lm=data/den_meta_de/den_lm.fst        \
        --mc-conf=./conf/mc_linear_finetune_de.json    \
        --trset=data/pickle/train_de.pickle         \
        --devset=data/pickle/dev_de.pickle          \
        --dir=$finetune_dir                         \
        --config=$dir/config.json                   \
        --data=data/train_de || exit 1;
fi

```
Finrtune这部分是我们对目标语言进行微调，当然你也可以不进行微调直接tesing但我们经过微调后的目标语言效果会更好。

`--grad-accum-fold` 梯度累加(默认为1)，变向增加batch_size。

`mc-conf` conf目录下json文件相关参数配置：

```
{
    "src_token": "./data/lang_phn/tokens.txt",
    "des_token": "./data/lang_phn_de/tokens.txt",
    "P": "./embedding/de.npy",
    "hdim": 640,
    "odim": 43, 
    "lr": 1e-5,
    "mode": "joinap_linear",
    "usg": "finetune"
}

```
`src_token`：原始模型单元
 
`des_token`: 训练目标语言单元

`P`: finetune目标语言音位矢量
 
`lr`: 模型微调时候学习率

`hdim`：隐含层维度

`odim`：目标语言的音素集数量

`mode`: 三种模型类型 `["flat_phone", "joinap_linear", "joinap_nonlinear"]`

`usg`: `["fientune", "zero-shot-eval", "few-shot-eval", "multi-eval", "multi-finetune-eval"]`；Finetune阶段我们默认选择`finetune` 即可 


**测试部分代码**


```
nj=20

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    for lang in de; do
        scp=data/all_ark/test_${lang}.scp
        ark_dir=$finetune_dir/decode_${lang}_test_bd_tgpr/logits
        mkdir -p $ark_dir
        CUDA_VISIBLE_DEVICES=0,3            \
        python3 ctc-crf/calculate_logits.py                 \
            --mc-conf=./conf/mc_linear_finetune_de_eval.json                   \
            --mc-train-pv=./embedding/de.npy            \
            --resume=$finetune_dir/ckpt/bestckpt.pt                     \
            --config=$finetune_dir/config.json                       \
            --nj=$nj --input_scp=$scp                       \
            --output_dir=$ark_dir                           \
            || exit 1
        
        ctc-crf/decode.sh  --stage 1 --cmd "$decode_cmd" --nj $nj --acwt 1.0 data/lang_phn_${lang}_test_bd_tgpr \
            data/test_${lang} data/all_ark/test_${lang}.ark $finetune_dir/decode_${lang}_test_bd_tgpr || exit 1
    done
fi
```
**注意**：这时`mc-train-pv`要指定我们目标语言的音位矢量

`--mc-conf` conf目录下json文件相关参数配置：

```
{
    "src_token": "./data/lang_phn/tokens.txt",
    "des_token": "./data/lang_phn_de/tokens.txt",
    "P": "./embedding/de.npy",
    "hdim": 640,
    "odim": 43,
    "lr": 1e-5,
    "mode": "joinap_linear",
    "usg": "multi-finetune-eval"
}
```
与Finetune阶段不同的是`usg`需要修改：


`multi-finetune-eval` 经过Finetune后多语言测试

`multi-eval` 没有经过Finetune多语言测试

`few-shot-eval` 经过Finetune跨语言测试

`zero-shot-eval` 没有Finetune跨语言测试

**至此我们完成**

✈-🐱‍🏍
