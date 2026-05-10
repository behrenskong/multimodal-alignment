# 项目说明
- Github仓库
  - https://github.com/behrenskong/multimodal-alignment/tree/main
  - Fork自https://github.com/salesforce/LAVIS
- 主要工作目录在：
https://github.com/behrenskong/multimodal-alignment/tree/main/cross_modal_alignment

- 图片、文本embedding，meta json等非代码较大文件，已上传至百度网盘，可直接复用
https://pan.baidu.com/s/1IHt1YUyvPr-Uh11BYI6I9Q?pwd=cd38
- BLIP、BLIP2使用的transformers库版本依赖不同，运行本项目需要两个conda环境

# 一、模型梳理

## 1. CLIP
<div align="center">
    <img src="assets/clip.png" width="65%">
</div>

```python
# extract feature representations of each modality 
I_f = image_encoder(I) #[n, d_i] 
T_f = text_encoder(T) #[n, d_t]  

# joint multimodal embedding [n, d_e] 
I_e = l2_normalize(np.dot(I_f, W_i), axis=1) 
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)  
# scaled pairwise cosine similarities [n, n] 
logits = np.dot(I_e, T_e.T) * np.exp(t)  

# symmetric loss function labels = np.arange(n) 
loss_i = cross_entropy_loss(logits, labels, axis=0) 
loss_t = cross_entropy_loss(logits, labels, axis=1) 
loss = (loss_i + loss_t)/2
```

- 双塔结构，图片Encoder、文本Encoder解耦，没有任何交互，分别得到文本、图片模态的向量，然后用对比学习Loss进行优化
- 每个图片和自己配对的text拉近，和Batch内其它的text拉远
- 每个文本和自己配对的image拉近，和Batch内其它的image拉远

**纯对比学习，训练期间完全没有图文之间信息交互，text、image可能学习到的向量空间存在模态gap**

**又因为对比学习的优化过程，即使不同模态整体向量空间有gap，配对的(text, image) pair，相对于其它不配对的pair，余弦相似度是比较高的，具有下游检索任务的能力**

## 2. BLIP
### 2.1 Model
<div align="center">
    <img src="assets/blip_model.png" width="75%">
</div>

- 图片经过一个ViT，image Encoder（可学习）
- 文本经过3个不同的分支，同时适配图文检索、文本生成等多种下游任务
- 除了CLIP范式的对比学习外，引入了image-text matching任务，通过交叉注意力，在训练的时候，有图文之间向量信息交互，图文检索能力要显著强于CLIP
- 通过casual self attention引入了decoder头，具有文本生成能力，可以进行capitoning、VQA等下游任务

**局限性**
- **在LLM未发展的时候完成的paper，完全自己从头训练的文本Decoder，没有利用强大的LLM文本生成能力，导致BLIP的文本生成下游任务表现不佳**
- **虽然引入了ITC、ITM两个任务解决图文对齐的问题，但是图片、文本向量的语义空间还是有一定Gap，文本、图片模态的向量抽取还是用的单独的Encoder，训练时多走一路ITM，多走一个分类头，检索的时候，在产完Embedding一阶段检索后，用ITM分类头对top256的图文pair，进行rerank。总的来说，图文向量表示学习期间，各自走的独立的block，有独立的self attention，仍存在一定解耦，图片Encoder、文本Encoder得到的不同模态的向量空间，仍有一定gap。**

### 2.2 预训练+微调适配不同任务
以下的训练调优流程 + Loss代码，参考[BLIP Github仓库]([www.baidu.com](https://github.com/salesforce/BLIP/tree/main))

#### 2.2.0 三个Loss
- ITC：image-text contrastive (ITC) loss，和CLIP一致的对比学习
- ITM：image-text matching (ITM) loss，图文向量多走一个交叉注意力，然后对一个图文pair，做二分类任务
- LM：language modeling (LM) loss，Decoder模块，文本过Casual Mask，和图片向量cross attention，大语言模型范式的自回归生成文本的Loss

#### 2.2.1 预训练阶段
用图文pair数据集，联合优化三个模块：ITC + ITM + LM

```python
loss_ita, loss_itm, loss_lm = model(image, caption, alpha=alpha)
loss = loss_ita + loss_itm + loss_lm
```

#### 2.2.2 Retrieval微调
<div align="center">
    <img src="assets/blip_retrieval.png" width="55%">
</div>

在检索数据上进行微调，解决检索任务

ITC + ITM，训练：文本Encoder(Text Encoder) + 文本-图片交叉Encoder模块(Image-grounded Text encoder)

```python
loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)                  
loss = loss_ita + loss_itm
```

#### 2.2.3 Captioning微调
<div align="center">
    <img src="assets/blip_captioning.png" width="55%">
</div>

在caption数据上进行微调，解决captioning任务

LM，训练：Decoder模块(Image-grounded Text decoder)

```python
loss = model(image, caption) # LM Loss
```

#### 2.2.4 VQA微调
<div align="center">
    <img src="assets/blip_vqa.png" width="55%">
</div>

LM，训练：文本-图片交叉Encoder模块(Image-grounded Text encoder) + Decoder模块(Image-grounded Text decoder)

```python
loss = model(image, question, answer, train=True, n=n, weights=weights) # LM Loss
```

## 3. BLIP2
### 3.1 Model
<div align="center">
    <img src="assets/blip2.png" width="100%">
</div>

- 整个BLIP2只训练投影器Q-Former，而Image Encoder、LLM全部都是冻结的，参与训练参数很少
  - **与BLIP的核心区别，不再自己训练image encoder，不再自己训练text decoder，直接使用现有的image encoder和预训练好的LLM，自己只训练投影器从文本、图片中抽取核心信息的向量表示**
- 投影器Q-Former(Querying Transformer)，包含两个BERT分支，分别得到图片、文本的embedding
  - image这里，由原始ViT的257个token，轻量化至32个learnable queries，用少量token，从图片、文本中抽取关键信息
  - **两个BERT共享self attention，通过不同的mask适配不同的任务，算不同的Loss，这点对图文对齐的表示学习非常重要，相当于在self attention阶段，图片learnable query和文本token是在同一个向量空间下学习的，与CLIP、BLIP解耦的Encoder完全不同，共享self-attention，在向量空间层面拉近了两个模态，文本、图片可以融合在一个空间中，模态Gap问题得到很大缓解**
  - 图片、文本BERT走不同的FFN，相当于不同的语义各自学习MLP，增加不同模态的表示学习

### 3.2 Stage1 vision-language representation learning
- 第一阶段，学习image、text的向量表示能力，即学习Q-Former的两个BERT分支，能得到不同模态的向量表示，此阶段不涉及LLM
- 优化目标与BLIP一致，用图文pair数据集，联合训练3个优化目标：Image-Text Contrastive Learning (ITC)、Image-Text Matching (ITM)、Image-grounded Text Generation (ITG)
  - ITC：32个learnedqueries，和text token，先走Uni-modal Self-Attention Mask，各看各的信息不泄露，然后图片BERT这边，每个block，32个query和ViT 257个token交叉注意力，两个BERT分别得到了图片、文本的向量表示，因为有32个图片token，这里作者取max(cos(query, text token))，用余弦相似度最大的query用于算对比学习Loss和下游检索任务
  - ITM：这里做图文匹配任务，就是对(text, image) pair做二分类，此时图片、文本信息之间相互可以看见，所以走Bi-directional Self-Attention Mask，全部都能看见，然后最后得到的向量表示进二分类头，得到logits，对32个query对logits取平均，最后得到ITM Loss
  - ITG：基于图片信息的文本生成任务，此时Multi-modal Causal Self-Attention Mask，图片之间全部能看见，文本能看到所有图片和之前的文本token，然后计算LLM的自回归Loss
  - 这里有个有意思的问题，原生Q-Former在一阶段训练的时候，已经具备了text decoder的能力，其实已经能解决简单任务了，但现在有别人训好的比当前这个自己训的decoder强很多很多倍的LLM，那我们还要原始的decoder干嘛呢？所以作者在二阶段训练的时候，把Q-Former得到的向量表示，投影到LLM语义空间，拼在LLM token的前面，直接让LLM进行text generation

### 3.3 Stage2 vision-to-language generative pre-training
<div align="center">
    <img src="assets/blip2_stage2.png" width="100%">
</div>

- 冻结Image Encoder、LLM
- stage1，已经学好了Q-Former，能得到图片的32个query，然后用MLP把它投影到LLM的向量空间，放在LLM里面，作为`soft prompt`，让LLM完成text generation的任务

<div align="center">
    <img src="assets/blip2_vqa.png" width="60%">
</div>

- 对VQA这种任务，输入文本和image一起进Q-Former，然后输入文本和32个query一起进LLM

### 3.4 局限性
- **Q-Former比较难训练，现在多模态大模型很多都采用LLaVA架构，不使用Q-Former这种用learnable query去提取图片、文本关键信息的思路，LLaVA把image token直接通过projector投影到LLM token空间，微调LLM，充分利用LLM的能力，相对来说LLaVA架构成本更高**

# 二、Retrieval

## 1. 实验设置

本部分在COCO val2017上完成图文检索实验，数据集一共包含5000张图片，25014条caption。评估分为两个方向：

- Text-to-Image Retrieval：给定一条caption，在5000张图片中检索对应图片
- Image-to-Text Retrieval：给定一张图片，在25014条caption中检索对应caption；因为COCO每张图片有5条人工caption，所以topK中命中任意一个GT caption都认为检索正确

为了公平比较三个模型的图文检索能力，本实验实现方案为：**对每一个模型，先分别提取图片向量、文本向量，再计算相似度矩阵，最后自己实现Recall@K评估器**。

本实验的图文检索，只涉及图片、文本单独向量之间的余弦相似度检索，不涉及(text, image) pair，进入ITM模型的二分类头，再rerank的过程。

## 2. 代码实现思路

retrieval.py代码见本项目Github仓库
https://github.com/behrenskong/multimodal-alignment/tree/main/cross_modal_alignment/retrieval
`cross_modal_alignment/retrieval/retrieval.py`，对三个模型，实现了统一的retrieval pipeline
整体流程是：

```text
COCO annotation
-> 读取 image path 和 caption
-> 分别抽取 image embedding / text embedding
-> 计算 image-text similarity matrix
-> 分别评估 Text-to-Image 和 Image-to-Text Recall@1/5/10
```

三个模型结构不同，有各自的图片、文本向量抽取方式：

- **CLIP**：使用`CLIPModel.get_image_features`和`CLIPModel.get_text_features`，得到512维图片、文本向量，L2 normalize之后直接做点积相似度
- **BLIP**：使用`BlipForImageTextRetrieval`，图片侧取`vision_model`输出的`[CLS]`，经过`vision_proj`得到256维image embedding；文本侧取`text_encoder`输出的`[CLS]`，经过`text_proj`得到256维text embedding；最后做normalize和点积
- **BLIP2**：使用`Blip2ForImageTextRetrieval`，图片先经过冻结视觉编码器，然后32个learned query通过Q-Former从图片token里抽取信息，得到`[32, 256]`的image query embeddings；文本侧经过Q-Former文本分支得到256维text embedding；相似度计算时对32个query和文本向量的相似度取max

评估器是自己实现。Text-to-Image方向，每条caption只有一个正确图片；Image-to-Text方向，每张图片对应5条caption，只要topK里出现任意一条该图片的GT caption就算命中。

另外，BLIP原论文的检索通常还在向量余弦相似度召回top256之后，用ITM head二分类头做rerank。本实验只汇报embedding-only结果。

## 3. 量化结果

| Model | Text→Image R@1 | Text→Image R@5 | Text→Image R@10 | Image→Text R@1 | Image→Text R@5 | Image→Text R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clip-vit-base-patch32 | 30.45 | 54.79 | 66.24 | 50.02 | 74.80 | 83.16 |
| blip-itm-base-coco | 61.97 | 85.36 | 91.40 | **78.64** | 95.04 | 97.58 |
| blip2-itm-vit-g-coco | **65.18** | **87.65** | **92.99** | 77.54 | **95.08** | **97.72** |

从量化结果看，CLIP和BLIP/BLIP2之间差距非常明显。CLIP的Text→Image R@1只有30.45，而BLIP达到61.97，BLIP2达到65.18。

BLIP和BLIP2整体接近，BLIP2比BLIP要强，虽然没达到原论文指标，但可以看出不同模型的基本表现。

- CLIP只依赖纯对比学习的全局embedding，图片和文本之间没有显式交互；
- BLIP在训练时引入了额外的ITM二分类任务，模型上也有图片、文本token之间的cross-attention，图文检索能力比纯对比学习要强很多
- BLIP2在对比学习之外，也引入了ITM二分类Loss，在模型侧，Q-Former的两个BERT，更是共享了self-attention，不仅在训练时多了图片、文本之间信息交互
  - 由BLIP的单独Encoder，然后cross-attention，升级为text、image共用self-attention，直接共用自注意力语义空间，不仅增强了图文检索能力，更增强了图文向量表示的学习，表现在向量空间上，不再向CLIP、BLIP那么解耦



## 4. 分析

CLIP的主要瓶颈来自纯对比学习范式。它优化的是“正样本pair相似度高于batch内负样本pair”，并不要求模型理解图片中具体的对象关系、属性组合、计数关系等细粒度信息。比如“man behind horse”和“man riding horse”可能共享大量主物体语义，纯全局对比学习容易把它们拉得很近。

BLIP相比CLIP有明显提升，说明ITM任务确实能补足一部分纯对比学习缺失的pair-level判断能力。但在主实验的embedding-only检索中，BLIP仍然是分别抽取图片、文本向量；cross-attention matching是在二阶段ITM rerank中。BLIP的embedding能力强于CLIP，但模型侧图片、文本Encoder还是有一定解耦，信息交互发生在两个Encoder之后，仍可能存在模态gap。

BLIP2的优势主要来自Q-Former。Q-Former用32个learned queries从图片token中抽取与语言相关的视觉信息，而且第一阶段同时受到ITC、ITM、ITG三个目标约束。相比CLIP/BLIP单个全局向量，BLIP2的多query表示更容易覆盖多个物体、区域和局部语义，因此在检索上表现最好。并且因为text、image query共享Q-Former的self-attention，BLIP2的语义空间跨模态解耦问题得到很大缓解。



# 三、Captioning

## 1. 实验设置

Captioning任务只比较BLIP和BLIP2，因为CLIP本身不是生成式模型，不能直接生成caption。同样使用COCO val2017的5000张图片。每张图片生成1条caption，然后和该图片对应的5条GT captions一起送入`pycocoevalcap`评估

评估指标包括BLEU-4、CIDEr、METEOR、ROUGE-L、SPICE

声明： COCO captioning评估不是把生成caption分别和5条GT算分再简单平均，而是把5条人工标注作为multi-reference集合参与评估。这样可以缓解同一图片存在多种合理描述的问题

## 2. 代码实现思路

captioning.py代码见本项目Github仓库
https://github.com/behrenskong/multimodal-alignment/tree/main/cross_modal_alignment/captioning
`cross_modal_alignment/captioning/captioning.py`，实现了统一的caption generation和评估流程：

```text
COCO image
-> processor预处理图片
-> BLIP / BLIP2 generate
-> 每张图片保存一条prediction
-> 调用pycocoevalcap和5条GT caption做multi-reference评估
```

模型调用方式：

- **BLIP**：使用`BlipForConditionalGeneration`和`BlipProcessor`，输入图片后调用`model.generate`生成caption。这里按照HuggingFace官方unconditional image captioning方式，不额外加prompt；之前尝试使用LAVIS配置里的`a picture of `会让HF BLIP倾向生成很短的名词短语，所以最终采用无prompt设置
- **BLIP2**：使用`Blip2ForConditionalGeneration`和`AutoProcessor`，模型为`blip2-opt-2.7b-coco`，prompt使用`a photo of`，图片经过视觉编码器和Q-Former后作为视觉条件输入OPT语言模型生成文本

生成策略上，两个模型都使用确定性beam search。BLIP使用`num_beams=3, max_length=20`；BLIP2使用`num_beams=5, max_length=30`。

## 3. 量化结果

| Model | BLEU-4 | CIDEr | METEOR | ROUGE-L | SPICE |
| --- | ---: | ---: | ---: | ---: | ---: |
| blip-image-captioning-base | 0.324 | 1.033 | 0.247 | 0.525 | 0.190 |
| blip2-opt-2.7b-coco | **0.476** | **1.571** | **0.329** | **0.639** | **0.263** |

BLIP复现指标，与原始论文差距较大，但已经能说明BLIP与BLIP2相比，模型架构的缺陷。

BLIP2在所有指标上都明显高于BLIP。其中CIDEr从1.033提升到1.571，BLEU-4从0.324提升到0.476。这说明BLIP2生成的caption不仅n-gram重合度更高，在CIDEr这种更偏COCO caption语义一致性的指标上也更强。

## 4. Case分析
除了量化指标，从具体case也可以看出，BLIP2生成的caption明显优于BLIP

BLIP经常能识别主物体和大场景，但是描述偏短，BLIP2的caption更加细化

### 4.1 Case1

<div align="center">
    <img src="assets/000000000632.jpg" width="70%">
</div>

```json
  {
    "image_path": "data/coco/val2017/000000000632.jpg",
    "blip_caption": "a bedroom with a bed and a window",
    "blip2_caption": "a bedroom with a bed, dresser, mirror, and bookshelf",
    "ground_truth": [
      "Bedroom scene with a bookcase, blue comforter and window.",
      "A bedroom with a bookshelf full of books.",
      "This room has a bed with blue sheets and a large bookcase",
      "A bed and a mirror in a small room.",
      "a bed room with a neatly made bed a window and a book shelf"
    ]
  }

```

对于632这张图片
- BLIP生成“a bedroom with a bed and a window”，能抓住`床和窗户`两个主体，却缺失了细节信息
- 相比之下，BLIP2生成“a bedroom with a bed, dresser, mirror, and bookshelf”，捕获了更多细节，床、梳妆台、镜子、书架

### 4.2 Case2

<div align="center">
    <img src="assets/000000004134.jpg" width="70%">
</div>

```json
  {
    "image_path": "data/coco/val2017/000000004134.jpg",
    "blip_caption": "two men shaking hands",
    "blip2_caption": "two men shaking hands at a banquet hall",
    "ground_truth": [
     "A man is shaking hands with another man.",
     "Two men shake hands at a formal dinner gathering.",
     "Two men standing next to each other holding hands.",
     "Two men are shaking hands at a social gathering.",
     "Two men shaking hands after a dinner speech."
    ]
  }
```

对于4134这张图片
- BLIP生成“two men shaking hands”，只抓住了两个男人在握手，其他信息全丢了
- 相比之下，BLIP2生成“two men shaking hands at a banquet hall”，捕获了宴会的背景、场景信息


## 5. 分析
CLIP纯对比学习Loss，解决全局图文相似度排序，不存在文本生成能力

Captioning要求模型把视觉内容组织成自然语言，必须处理对象、属性、关系和组合语义。BLIP和BLIP2，在训练目标中显式包含LM Loss，引入自回归生成任务，BLIP自己训练了一个text decoder，BLIP2把query投影到LLM空间，直接用LLM作为Decoder

实验指标和case分析明显表明，BLIP2使用预训练LLM作为text decoder，在复杂描述和细节表达上明显优于BLIP自己训练的text decoder

# 四、Visualization

## 1. 实验设置
Visualization代码见本项目Github仓库
https://github.com/behrenskong/multimodal-alignment/tree/main/cross_modal_alignment/visualization


复用Retrieval阶段已经保存的图文向量：

- CLIP：512维image embedding / text embedding
- BLIP：256维image embedding / text embedding
- BLIP2：text embedding为256维，image embedding为32个query的256维向量；计算图文pair相似度时，和Retrieval部分保持一致，对32个query与文本向量的余弦相似度取max

为了让对比更清晰，本部分只使用UMAP做三维可视化，不再同时使用PCA、t-SNE。UMAP参数为：

```text
n_components = 3
n_neighbors = 15
min_dist = 0.1
metric = cosine
random_seed = 42
```

样本选择方式如下：

- 图文pair距离统计：使用COCO val2017全部5000张图片，每张图片取第1条人工caption，因此一共5000个image-text pair
- 模态混合可视化：固定随机种子42，从5000张图片中随机无放回选择500张图片，每张图片取第1条caption，因此一共500个pair，UMAP图里共有1000个点，其中500个image点、500个text点；三个模型使用完全相同的样本
- 语义簇可视化：使用COCO `instances_val2017.json`，对每张图片取最大面积instance的category作为主物体标签，然后统计主物体类别频次，选取最高频的两个主题：`person`和`dining table`。每个主题最多选300张图，实际选到`person=300`张，`dining table=249`张，因此一共549个pair，UMAP图里共有1098个点；三个模型也使用完全相同的样本



## 2. 图像点与文本点混合情况

这一部分使用随机选择的500个image-text pair做UMAP三维可视化。图中蓝色为image embedding，红色为text embedding。

### 2.1 CLIP

<div align="center">
    <img src="assets/clip_umap_modality.png" width="80%">
</div>

### 2.2 BLIP

<div align="center">
    <img src="assets/blip_umap_modality.png" width="80%">
</div>

### 2.3 BLIP2

<div align="center">
    <img src="assets/blip2_umap_modality.png" width="80%">
</div>

### 2.4 分析

**肉眼可见：**
  - **CLIP、BLIP都存在明显的模态gap，不同模态的点分布聚在两堆**
  - **BLIP2模态gap问题得到明显缓解，不同模态的点混在一起，BLIP2的Q-Former，文本、图片共用self-attention，应该是解决这个问题的关键优化**

为了避免只凭肉眼看图，本实验额外计算两个量化指标：

- **Modality Silhouette**：把image/text当成两个类别计算轮廓系数。越接近1，说明两个模态分得越开；越接近0，说明两个模态越混合
- **Cross-modal NN Rate**：对每个点找最近邻，如果最近邻来自另一个模态，就记为cross-modal nearest neighbor。比例越高，说明image和text越容易混在一起

| Model | Original Modality Silhouette | Original Cross-modal NN Rate | UMAP Modality Silhouette | UMAP Cross-modal NN Rate |
| --- | ---: | ---: | ---: | ---: |
| CLIP | 0.400 | 0.00% | 0.872 | 0.00% |
| BLIP | 0.280 | 1.10% | 0.850 | 0.00% |
| BLIP2 | **0.277** | **8.80%** | **0.013** | **9.60%** |

CLIP的image点和text点在UMAP图中明显分开，UMAP后的modality silhouette达到0.872，cross-modal nearest neighbor为0。这说明CLIP虽然能做检索，但不同模态在表征空间中仍存在明显modality gap。

BLIP在原始高维空间中的modality silhouette低于CLIP，说明BLIP的图文空间确实更接近。但UMAP可视化后，image和text仍然明显分成两团。这和BLIP结构是吻合的：BLIP虽然训练了ITM任务，但embedding-only检索阶段仍然是图片Encoder和文本Encoder分别抽取向量；真正的图文交互主要发生在ITM rerank阶段。

BLIP2在UMAP中的modality silhouette只有0.013，几乎接近0，image/text点更容易混在一起。这和Q-Former结构有关：BLIP2第一阶段中，query和text token在Q-Former内部通过不同mask共享self-attention参数，图片query和文本token更早进入同一个语义建模空间，因此模态gap明显小于CLIP和BLIP。

## 3. 不同模型图文pair向量距离分析
### 3.1 降维后的坐标
从上面三张图的坐标绝对值可以明显看出，CLIP x取值在[-5,15]，BLIP x取值在[-2,10]，BLIP2 x取值在[2,7]

**肉眼可见，对于相同的image-text pair，文本、图片向量之间的距离： BLIP2 < BLIP < CLIP**

### 3.2 原始高维embedding空间余弦相似度
不看降维后的三维距离，而是在原始高维embedding space中直接计算同一图文pair的余弦相似度。

对于CLIP、BLIP，计算方式是：

```text
score(image, text) = cosine(image_embedding, text_embedding)
```

对于BLIP2，因为一张图片对应32个query向量，所以计算方式是：

```text
score(image, text) = max_i cosine(image_query_i, text_embedding), i = 1 ... 32
```

这个指标越高，说明模型把同一张图片和对应caption拉得越近。

| Model | Pair Cosine Mean | Median | Std | P25 | P75 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP | 0.303 | 0.303 | 0.034 | 0.281 | 0.325 |
| BLIP | 0.472 | 0.475 | 0.042 | 0.447 | 0.500 |
| BLIP2 | **0.499** | **0.504** | 0.038 | **0.479** | **0.524** |

从同一图文pair的平均余弦相似度看，CLIP明显最低，BLIP和BLIP2明显更高。这个结果和Retrieval实验是一致的：CLIP虽然也学习了一个共享检索空间，但纯对比学习主要优化的是相对排序，不一定让两个模态在绝对空间中高度融合。

BLIP相比CLIP提升明显，说明ITC+ITM的训练目标确实增强了图文对齐能力。BLIP2最高，主要是因为Q-Former的query被训练成从图像中抽取和文本最相关的视觉信息，而且32个query可以覆盖多个局部区域，最终用max similarity选出与文本最匹配的query。

## 4. 语义簇分析

### 4.1 CLIP

<div align="center">
    <img src="assets/clip_semantic_umap.png" width="80%">
</div>

### 4.2 BLIP

<div align="center">
    <img src="assets/blip_semantic_umap.png" width="80%">
</div>

### 4.3 BLIP2

<div align="center">
    <img src="assets/blip2_semantic_umap.png" width="80%">
</div>

### 4.4 分析

这一部分分析“主物体语义”是否能在embedding space中形成簇。具体做法是：

```text
COCO instances标注
-> 每张图片取最大面积instance作为主物体
-> 统计最高频两个主物体类别
-> 选取person和dining table两个类别
-> person选300张，dining table选249张
-> 每张图片取第1条caption
-> 对image/text embedding一起做UMAP三维可视化
```
#### 4.4.1 肉眼分析
- CLIP同一个语义的点，全聚在一起，近乎球形，存在明显语义簇，现在是选取person、dining table两个类别，向量分布近乎球形，那么说明这些向量学习的近乎一致，很有可能是CLIP的向量关注于person、table等主物体，而形成了明显的聚集，细节性语义学习不到位，导致只有聚集，而学不到其他细致性差异
- BLIP的情况与CLIP类似，但比CLIP稍微散一点，但也存在明显语义簇，可能也是细节性语义学习不到位，导致只有聚集，而学不到其他细致性差异
- BLIP2的点就比较分散，但相同语义的点有一点点聚集趋势，这算是正常现象，相同语义的点相对离得近，但又不是完全聚在一起，BLIP2的向量学习捕获了丰富信息，不止关注主物体、场景



#### 4.4.2 量化指标辅助分析

- **Semantic Silhouette**：把`person`和`dining table`当作两个语义类别，计算轮廓系数。越高说明语义簇分得越清楚
- **kNN Purity**：对每个点找最近的10个邻居，统计邻居中和自己语义类别相同的比例。越高说明局部邻域语义更一致

| Model | UMAP Image Semantic Silhouette | UMAP Text Semantic Silhouette | UMAP All Semantic Silhouette | UMAP Image kNN Purity | UMAP Text kNN Purity |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP | 0.332 | 0.326 | 0.058 | 85.23% | 84.74% |
| BLIP | **0.395** | **0.389** | 0.096 | **87.69%** | 84.48% |
| BLIP2 | 0.316 | 0.342 | **0.271** | 84.79% | **85.37%** |

如果只看单模态内部的语义聚类，BLIP的image semantic silhouette最高，image kNN purity也最高，说明BLIP的image embedding对`person`和`dining table`这两个主物体类别有比较强的区分能力。但是主物体相关的点完全聚在一起，也不是好事，说明向量学习只聚焦在核心物体上，学不到细节。

但如果把image和text点放在一起看，BLIP2的All Semantic Silhouette最高，达到0.271，而CLIP和BLIP分别只有0.058和0.096。这说明BLIP2不仅能在单个模态内部形成语义簇，更重要的是，它的image点和text点可以沿着同一个语义方向聚在一起。也就是说，在BLIP2空间里，`person`相关的图片和caption更可能靠近同一个区域，`dining table`相关的图片和caption也更可能靠近同一个区域。

CLIP和BLIP虽然在单模态内部也能形成一定语义簇，但由于image/text模态本身分离明显，两个模态合在一起之后，语义结构会被modality gap干扰。图像点可能按主物体形成一簇，文本点也按语义形成一簇，但image-person和text-person未必自然混在一起。

## 5. 小结

从Visualization部分可以得到三个结论：

- **同一图文pair距离**：BLIP和BLIP2明显高于CLIP，BLIP2最高。这说明相比纯对比学习，加入matching/generation目标，以及Q-Former这种query-based结构，能把正样本图文pair拉得更近
- **模态混合程度**：CLIP和BLIP的image/text点仍然明显分离，BLIP2的image/text点混合最好。这说明CLIP式纯对比学习虽然能解决检索排序问题，但不能完全消除modality gap；BLIP的ITM交互发生在rerank阶段，embedding-only空间仍有模态分离；BLIP2的共享Q-Former self-attention更有利于跨模态融合
- **语义簇结构**：CLIP和BLIP在单模态内部能形成一定主物体簇，但跨模态合并后语义结构容易被模态gap打散；BLIP2的跨模态语义簇更明显，说明其表征空间更适合把图片语义和文本语义统一起来。
    - 另外CLIP、BLIP语义簇现象过于明显，反而表明其向量表示学习语义不丰富，过度依赖主物体

# 五、Nearest-Neighbor Case Study
Case分析代码见本项目Github仓库
https://github.com/behrenskong/multimodal-alignment/tree/main/cross_modal_alignment/case

## 1. 文本最近邻图片
### 1.1 case1
```json
caption = "There is an old toilet sitting under a sign for this picture."
```
CLIP文搜图，只关注在了主物体toilet，忽略了sign，而BLIP、BLIP2能正确捕获

CLIP:
<div align="center">
    <img src="assets/case1_clip.png" width="100%">
</div>

BLIP:
<div align="center">
    <img src="assets/case1_blip.png" width="100%">
</div>

BLIP2:
<div align="center">
    <img src="assets/case1_blip2.png" width="100%">
</div>


### 1.2 case2
```json
caption = "Two men hold up a man holding a soccer ball and all the men have on orange shirts."
```
CLIP文搜图，只关注在了主物体man和soccer，忽略了hold up

GT:
<div align="center">
    <img src="assets/case2_gt.png" width="30%">
</div>


CLIP:
<div align="center">
    <img src="assets/case2_clip.png" width="100%">
</div>

BLIP:
<div align="center">
    <img src="assets/case2_blip.png" width="100%">
</div>

BLIP2:
<div align="center">
    <img src="assets/case2_blip2.png" width="100%">
</div>


### 1.3 case3
```json
caption = "A yellow and green object with a brown bird perched on top of it."
```

CLIP rank=1229, BLIP rank=4, BLIP2 rank=3
- CLIP完全关注在主物体鸟，忽略了另一个object要求
- BLIP甚至很多case忽略了主物体鸟，而只关注在了object
- BLIP、BLIP2把hard negative排在前面，都是鸟站在绿色物体上，但是不匹配要求的yellow and green
- 对于yellow and green object这种抽象的描述，几个模型表现的都不是完美，尤其是CLIP完全无法正确检索

GT:
<div align="center">
    <img src="assets/case3_gt.png" width="30%">
</div>


CLIP:
<div align="center">
    <img src="assets/case3_clip.png" width="100%">
</div>

BLIP:
<div align="center">
    <img src="assets/case3_blip.png" width="100%">
</div>

BLIP2:
<div align="center">
    <img src="assets/case3_blip2.png" width="100%">
</div>

### 1.4 case4
```json
caption = "A person looking at their cell phone at another person taking a picture."
```

- CLIP完全关注在手机、拍照，忽略了嵌套关系，把人拍照排在靠前
- BLIP、BLIP2正确匹配
- BLIP、BLIP2也会把“人拍照”这种语义相近hard negative排的很靠前。

GT:
<div align="center">
    <img src="assets/case4_gt.png" width="30%">
</div>


CLIP:
<div align="center">
    <img src="assets/case4_clip.png" width="100%">
</div>

BLIP:
<div align="center">
    <img src="assets/case4_blip.png" width="100%">
</div>

BLIP2:
<div align="center">
    <img src="assets/case4_blip2.png" width="100%">
</div>


## 2. 图片最近邻文本
### 2.1 case1
Image:
<div align="center">
    <img src="assets/case5_gt.png" width="40%">
</div>


Groud Truth:
```json
A person looking at their cell phone at another person taking a picture.
A person holds a smartphone taking a picture of a bottle and glasses on the table
A person holding up a cell phone by a lit candle.
A woman takes a picture of the beer on the table
A person showing a picture on their cellular phone.
```


- CLIP图搜文，肉眼可见，CLIP搜出来的文本只关注在主物体
  - 搜出来的文本依赖人、手机、桌子主物体，捕获不到细节的动作、图片的细节
  - top的都是人拿着手机在桌子旁拍照，这是符合的，但是没捕获到图片的嵌套关系
  - 第9、第10名的text，错误地捕获了两个手机，而不是嵌套拍照的信息，hard negative排的很前
- BLIP、BLIP2图搜文，groud truth的caption全部正确召回

CLIP top10:
```json
A person holds a smartphone taking a picture of a bottle and glasses on the table
A hand holding a smart phone above a wooden table.
A person holding up a cell phone by a lit candle.
A woman holding a smart phone at a table.
A woman sitting in front of several glasses, talking on her phone.
Two people sitting at a table with beverages on it and the woman holding her phone.
A woman takes a picture of the beer on the table
Two sets of hands are each holding a cell phone while another hand in the background is holding a glass.
Two cell phones of differing quality are set side by side on a table.
Two cell phones are sitting next to each other on a table.
```


BLIP top10:
```json
A person holds a smartphone taking a picture of a bottle and glasses on the table
A person holding up a cell phone by a lit candle.
A person showing a picture on their cellular phone.
A picture of a person holding a phone displaying a picture of a person holding a phone twice
A person looking at their cell phone at another person taking a picture.
A person using a photo filter holding a samsung cell phone.
A woman takes a picture of the beer on the table
A hand holding a smart phone above a wooden table.
A woman holding a smart phone at a table.
A man holding up his smart phone to take a picture.
```



BLIP2 top10:
```json
A hand holding a smart phone above a wooden table.
A woman takes a picture of the beer on the table
A person holding up a cell phone by a lit candle.
A person holds a smartphone taking a picture of a bottle and glasses on the table
A person showing a picture on their cellular phone.
A person using a photo filter holding a samsung cell phone.
Person taking a photo of a black cellphone.
The hand is holding an iPhone for the picture.
A person looking at their cell phone at another person taking a picture.
A person's hand holding up an active smartphone.
```


## 3. 组合泛化
### 3.1 case1
```json
"a man riding a horse"
```

<div align="center">
    <img src="assets/case6.png" width="100%">
</div>

```json
"a man standing next to a horse"
```

<div align="center">
    <img src="assets/case7.png" width="100%">
</div>


- 可以看出
  - "a man riding a horse"，CLIP、BLIP、BLIP2都表现很好
  - "a man standing next to a horse"，BLIP、BLIP2能正确检索，而CLIP还是会检索到“人骑马”，关注在了人和马主物体上，忽略了细节的关系

# 六、总结
## Retrieval
纯contrastive learning的瓶颈不是不能做检索，而是它主要学习全局相似度排序，容易依赖主物体和场景共现关系；对于属性、关系、计数、组合语义这些细粒度对齐，纯全局对比目标约束不足

BLIP通过ITM和LM增强pair-level判断和生成能力

BLIP2进一步通过Q-Former，文本、图片共享self-attention，在一个向量空间学习，使图文向量在表示空间上更容易融合。且BLIP2多个query设计，使得BLIP2的向量表示能捕获语义细节

## Captioning
纯contrastive learning的CLIP不具备生成能力

BLIP通过LM训练生成能力，BLIP自己训练的decoder，有简单的caption生成能力，但比BLIP2使用LLM作为decoder，生成质量要差得远

BLIP2在一阶段训好Q-Former后，又二阶段训练，将Q-Former投影到LLM空间，使用LLM作为decoder，文本生成能力比BLIP强得多
