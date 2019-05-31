# Machine-learning相关project

## 基础工具类 

### CommonLibs
基础类  
  1. third_party: 第三方类
  2. DataHelper: 数据处理相关   
    a. QueryPreprocess: 数据预处理  
    b. SplitParagraphToSent: 处理百科数据，从段落切分为句  
    c. GenerateWordemb: 词向量读取
  3. FileIoUtils: 文件IO
  4. Metrix: 数据计算
  5. ModelUtils: 模型相关接口
  6. OtherUtils: 其他工具类接口

### Tools
工具类目录
  
### Embedding
词向量目录
    
## 项目相关目录

### Similarity

#### Pretrain
  1. biLSTM_conv_network：使用BiLSTM+CONV做隐藏层进行auto-encoder的方法
  2. fine_tuning：fine tuning代码

### NewWordExtractor: 新词发现项目

### KeywordExtraction: 关键词抽取项目
  待开发

### pre_train: 预训练项目
  使用bert进行预训练

### siamese_transformer:
使用孪生网络+transformer进行语义匹配
  1. cosine_similarity: 使用cosine函数作为目标函数，用作faiss语义召回
  2. fc_similarity: 使用全连接+softmax作为目标函数，用作语义匹配
  
