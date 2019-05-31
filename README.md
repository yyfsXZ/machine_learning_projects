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
  7. database: 数据库相关操作接口

### Tools
工具类目录
  
### Embedding
词向量目录
    
## 项目相关目录

### Similarity

#### Pretrain 预训练思路实现代码
  1. biLSTM_conv_network：使用BiLSTM+CONV做隐藏层进行auto-encoder的方法
  2. fine_tuning：fine tuning代码

#### SiameseNetwork 使用孪生网络的end-to-end模型
  1. cosine_similarity：输出层的目标函数使用cosine进行训练，主要用于基于向量相似度的语义召回
  2. fc_similarity: 相似度模型

### NewWordExtractor: 新词发现项目

### KeywordExtraction: 关键词抽取项目
  待开发

  
