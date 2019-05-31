使用方法：
    python NewWordExtractor.py input output > log
运行参数：
    input：     输入文件,每行是一个utf-8格式的已切词的query
    outpu：     输出的文件,每行会打印 new_word、新词出现次数、新词左边熵的大小、新词右边熵的大小（熵的最大值未1.0），\t分隔

可调参数：
    运行代码时可调的参数都直接写在NewWordExtractor.py里，可以调节的几个参数如下(直接修改代码的__init__()函数对应变量值即可)：
    1、ngram：              代表窗口大小，即由几个子query组成一个新query
    2、minPhraseNum：       新query出现最小次数，小于该次数直接过滤，可视总文本量定义该阈值
    3、linkage：            新词的凝固度，低于凝固度直接过滤。假设新词phrase由query1和query2组成，那么计算方式为 freq(phrase) / (freq(query1)*freq(query2))，freq为频率。
                            两个词组成的新词可以设1000-3000
    4、entrory_threshold    新词左右邻的信息熵，低于该值直接过滤


计算过程中大量时间都耗在切词上，因此先生成一个切过词的中间文件。这步可以用word_cut.py生成，执行方式如下：
    python word_cut.py input output > log
参数：
    input：     输入，需要为utf-8编码
    output:     输出，每行为切词后的\t分隔的词
