encoder_outputs：[max_len, hidden_size]，对一句话而言，每个词有一个隐藏向量。

最初的seq2seq
![](https://raw.githubusercontent.com/silentcrown/encoder_decoder/master/images/basic_seq2seq.jpg)


改进版，把encoder的输出信息也一起输入decoder。
![](https://raw.githubusercontent.com/silentcrown/encoder_decoder/master/images/seq2seqplus.jpg)
