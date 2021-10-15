## 机器翻译与数据集

数据集中的每一行都是制表符分隔文本序列对，序列对由英文文本序列和翻译后的法语文本序列组成。每个文本序列可以是一个句子，也可以是包含多个句子的一个段落。英语是 *源语言*（source language），法语是 *目标语言*（target language）。

### 获取数据

```python
import os
import torch
from d2l import torch as d2l

def read_data_nmt():
    """载入“英语－法语”数据集。"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', 
             encoding='utf-8') as f:
        return f.read()
raw_text = read_data_nmt()
print(raw_text[:75])
#Go.	Va !
#Hi.	Salut !
#Run!	Cours !
#Run!	Courez !
#Who?	Qui ?
#Wow!	Ça alors !
```

### 文本预处理

将大写字母替换成小写，在单词和标点之间空格

```python
def preprocess_nmt(text):
    def no_space(char,pre_char): 
        # 这个函数是输入字符和该字符的前一个字符，如果该字符是标点符号，
        #且前一符号没有空格，就返回true，否则返回false。
        return char in set(',.!?') and pre_char != ' '
    text = text.replace('\t202f',' ').replace('\xa0',' ').lower() 
    # utf-8中半角全角都替换成空格，并且大写转换成小写
    out = [' '+char if i > 0 and no_space(char,text[i-1]) else char
          for i, char in enumerate(text)]
    return ''.join(out)
text = process_nmt(raw_text)
```

### 词元化

```python
def tokenize_nmt(text,num_examples=None):
    source,target=[],[]
    for i,line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t') # 将每条样本划分成source和target
        if len(parts) == 2:
            source.append(parts[0].split(' '))# 取出source，再以空格分成词元
            target.append(parts[1].split(' '))
    return source,target
source, target = tokenize_nmt(text)
```

### 构建词汇表

为缓解词汇表过大，将出现次数少于2次的低频率词元视为相同的未知（“<unk>”）词元。除此之外，我们还指定了额外的特定词元，例如在小批量时用于将序列填充到相同长度的填充词元（“<pad>”），以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）

```python
src_vocab = d2l.Vocab(source,min_freq=2,
                      reversed_token=['<pad>', '<bos>', '<eos>'])
```

### 加载数据集

特定的“<eos>”词元添加到所有序列的末尾，用于表示序列的结束。还记录了每个文本序列的长度

```python
# 该函数输入一个序列样本，然后返回做好填充或截断的序列样本
def truncate_pad(line,num_steps,padding_token):
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[padding_token]*(num_steps-len(line)) #填充

def build_array_nmt(lines,vocab,num_steps):
    lines = [vocab[l] for l in lines]
    lines = [l+[vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array,valid_len
```

### 训练模型

**就是传入batch_size，num_steps等参数，构建一个数据迭代器。该函数先获取原始文本，然后对文本做预处理，处理完后把文本词元化，然后送到之前写好的词汇表构建函数（该函数需要词元化的文本，需要新添加的词元列表等），然后就获得了文本的词汇表，就可以将文本再处理，使其每个序列文本的时间步统一（这里将每个句子作为序列样本，因此需要截断或填充，使其长度对齐）。然后将序列长度统一好的文本，输入到之前写好的数据迭代器函数（该函数需要传入可分批的文本以及batch_size），最后获得输出。**

```python
def load_data_nmt(batch_size,num_steps,num_examples=600):
    text = preprocess_num(read_data_nmt())
    source,target = tokenize_nmt(text,num_examples)
    src_vocab = d2l.Vocab(source,min_freq=2,
                      reversed_token=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target,min_freq=2,
                      reversed_token=['<pad>', '<bos>', '<eos>'])
    src_arrays,src_valid_len = build_array_nmt(source,src_vocab,num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter,src_vocab,tgt_vocab
```



## 编码器-解码器架构

<img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211014155434202.png" alt="image-20211014155434202" style="zoom:67%;" />

### 编码器

```python
from torch import nn
class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)
    def forward(self,X,*args):
        raise NotImplementedError
```

### 解码器

```python
class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
    def ini_state(self,enc_outputs,*args):
        raise NotImplementedError
    def forward(self, X, state):
        raise NotImplementedError
```

### 合并编码器和解码器

```python
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,**kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder = Encoder
        self.decoder = Decoder
    def forward(self,enc_X,dec_X,*args):
        enc_outputs = self.encoder(enc_X,*args)
        dec_state = self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_X,dec_state)
```



## seq2seq模型

<img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211014155847402.png" alt="image-20211014155847402" style="zoom: 80%;" />

### 编码器

编码器用nn中的Embedding层来将词表中的词元转换为特征向量（类似于前面用到的one-hot），这里用了两层的gru来做编码器，编码器的前向传播输出有两个

```python
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,
                dropout=0,**kwargs):
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)
    def forward(self,X,*args):
        # 输入X的形状变为(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在rnn模型中做维度交换，把时间步放在第一维
        X = X.permute(1,0,2)
        output,state = self.rnn(X)
        # `output`的形状: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state`的形状: (`num_layers`, `batch_size`, `num_hiddens`)
        return output,state
```

### 解码器

直接使用编码器最后一个时间步的隐藏状态来初始化解码器的隐藏状态。这就要求使用循环神经网络实现的编码器和解码器具有相同数量的层和隐藏单元

```python
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,
                 dropout=0,**kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn = nn.GRU(embed_size+num_hiddens,num_hiddens,
                          num_layers,dropout=dropout)
        self.dense = nn.Linear(num_hiddens,vocab_size) #获取输出结果
    def ini_state(self,enc_outputs,*args):
        return enc_outputs[1] # 用编码器最后一个时间步的所有层的隐层状态来初始化
    def forward(self, X, state):
        # 此时X的形状为num_step,batch_size,embeding_size
        X = self.embedding(X).permute(1,0,2) 
        context = state[-1].repeat(X[0],1,1)
        # state[-1]的形状(1, `batch_size`, `num_hiddens`)
        # context的形状(num_step, `batch_size`, `num_hiddens`)
        #将context和X在第三维拼接
        #最后就是(num_step, batch_size, num_hiddens+embed_size)
        X_and_context = torch.concat((X,context),2)
        output,state = self.rnn(X_and_context)
        output = self.dense(output).premute(1,0,2)
        return output, state
    
#解码器的输出形状变为（批量大小, 时间步数, 词表大小)，其中张量的最后一个维度存储预测的词元分布。
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
output.shape, state.shape
#(torch.Size([4, 7, 10]), torch.Size([2, 4, 16]))
```

### 损失函数

应该将填充词元的预测排除在损失函数的计算之外。为此，使用下面的 `sequence_mask` 函数[**通过零值化屏蔽不相关的项**]，以便后面任何不相关预测的计算都是与零的乘积，结果都等于零。

```python
def sequence_mask(X,valid_len,value=0):
    maxlen = X.size(1) # 获取时间步大小
    # 下面这里先做了升维然后用的广播机制
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
# [:, None]是升一个维度的操作,在none出现的位置升一维
#mask, torch.arange((maxlen), valid_len[:, None]三个的值分别是以下的样子
#(tensor([[ True, False, False],
#        [ True,  True, False]]),
# tensor([0., 1., 2.]),
# tensor([[1],
#        [2]]))
```

现在，可以**通过扩展softmax交叉熵损失函数来遮蔽不相关的预测**

```python
class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,pred,label,valid_len):
        # `pred` 的形状：(`batch_size`, `num_steps`, `vocab_size`)
    	# `label` 的形状：(`batch_size`, `num_steps`)
   	 	# `valid_len` 的形状：(`batch_size`,)
        #先生成一个全1矩阵，然后用上面的方法，把填充部分的权重设为0
        weights = torch.ones_like(label)
        weights = sequence_mask(weights,valid_len)
        self.reduction='none'
        # pytorch需要把预测的维度放在中间，因此先做一个维度交换
        unweight_loss = super().forward(pred.permute(0,2,1),label)
        weighted_loss = (unweight_loss*weights).mean(dim=1)
        return weighted_loss
```

### 训练

**特定的序列开始词元（“<bos> ”）和原始的输出序列（不包括序列结束词元“<eos>”）拼接在一起作为解码器的输入**，这被称为 *教师强制（teacher forcing），因为原始的输出序列（词元的标签）被送入解码器。或者也可以将来自**上一个时间步的 预测 得到的词元作为解码器的当前输入。**

```python
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    # 上面这些都是一样的操作：参数初始化，模型移到gpu，定义优化器、损失
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metirc = d2l.Accumulator(2) # 训练损失总和，词元数量
        for batch in data_iter:
            X,X_valid_len,Y,Y_valid_len = [x.to(device) for x in batch]
            #tgt_vocab['<bos>']*Y.shape[0]先复制batch_size个<bos>
            #然后在reshape成列数为1，方便拼接（也就是大小为(batch_size,1)）
            bos = torch.tensor([tgt_vocab['<bos>']*Y.shape[0],
                                device=device]).reshape(-1,1)
            # 在哪个维度上拼接，就是把两个矩阵这个维度上的数进行相加
            dec_input = torch.concat([bos,Y[:,:-1]],1)
            #EncoderDecoder做前向的时候就是需要编码和解码的输入数据，其他的如根据情况来定
            #如X_valid_len在EncoderDecoder时需要，而Y_valid_len在算loss的时候才需要
            Y_hat,_ = net(X,dec_input,X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
        f'tokens/sec on {str(device)}')

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
```

### 预测

为了采用一个接着一个词元的方式预测输出序列，每个解码器当前时间步的输入都将来自于前一时间步的预测词元。序列开始词元（“<bos>”）在初始时间步被输入到解码器中。当输出序列的预测遇到序列结束词元（“<eos>”）时，预测就结束了。

```python
def predict_seq2seq(net,src_sentence,src_vocab,tgt_vocab,num_steps
                   device, save_attention_weights=False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']] # 将原始句子划分成词元并加上<eos>
    # 计算出valid_len并做填充或裁剪，和前面训练的差不多
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    #unsqueeze时添加维度，这里是对张量在第0维加个维度，即batch的维度
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y,dec_state = net.decoder(dec_X,dec_state)
        dec_X = Y.argmax(dim=2) #Y的最后一维就是输出的类别，这一维的大小是vocab_size
        # 这里采用的是贪心搜索法，即将预测最高可能性的词元，作为解码器下一时间步的输入
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        #squeeze将输入张量形状中dim=0维的1 去除并返回，实现降维
    	# 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq
```

### 预测序列的评估（BLEU）

bleu定义如下：

<img src="C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211014185518053.png" alt="image-20211014185518053" style="zoom: 67%;" />

其中 𝑘 是用于匹配的最长的 𝑛 元语法。

```python
def bleu(pred_seq, label_seq, k):
    """计算 BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))  # 计算出前半部分
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1 #匹配到就减去，防止重复
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```



## 束搜索

束搜索介于穷举搜索和贪心搜索之间。第一次选取前K个大的值，然后一直向下搜索，最终得到k个结果

![image-20211014155543908](C:\Users\User\AppData\Roaming\Typora\typora-user-images\image-20211014155543908.png)

前面预测的模型中，使用的是贪心，速度快，但准确低。可以改用束搜索，在准确度和速度之间有个权衡。