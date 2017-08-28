self.graph = tf.Graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            # 首先定义两个用作输入的占位符，分别输入输入集(train_inputs)和标签集(train_labels)
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])   
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # 词向量矩阵，初始时为均匀随机正态分布
            self.embedding_dict = tf.Variable(
                tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0)
            )

            # 模型内部参数矩阵，初始为截断正太分布
            self.nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                                              stddev=1.0/math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))

            # 将输入序列向量化，具体可见我的【常用函数说明】那一篇
            embed = tf.nn.embedding_lookup(self.embedding_dict, self.train_inputs) # batch_size

            # 得到NCE损失(负采样得到的损失)
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights = self.nce_weight,  # 权重
                    biases = self.nce_biases,   # 偏差
                    labels = self.train_labels, # 输入的标签
                    inputs = embed,             # 输入向量
                    num_sampled = self.num_sampled, # 负采样的个数
                    num_classes = self.vocab_size # 类别数目
                )
            )

            # tensorboard 相关
            tf.scalar_summary('loss',self.loss)  # 让tensorflow记录参数

            # 根据 nce loss 来更新梯度和embedding，使用梯度下降法(gradient descent)来实现
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)  # 训练操作

            # 计算与指定若干单词的相似度
            self.test_word_id = tf.placeholder(tf.int32,shape=[None])
            vec_l2_model = tf.sqrt(  # 求各词向量的L2模
                tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True)
            )

            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.scalar_summary('avg_vec_model',avg_l2_model)

            self.normed_embedding = self.embedding_dict / vec_l2_model
            # self.embedding_dict = norm_vec # 对embedding向量正则化
            test_embed = tf.nn.embedding_lookup(self.normed_embedding, self.test_word_id)
            self.similarity = tf.matmul(test_embed, self.normed_embedding, transpose_b=True)

            # 变量初始化操作
            self.init = tf.global_variables_initializer()
            # 汇总所有的变量记录
            self.merged_summary_op = tf.merge_all_summaries()
            # 保存模型的操作
            self.saver = tf.train.Saver()





def train_by_sentence(self, input_sentence=[]):
        #  input_sentence: [sub_sent1, sub_sent2, ...]
        # 每个sub_sent是一个单词序列，例如['这次','大选','让']
        sent_num = input_sentence.__len__()
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence: # 输入有可能是多个句子，这里每个循环处理一个句子
            for i in range(sent.__len__()): # 处理单个句子中的每个单词
                start = max(0,i-self.win_len)   # 窗口为 [-win_len,+win_len],总计长2*win_len+1
                end = min(sent.__len__(),i+self.win_len+1)
                # 将某个单词对应窗口中的其他单词转化为id计入label，该单词本身计入input
                for index in range(start,end): 
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent[index])
                        if not (input_id and label_id): #　如果单词不在词典中，则跳过
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if len(batch_inputs)==0:　#　如果标签集为空，则跳过
            return
        batch_inputs = np.array(batch_inputs,dtype=np.int32)
        batch_labels = np.array(batch_labels,dtype=np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels.__len__(),1])

        #　生成供tensorflow训练用的数据
        feed_dict = {   
            self.train_inputs: batch_inputs,
            self.train_labels: batch_labels
        }
        # 这句操控tf进行各项操作。数组中的选项，train_op等，是让tf运行的操作，feed_dict选项用来输入数据
        _, loss_val, summary_str = self.sess.run([self.train_op,self.loss,self.merged_summary_op], feed_dict=feed_dict)

        # train loss，记录这次训练的loss值
        self.train_loss_records.append(loss_val)
        # self.train_loss_k10 = sum(self.train_loss_records)/self.train_loss_records.__len__()
        self.train_loss_k10 = np.mean(self.train_loss_records) # 求loss均值
        if self.train_sents_num % 1000 == 0 :
            self.summary_writer.add_summary(summary_str,self.train_sents_num)
            print("{a} sentences dealed, loss: {b}"
                  .format(a=self.train_sents_num,b=self.train_loss_k10))

        # train times
        self.train_words_num += batch_inputs.__len__()
        self.train_sents_num += input_sentence.__len__()
        self.train_times_num += 1