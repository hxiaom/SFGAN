import tornado.ioloop
import tornado.web


from tensorflow.python.client import device_lib
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Embedding
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate
from tensorflow.keras.models import Model

import datetime
import sys
import numpy as np
import random

import innvestigate
import matplotlib.pyplot as plt
import re

from utils import create_env_dir, plot_text_heatmap, get_data_plain, set_gpu, get_data_singlelabel


class MainHandler(tornado.web.RequestHandler):
    def initialize(self, embeddings_index, textcnn_model):
        self.embeddings_index = embeddings_index
        self.textcnn_model = textcnn_model

    def get(self):
        # self.write("Hello, world")
        self.render("./main.html", discipline1 = 'discipline1', discipline2 = 'discipline2', discipline3 = 'discipline3',
                    discipline1_value = 0,
                    discipline2_value = 0,
                    discipline3_value = 0)


    def post(self):
        MAX_SEQ_LENGTH = 400
        EMBEDDING_DIM = 300
        DROPOUT_RATE = 0.4
        NUM_CLASSES = 91
        proposal = self.get_argument('proposal')
        print(proposal)
        abstracts = []
        abstracts.append(proposal)

        

        # load NSFC data
        # proposal = ["To support urgent research to combat the ongoing outbreak of COVID-19, caused by the novel coronavirus SARS-CoV-2, the editorial teams at Nature Research have curated a collection of relevant articles. Our collection includes research into the basic biology of coronavirus infection, its detection, treatment and evolution, research into the epidemiology of emerging viral diseases, and our coverage of current events. The articles will remain free to access for as long as the outbreak remains a public health emergency of international concern"]

        # abstracts = proposal
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(abstracts)
        self.word_index = tokenizer.word_index

        print('Total %s unique tokens.' % len(self.word_index))

        text = proposal.lower()
        tokens = [[word for word in text_to_word_sequence(text) if re.search('[a-zA-Z]', word)]]

        # prepare text samples and their labels
        embedding_matrix = np.zeros((1, 400, 300))
        papers = []
        for i, content in enumerate(tokens):
            paper = []
            counter = 0
            for j, v in enumerate(content[:400]):
                embedding_vector = self.embeddings_index.get(v)
                if embedding_vector is not None:
                    embedding_matrix[i, j, :] = embedding_vector
                    counter = counter + 1
                paper.append(v)
            papers.append(paper)

        self.X_test = embedding_matrix

        print(self.textcnn_model.summary())

        # add record when customer make an appointment
        proposal = self.get_argument('proposal')

        
        test_result = self.textcnn_model.predict(self.X_test)
        print('result: ', test_result)
        # test_result_label = np.argmax(test_result, axis=1)
        # print('test result label')
        # print(test_result_label[0])

        # argsort method
        idxs = np.argsort(test_result, axis=1)[:,-3:]
        test_result_label = np.zeros_like(test_result)
        # test_result_label.fill(0)
        label = []
        value = []
        for i in range(1):
            for j in range(idxs.shape[1]):
                # if test_result[i][idxs[i][j]] >= 0.5:
                label.append(idxs[i][j])
                value.append(test_result[i][idxs[i][j]])
                print(test_result[0])
                print('!!!!!!!!!!')
                print(type(test_result[i][idxs[i][j]]))
                print(test_result[i][idxs[i][j]])
                test_result_label[i][idxs[i][j]] = 1

        print(label)
        code_to_index = {'A01 数学':0, 'A02 力学':1, 'A03 天文学':2, 'A04 物理学I':3, 'A05 物理学II':4,
                            'B01 合成化学':5, 'B02 催化与表面界化学':6, 'B03 化学理论与机制':7, 'B04 化学测量学':8, 'B05 材料化学与能源化学':9,
                            'B06 环境化学':10, 'B07 化学生物学':11, 'B08 化学工程与工业化学': 12, 'C01 微生物学':13, 'C02 植物学':14, 
                            'C03 生态学':15, 'C04 动物学':16, 'C05 生物物理与生物化学':17, 'C06 遗传学与生物信息学':18, 'C07 细胞生物学':19,
                            'C08 免疫学':20, 'C09 神经科学与心理学':21, 'C10 生物材料、成像与组织工程学':22, 'C11 生理学与整合生物学':23, 'C12 发育生物学与生殖生物学':24,
                            'C13 农学基础与作物学':25, 'C14 植物保护学':26, 'C15 园艺学与植物营养学':27, 'C16 林学与草地科学':28, 'C17 畜牧学':29, 
                            'C18 兽医学':30, 'C19 水产学':31, 'C20 食品科学':32, 'C21 分子生物学与生物技术':33, 'D01 地理学':34, 
                            'D02 地质学':35, 'D03 地球化学':36, 'D04 地球物理学和空间物理学':37, 'D05 大气科学':38, 'D06 海洋科学':39, 
                            'D07 环境地球科学':40, 'E01 金属材料':41, 'E02 无机非金属材料':42, 'E03 有机高分子材料':43, 'E04 矿业与冶金工程':44, 
                            'E05 机械设计与制造':45, 'E06 工程热物理与能源利用':46, 'E07 电气科学与工程':47, 'E08 建筑与土木工程':48, 'E09 水利科学与海洋工程':49, 
                            'F01 电子学与信息系统':50, 'F02 计算机科学':51, 'F03 自动化':52, 'F04 半导体科学与信息器件':53, 'F05 光学和光电子学':54, 
                            'F06 人工智能':55, 'G01 管理科学与工程':56, 'G02 工商管理':57, 'G03 经济科学':58, 'G04 宏观管理与政策':59, 
                            'H01 呼吸系统':60, 'H02 循环系统':61, 'H03 消化系统':62, 'H04 生殖系统/围生医学/新生儿':63, 'H05 泌尿系统':64, 
                            'H06 运动系统':65, 'H07 内分泌系统/代谢和营养支持':66, 'H08 血液系统':67, 'H09 神经系统和精神疾病':68, 'H10 医学免疫学':69, 
                            'H11 皮肤及其附属器':70, 'H12 眼科学':71, 'H13 耳鼻咽喉头颈科学':72, 'H14 口腔颅颌面科学':73, 'H15 急重症医学/创伤/烧伤/整形':74, 
                            'H16 肿瘤学':75, 'H17 康复医学':76, 'H18 影像医学与生物医学工程':77, 'H19 医学病原生物与感染':78, 'H20 检验医学':79,
                            'H21 特种医学':80, 'H22 放射医学':81, 'H23 法医学':82, 'H24 地方病学/职业病学':83, 'H25 老年医学':84, 
                            'H26 预防医学':85, 'H27 中医学':86, 'H28 中药学':87, 'H29 中西医结合':88, 'H30 药物学':89, 
                            'H31 药理学':90}
        index_to_code = dict([(v, k) for (k, v) in code_to_index.items()])

        # LRP methods
        # Remove softmax layer
        model_with_softmax = self.textcnn_model
        # model_without_softmax = iutils.model_wo_softmax(textcnn_model.model)
        model_without_softmax = model_with_softmax
        print(model_without_softmax.summary())

        # ['lrp', 'lrp.z', 'lrp.z_IB', 'lrp.epsilon', 'lrp.epsilon_IB', 'lrp.w_square', 
        # 'lrp.flat', 'lrp.alpha_beta', 'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB', 
        # 'lrp.alpha_1_beta_0', 'lrp.alpha_1_beta_0_IB', 'lrp.z_plus', 'lrp.z_plus_fast', 
        # 'lrp.sequential_preset_a', 'lrp.sequential_preset_b', 'lrp.sequential_preset_a_flat', 
        # 'lrp.sequential_preset_b_flat', 'lrp.rule_until_index']
        analyzer = innvestigate.create_analyzer('lrp.alpha_1_beta_0', model_without_softmax)
        # analyzer = innvestigate.create_analyzer('sa', model_without_softmax)

        x = self.X_test[0]
        x = x.reshape((1, MAX_SEQ_LENGTH, 300))  

        presm = model_without_softmax.predict_on_batch(x)[0] #forward pass without softmax
        print(presm)
        # argsort method
        idxs = np.argsort(presm, axis=0)
        print(idxs)

        neuron_list = [idxs[-1], idxs[-2], idxs[-3]]

        for i, neuron in enumerate(neuron_list):
            print(i)
            print(neuron)
            neuron = int(neuron)
            lrp_score = analyzer.analyze(x, neuron_selection=neuron)
            lrp_score = lrp_score['input_1']
            lrp_score = np.squeeze(lrp_score)
            lrp_score = np.sum(lrp_score, axis=1)
            print(lrp_score)
            print(lrp_score.shape)

            # words = ['test']*400
            plot_text_heatmap(papers[0], lrp_score.reshape(-1), title='Method: %s' % 'lrp', verbose=0)
            plt.savefig('./img/plot%s.png' % str(3-i), format='png')
            # plt.show()

        src = "img/plot1.png?" + str(random.random())
        # self.textcnn_model = None
        self.render("./result.html",
                    doc = proposal,
                    src = src,
                    discipline1 = index_to_code[label[2]], 
                    discipline2 = index_to_code[label[1]], 
                    discipline3 = index_to_code[label[0]],
                    discipline1_value = value[2],
                    discipline2_value = value[1],
                    discipline3_value = value[0])


def make_app():
    MAX_SEQ_LENGTH = 400
    EMBEDDING_DIM = 300
    DROPOUT_RATE = 0.4
    NUM_CLASSES = 91
    embeddings_index = {}
    f = open('../data/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Glove 300d contains total %s word vectors.' % len(embeddings_index))


    create_env_dir('web')
    set_gpu()

    docs_input = Input(shape=(MAX_SEQ_LENGTH, EMBEDDING_DIM))
    kernel_sizes = [3, 4, 5]
    pooled = []
    for kernel in kernel_sizes:
        conv = Conv1D(filters=100,
                    kernel_size=kernel,
                    padding='valid',
                    strides=1,
                    kernel_initializer='he_uniform',
                    activation='relu')(docs_input)
        pool = MaxPooling1D(pool_size=400 - kernel + 1)(conv)
        pooled.append(pool)
    merged = Concatenate(axis=-1)(pooled)
    flatten = Flatten()(merged)
    drop = Dropout(rate=DROPOUT_RATE)(flatten)
    x_output = Dense(NUM_CLASSES, 
                    kernel_initializer='he_uniform', 
                    activation='sigmoid', 
                    kernel_regularizer=tf.keras.regularizers.l1(0.01))(drop)

    textcnn_model = Model(inputs=docs_input, outputs=x_output)
    textcnn_model.load_weights('../weight_6.h5', by_name=True)
    # textcnn_model.load_weights('../experiments/2021-04-08/textcnn_4/checkpoints/textcnn_4-89-2.22.hdf5', by_name=True)
    print('model loaded')

    return tornado.web.Application([
        (r"/", MainHandler, {"embeddings_index":embeddings_index, "textcnn_model":textcnn_model}),
        (r"/(.*)", tornado.web.StaticFileHandler, {'path':'./'}),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()