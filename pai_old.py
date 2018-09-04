#/usr/bin/env python
#coding=utf-8
#===================================================================================
#                                      传统方法
#===================================================================================
import numpy as np
import pandas as pd
import re
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, KFold
import gensim
try:
    import jieba_fast as jieba
except Exception as e:
    import jieba

try:
    print(model_dir)
    test_size = 0.025
    online=True
except:
    model_dir = "pai_model/"
    test_size = 0.05
    online=False

new_words = "支付宝 付款码 二维码 收钱码 转账 退款 退钱 余额宝 运费险 还钱 还款 花呗 借呗 蚂蚁花呗 蚂蚁借呗 蚂蚁森林 小黄车 飞猪 微客 宝卡 芝麻信用 亲密付 淘票票 饿了么 摩拜 滴滴 滴滴出行".split(" ")
for word in new_words:
    jieba.add_word(word)

star = re.compile("\*+")
if False:
    stops = ["、","。","〈","〉","《","》","一","一切","一则","一方面","一旦","一来","一样","一般","七","万一","三","上下","不仅","不但","不光","不单","不只","不如","不怕","不惟","不成","不拘","不比","不然","不特","不独","不管","不论","不过","不问","与","与其","与否","与此同时","且","两者","个","临","为","为了","为什么","为何","为着","乃","乃至","么","之","之一","之所以","之类","乌乎","乎","乘","九","也","也好","也罢","了","二","于","于是","于是乎","云云","五","人家","什么","什么样","从","从而","他","他人","他们","以","以便","以免","以及","以至","以至于","以致","们","任","任何","任凭","似的","但","但是","何","何况","何处","何时","作为","你","你们","使得","例如","依","依照","俺","俺们","倘","倘使","倘或","倘然","倘若","借","假使","假如","假若","像","八","六","兮","关于","其","其一","其中","其二","其他","其余","其它","其次","具体地说","具体说来","再者","再说","冒","冲","况且","几","几时","凭","凭借","则","别","别的","别说","到","前后","前者","加之","即","即令","即使","即便","即或","即若","又","及","及其","及至","反之","反过来","反过来说","另","另一方面","另外","只是","只有","只要","只限","叫","叮咚","可","可以","可是","可见","各","各个","各位","各种","各自","同","同时","向","向着","吓","吗","否则","吧","吧哒","吱","呀","呃","呕","呗","呜","呜呼","呢","呵","呸","呼哧","咋","和","咚","咦","咱","咱们","咳","哇","哈","哈哈","哉","哎","哎呀","哎哟","哗","哟","哦","哩","哪","哪个","哪些","哪儿","哪天","哪年","哪怕","哪样","哪边","哪里","哼","哼唷","唉","啊","啐","啥","啦","啪达","喂","喏","喔唷","嗡嗡","嗬","嗯","嗳","嘎","嘎登","嘘","嘛","嘻","嘿","四","因","因为","因此","因而","固然","在","在下","地","多","多少","她","她们","如","如上所述","如何","如其","如果","如此","如若","宁","宁可","宁愿","宁肯","它","它们","对","对于","将","尔后","尚且","就","就是","就是说","尽","尽管","岂但","己","并","并且","开外","开始","归","当","当着","彼","彼此","往","待","得","怎","怎么","怎么办","怎么样","怎样","总之","总的来看","总的来说","总的说来","总而言之","恰恰相反","您","慢说","我","我们","或","或是","或者","所","所以","打","把","抑或","拿","按","按照","换句话说","换言之","据","接着","故","故此","旁人","无宁","无论","既","既是","既然","时候","是","是的","替","有","有些","有关","有的","望","朝","朝着","本","本着","来","来着","极了","果然","果真","某","某个","某些","根据","正如","此","此外","此间","毋宁","每","每当","比","比如","比方","沿","沿着","漫说","焉","然则","然后","然而","照","照着","甚么","甚而","甚至","用","由","由于","由此可见","的","的话","相对而言","省得","着","着呢","矣","离","第","等","等等","管","紧接着","纵","纵令","纵使","纵然","经","经过","结果","给","继而","综上所述","罢了","者","而","而且","而况","而外","而已","而是","而言","能","腾","自","自个儿","自从","自各儿","自家","自己","自身","至","至于","若","若是","若非","莫若","虽","虽则","虽然","虽说","被","要","要不","要不是","要不然","要么","要是","让","论","设使","设若","该","诸位","谁","谁知","赶","起","起见","趁","趁着","越是","跟","较","较之","边","过","还是","还有","这","这个","这么","这么些","这么样","这么点儿","这些","这会儿","这儿","这就是说","这时","这样","这边","这里","进而","连","连同","通过","遵照","那","那个","那么","那么些","那么样","那些","那会儿","那儿","那时","那样","那边","那里","鄙人","鉴于","阿","除","除了","除此之外","除非","随","随着","零","非但","非徒","靠","顺","顺着","首先","︿","！","＃","＄","％","＆","（","）","＊","＋","，","０","１","２","３","４","５","６","７","８","９","：","；","＜","＞","？","＠","［","］","｛","｜","｝","～","￥"]
    stops = set(stops)
else:
    stops = set()

train_file = model_dir+"atec_nlp_sim_train.csv"
df1 = pd.read_csv(train_file,sep="\t", header=None, names =["id","sent1","sent2","label"], encoding="utf8")
# if len(df1) >= 102477: df1 = df1[:1000]

# 文本清理，预处理（分词）
clean_path = model_dir+"atec_clean.csv"
def pre_process(df, train_mode=True):
    x = lambda s: list(jieba.cut(star.sub("X",s)))
    df["words1"] = df["sent1"].apply(x)
    df["words2"] = df["sent2"].apply(x)
    if train_mode: df.to_csv(clean_path, sep="\t", index=False, encoding="utf8")
    return df

# 特征提取
feature_path = model_dir+"atec_feature.pkl"
feature_cfg = ["Not", "Length", "WordMatchShare", "TFIDFWordMatchShare", 
                # "PowerfulWordDoubleSide", "PowerfulWordDoubleSideRate", "PowerfulWordOneSide", "PowerfulWordOneSideRate", 
                "TFIDF", "NgramJaccardCoef", "NgramDiceDistance", "NgramDistance", "WordEmbeddingAveDis", "WordEmbeddingTFIDFAveDis"]
def feature_extract(df, train_mode=True):

    if "Not" in feature_cfg:
        def extract_row(row):
            not_cnt1 = row["words1"].count('不')
            not_cnt2 = row["words2"].count('不')

            fs = []
            fs.append(not_cnt1)
            fs.append(not_cnt2)
            if not_cnt1 > 0 and not_cnt2 > 0:
                fs.append(1.)
            else:
                fs.append(0.)
            if (not_cnt1 > 0) or (not_cnt2 > 0):
                fs.append(1.)
            else:
                fs.append(0.)
            if not_cnt2 <= 0 < not_cnt1 or not_cnt1 <= 0 < not_cnt2:
                fs.append(1.)
            else:
                fs.append(0.)

            return fs

        df["Not"] = df.apply(extract_row, axis=1)
        print("done Not")

    if "Length" in feature_cfg:
        def extract_row(row):
            len_q1, len_q2 = len(row["sent1"]), len(row["sent2"])
            return [len_q1,
                    len_q2,
                    len(row["words1"]), 
                    len(row["words2"]),
                    abs(len_q1 - len_q2),
                    1.0 * min(len_q1, len_q2) / max(len_q1, len_q2)]

        df["Length"] = df.apply(extract_row, axis=1)
        print("done Length")

    if "WordMatchShare" in feature_cfg:
        def extract_row(row):
            q1words = {}
            q2words = {}
            for word in row["words1"]:
                if word not in stops:
                    q1words[word] = q1words.get(word, 0) + 1
            for word in row["words2"]:
                if word not in stops:
                    q2words[word] = q2words.get(word, 0) + 1
            n_shared_word_in_q1 = sum([q1words[w] for w in q1words if w in q2words])
            n_shared_word_in_q2 = sum([q2words[w] for w in q2words if w in q1words])
            n_tol = sum(q1words.values()) + sum(q2words.values())
            if 1e-6 > n_tol:
                return [0.]
            else:
                return [1.0 * (n_shared_word_in_q1 + n_shared_word_in_q2) / n_tol]

        df["WordMatchShare"] = df.apply(extract_row, axis=1)
        print("done WordMatchShare")

    if "TFIDFWordMatchShare" in feature_cfg:
        idf_path = model_dir + "idf_weights.pkl"
        def init_idf():  # init idf weights
            idf = {}
            q_set = set()
            for index, row in df.iterrows():
                q1 = str(row['sent1'])
                q2 = str(row['sent2'])
                if q1 not in q_set:
                    q_set.add(q1)
                    for word in row["words1"]:
                        idf[word] = idf.get(word, 0) + 1
                if q2 not in q_set:
                    q_set.add(q2)
                    for word in row["words2"]:
                        idf[word] = idf.get(word, 0) + 1
            num_docs = len(df)
            for word in idf:
                idf[word] = math.log(num_docs / (idf[word] + 1.)) / math.log(2.)
            print("idf calculation done, len(idf)=%d" % len(idf))
            pd.to_pickle(idf, idf_path)
            return idf

        if train_mode: idf = init_idf()
        else: idf = pd.read_pickle(idf_path)

        def extract_row(row):
            q1words = {}
            q2words = {}
            for word in row["words1"]:
                q1words[word] = q1words.get(word, 0) + 1
            for word in row["words2"]:
                q2words[word] = q2words.get(word, 0) + 1
            sum_shared_word_in_q1 = sum([q1words[w] * idf.get(w, 0) for w in q1words if w in q2words])
            sum_shared_word_in_q2 = sum([q2words[w] * idf.get(w, 0) for w in q2words if w in q1words])
            sum_tol = sum(q1words[w] * idf.get(w, 0) for w in q1words) + sum(
                q2words[w] * idf.get(w, 0) for w in q2words)
            if 1e-6 > sum_tol:
                return [0.]
            else:
                return [1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol]

        df["TFIDFWordMatchShare"] = df.apply(extract_row, axis=1)
        print("done TFIDFWordMatchShare")

    powerful_words_path = model_dir + "powerful_words.pkl"
    def generate_powerful_word():
        """
        计算数据中词语的影响力，格式如下：
            词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
        """
        words_power = {}
        for index, row in df.iterrows():
            label = int(row['label'])
            q1_words = row['words1']
            q2_words = row['words2']
            all_words = set(q1_words + q2_words)
            q1_words = set(q1_words)
            q2_words = set(q2_words)
            for word in all_words:
                if word not in words_power:
                    words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
                words_power[word][0] += 1.
                words_power[word][1] += 1.

                if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                    # 计算单侧语句数量
                    words_power[word][3] += 1.
                    if 0 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算单侧语句正确比例
                        words_power[word][4] += 1.
                if (word in q1_words) and (word in q2_words):
                    # 计算双侧语句数量
                    words_power[word][5] += 1.
                    if 1 == label:
                        # 计算正确语句对数量
                        words_power[word][2] += 1.
                        # 计算双侧语句正确比例
                        words_power[word][6] += 1.
        for word in words_power:
            # 计算出现语句对比例
            words_power[word][1] /= len(df)
            # 计算正确语句对比例
            words_power[word][2] /= words_power[word][0]
            # 计算单侧语句对正确比例
            if words_power[word][3] > 1e-6:
                words_power[word][4] /= words_power[word][3]
            # 计算单侧语句对比例
            words_power[word][3] /= words_power[word][0]
            # 计算双侧语句对正确比例
            if words_power[word][5] > 1e-6:
                words_power[word][6] /= words_power[word][5]
            # 计算双侧语句对比例
            words_power[word][5] /= words_power[word][0]
        sorted_words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)
        print("power words calculation done, len(words_power)=%d" % len(sorted_words_power))
        pd.to_pickle(sorted_words_power, powerful_words_path)
        return sorted_words_power

    if train_mode: pword = generate_powerful_word()
    else: pword = pd.load_pickle(powerful_words_path)

    # thresh_num, thresh_rate = 500, 0.9
    thresh_num, thresh_rate = 7, 0.3

    pword_filtered = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
    pword_sort = sorted(pword_filtered, key=lambda d: d[1][6], reverse=True)
    pword_dside = set(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
    print('Double side power words(%d): %s' % (len(pword_dside), str(pword_dside)))

    def extract_row(row):
        tags = []
        q1_words = row["words1"]
        q2_words = row["words2"]
        for word in pword_dside:
            if (word in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    if "PowerfulWordDoubleSide" in feature_cfg:
        df["PowerfulWordDoubleSide"] = df.apply(extract_row, axis=1)
        print("done PowerfulWordDoubleSide")

    pword_dict = dict(pword)
    def extract_row(row):
        num_least = 300
        rate = [1.0]
        q1_words = set(row["words1"])
        q2_words = set(row["words2"])
        share_words = list(q1_words.intersection(q2_words))
        for word in share_words:
            if word not in pword_dict:
                continue
            if pword_dict[word][0] * pword_dict[word][5] < num_least:
                continue
            rate[0] *= (1.0 - pword_dict[word][6])
        rate = [1 - num for num in rate]
        return rate

    if "PowerfulWordDoubleSideRate" in feature_cfg:
        df["PowerfulWordDoubleSideRate"] = df.apply(extract_row, axis=1)
        print("done PowerfulWordDoubleSideRate")


    thresh_num, thresh_rate = 20, 0.8

    pword_filtered = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
    pword_oside = set(map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword_filtered)))
    print('One side power words(%d): %s' % (len(pword_oside), str(pword_oside)))
    def extract_row(row):
        tags = []
        q1_words = set(row["words1"])
        q2_words = set(row["words2"])
        for word in pword_oside:
            if (word in q1_words) and (word not in q2_words):
                tags.append(1.0)
            elif (word not in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
        return tags

    if "PowerfulWordOneSide" in feature_cfg:
        df["PowerfulWordOneSide"] = df.apply(extract_row, axis=1)
        print("done PowerfulWordOneSide")

    def extract_row(row):
        num_least = 300
        rate = [1.0]
        q1_words = set(row["words1"])
        q2_words = set(row["words2"])
        q1_diff = list(q1_words.difference(q2_words))
        q2_diff = list(q2_words.difference(q1_words))
        all_diff = set(q1_diff + q2_diff)
        for word in all_diff:
            if word not in pword_dict:
                continue
            if pword_dict[word][0] * pword_dict[word][3] < num_least:
                continue
            rate[0] *= (1.0 - pword_dict[word][4])
        rate = [1 - num for num in rate]
        return rate

    if "PowerfulWordOneSideRate" in feature_cfg:
        df["PowerfulWordOneSideRate"] = df.apply(extract_row, axis=1)
        print("done PowerfulWordOneSideRate")

    if "TFIDF" in feature_cfg:
        tfidf_path = model_dir + "tfidf_transformer.pkl"
        def init_tfidf():
            tfidf = TfidfVectorizer(stop_words=list(stops), ngram_range=(1, 1), token_pattern=r"\w+")
            tfidf_txt = pd.Series(df['words1'].apply(lambda x: " ".join(x)).tolist() + 
                                  df['words2'].apply(lambda x: " ".join(x)).tolist())
            tfidf.fit_transform(tfidf_txt)
            print("init tfidf done ")
            # print(tfidf.vocabulary_)
            pd.to_pickle(tfidf, tfidf_path)
            return tfidf

        if train_mode: tfidf = init_tfidf()
        else: tfidf = pd.read_pickle(tfidf_path)

        def extract_row(row):
            q1 = " ".join(row['words1'])
            q2 = " ".join(row['words2'])
            a1 = tfidf.transform([q1]).data
            a2 = tfidf.transform([q2]).data
            fs = [np.sum(a1),np.sum(a2),np.mean(a1),np.mean(a2),len(a1),len(a2)]
            return fs

        df["TFIDF"] = df.apply(extract_row, axis=1)
        print("done TFIDF")

    if "NgramJaccardCoef" in feature_cfg:
        def extract_row(row):
            q1_words = row['words1']
            q2_words = row['words2']
            fs = list()
            for n in range(1, 4):
                q1_ngrams = NgramUtil.ngrams(q1_words, n)
                q2_ngrams = NgramUtil.ngrams(q2_words, n)
                A = set(q1_ngrams)
                B = set(q2_ngrams)
                x = len(A.intersection(B))
                y = len(A.union(B))
                val = 0.0 if y==0 else x/y
                fs.append(val)
            return fs

        df["NgramJaccardCoef"] = df.apply(extract_row, axis=1)
        print("done NgramJaccardCoef")

    if "NgramDiceDistance" in feature_cfg:
        def extract_row(row):
            q1_words = row['words1']
            q2_words = row['words2']
            fs = list()
            for n in range(1, 4):
                q1_ngrams = NgramUtil.ngrams(q1_words, n)
                q2_ngrams = NgramUtil.ngrams(q2_words, n)
                A = set(q1_ngrams)
                B = set(q2_ngrams)
                x = 2. * len(A.intersection(B))
                y = len(A) + len(B)
                val = 0.0 if y==0 else x/y
                fs.append(val)
            return fs

        df["NgramDiceDistance"] = df.apply(extract_row, axis=1)
        print("done NgramDiceDistance")

    if "NgramDistance" in feature_cfg:
        def extract_row(row):
            q1_words = row['words1']
            q2_words = row['words2']
            fs = list()
            aggregation_modes_outer = [np.mean,np.max,np.min,np.median]
            aggregation_modes_inner = [np.mean,np.std,np.max,np.min,np.median]
            for n_ngram in range(1, 4):
                q1_ngrams = NgramUtil.ngrams(q1_words, n_ngram)
                q2_ngrams = NgramUtil.ngrams(q2_words, n_ngram)
                val_list = list()
                for w1 in q1_ngrams:
                    _val_list = list()
                    for w2 in q2_ngrams:
                        s = 1. - SequenceMatcher(None, w1, w2, False).quick_ratio()     # ratio()
                        _val_list.append(s)
                    if len(_val_list) == 0:
                        _val_list = [MISSING_VALUE_NUMERIC]
                    val_list.append(_val_list)
                if len(val_list) == 0:
                    val_list = [[MISSING_VALUE_NUMERIC]]
                data = np.array(val_list)
                fs.extend([mode_outer(mode_inner(data,axis=1)) for mode_inner in aggregation_modes_inner for mode_outer in aggregation_modes_outer])
            return fs

        df["NgramDistance"] = df.apply(extract_row, axis=1)
        print("done NgramDistance")

    we_len = 300 if online else 256
    word_embedding_model = gensim.models.Word2Vec.load(model_dir + "word2vec_gensim%s"%we_len)
    word2index = {v:k for k,v in enumerate(word_embedding_model.wv.index2word)}
    if "WordEmbeddingAveDis" in feature_cfg:
        def extract_row(row):
            q1_words = row['words1']
            q2_words = row['words2']

            q1_vec = np.array(we_len * [0.])
            q2_vec = np.array(we_len * [0.])

            for word in q1_words:
                if word in word2index:
                    q1_vec += word_embedding_model[word]
            for word in q2_words:
                if word in word2index:
                    q2_vec += word_embedding_model[word]

            cos_sim = 0.
            q1_vec = np.mat(q1_vec)
            q2_vec = np.mat(q2_vec)
            factor = np.linalg.norm(q1_vec) * np.linalg.norm(q2_vec)
            if 1e-6 < factor:
                cos_sim = float(q1_vec * q2_vec.T) / factor

            return [cos_sim]

        df["WordEmbeddingAveDis"] = df.apply(extract_row, axis=1)

    if "WordEmbeddingTFIDFAveDis" in feature_cfg:
        idf = pd.read_pickle(idf_path)
        def extract_row(row):
            q1_words = row['words1']
            q2_words = row['words2']

            q1_vec = np.array(we_len * [0.])
            q2_vec = np.array(we_len * [0.])
            q1_words_cnt = {}
            q2_words_cnt = {}
            for word in q1_words:
                q1_words_cnt[word] = q1_words_cnt.get(word, 0.) + 1.
            for word in q2_words:
                q2_words_cnt[word] = q2_words_cnt.get(word, 0.) + 1.

            for word in q1_words_cnt:
                if word in word2index:
                    q1_vec += idf.get(word, 0.) * q1_words_cnt[word] * word_embedding_model[word]
            for word in q2_words_cnt:
                if word in word2index:
                    q2_vec += idf.get(word, 0.) * q2_words_cnt[word] * word_embedding_model[word]

            cos_sim = 0.
            q1_vec = np.mat(q1_vec)
            q2_vec = np.mat(q2_vec)
            factor = np.linalg.norm(q1_vec) * np.linalg.norm(q2_vec)
            if 1e-6 < factor:
                cos_sim = float(q1_vec * q2_vec.T) / factor

            return [cos_sim]

        df["WordEmbeddingTFIDFAveDis"] = df.apply(extract_row, axis=1)


    def merge_feature(row):
        fs = []
        for feature in feature_cfg:
            fs += row[feature]
        return fs

    df["feature"] = df.apply(merge_feature, axis=1)
    x, y = np.array(df["feature"].tolist()), np.array(df["label"].astype(int))
    if train_mode: pd.to_pickle((x,y),feature_path)
    return (x,y)


from difflib import SequenceMatcher
MISSING_VALUE_NUMERIC = -1

class NgramUtil(object):

    def __init__(self):
        pass

    @staticmethod
    def unigrams(words):
        """
            Input: a list of words, e.g., ["I", "am", "Denny"]
            Output: a list of unigram
        """
        assert type(words) == list
        return words

    @staticmethod
    def bigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of bigram, e.g., ["I_am", "am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 1:
            lst = []
            for i in range(L - 1):
                for k in range(1, skip + 2):
                    if i + k < L:
                        lst.append(join_string.join([words[i], words[i + k]]))
        else:
            # set it as unigram
            lst = NgramUtil.unigrams(words)
        return lst

    @staticmethod
    def trigrams(words, join_string, skip=0):
        """
           Input: a list of words, e.g., ["I", "am", "Denny"]
           Output: a list of trigram, e.g., ["I_am_Denny"]
        """
        assert type(words) == list
        L = len(words)
        if L > 2:
            lst = []
            for i in range(L - 2):
                for k1 in range(1, skip + 2):
                    for k2 in range(1, skip + 2):
                        if i + k1 < L and i + k1 + k2 < L:
                            lst.append(join_string.join([words[i], words[i + k1], words[i + k1 + k2]]))
        else:
            # set it as bigram
            lst = NgramUtil.bigrams(words, join_string, skip)
        return lst

    @staticmethod
    def fourgrams(words, join_string):
        """
            Input: a list of words, e.g., ["I", "am", "Denny", "boy"]
            Output: a list of trigram, e.g., ["I_am_Denny_boy"]
        """
        assert type(words) == list
        L = len(words)
        if L > 3:
            lst = []
            for i in xrange(L - 3):
                lst.append(join_string.join([words[i], words[i + 1], words[i + 2], words[i + 3]]))
        else:
            # set it as trigram
            lst = NgramUtil.trigrams(words, join_string)
        return lst

    @staticmethod
    def ngrams(words, ngram, join_string=" "):
        """
        wrapper for ngram
        """
        if ngram == 1:
            return NgramUtil.unigrams(words)
        elif ngram == 2:
            return NgramUtil.bigrams(words, join_string)
        elif ngram == 3:
            return NgramUtil.trigrams(words, join_string)
        elif ngram == 4:
            return NgramUtil.fourgrams(words, join_string)
        elif ngram == 12:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            return unigram + bigram
        elif ngram == 123:
            unigram = NgramUtil.unigrams(words)
            bigram = [x for x in NgramUtil.bigrams(words, join_string) if len(x.split(join_string)) == 2]
            trigram = [x for x in NgramUtil.trigrams(words, join_string) if len(x.split(join_string)) == 3]
            return unigram + bigram + trigram


def r_f1_thresh(y_pred,y_true,step=1000):
    e = np.zeros((len(y_true),2))
    e[:,0] = y_pred.reshape(-1)
    e[:,1] = y_true
    f = pd.DataFrame(e)
    thrs = np.linspace(0,1,step+1)
    x = np.array([f1_score(y_pred=f.loc[:,0]>thr, y_true=f.loc[:,1]) for thr in thrs])
    f1_, thresh = max(x),thrs[x.argmax()]
    return f.corr()[0][1], f1_, thresh

random_state = 42
def train_old_classifier(data=None,train_mode=True):
    if data is None:
        x,y = pd.read_pickle(feature_path)
    else:x,y = data

    trn_x, val_x, trn_y, val_y = train_test_split(x,y, test_size=test_size, random_state=random_state)


    classifier = ["lrcv","lgbm"][1]
    if classifier == "lgbm":
        print("lightgbm")
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'l2', 'auc'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }

        import lightgbm as lgb
        lgb_train = lgb.Dataset(trn_x, trn_y)
        lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)
        model_gbm = lgb.train(params, lgb_train, num_boost_round=200, 
                            valid_sets=lgb_eval, early_stopping_rounds=10)

        val_y_pred = model_gbm.predict(val_x, num_iteration=model_gbm.best_iteration)
        print(r_f1_thresh(val_y_pred, val_y))

    classifier = ["lrcv","lgbm"][0]
    if classifier == "lrcv":
        print("LogisticRegression")
        clf = LogisticRegression()
        clf.fit(trn_x, trn_y)
        val_y_pred = clf.predict_proba(val_x)[:,1]
        print(r_f1_thresh(val_y_pred, val_y))

df = None
df = pre_process(df1)
data = feature_extract(df)
train_old_classifier(data)