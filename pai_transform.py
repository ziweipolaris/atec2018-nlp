
import gensim
import numpy as np
model_dir = "pai_model/"
cfgs = [
    ("siamese","char",24,300,[64, 64, 64],90),
    ("siamese","word",20,300,[80, 64, 64],90),
]

def transform_weight(cfg):
    model_type,dtype,input_length,w2v_length,n_hidden,n_epoch = cfg

    w2v_path = "%s2vec_gensim%d"%(dtype,w2v_length)
        
    old_weights = gensim.models.Word2Vec.load("model/"+w2v_path).wv
    open(model_dir + "%s2index%d.csv"%(dtype,w2v_length),'w',encoding='utf8').write('\n'.join([char+'#'+str(i) for i,char in enumerate(old_weights.index2word)]))
    
    embedding_length = len(old_weights.index2word)
    print(dtype,embedding_length)
        
    with open(model_dir+w2v_path+".csv",'w',encoding="utf8") as out:
        indexs = [(kv.split("\t")[0],kv.split("\t")[1]) for kv in open(model_dir + "%s2index%d.csv"%(dtype,w2v_length),"r",encoding='utf8').read().split("\n")]
        for name,index in indexs:
            out.write(index + "#" +  ",".join(map(str,old_weights[name]))+"\n")

def transform_weight_npy(cfg):
    model_type,dtype,input_length,w2v_length,n_hidden,n_epoch = cfg
    if dtype == 'word':
        embedding_length = 429200
    elif dtype == 'char':
        embedding_length = 9436

    if dtype == 'word':
        embedding_length = 501344
    elif dtype == 'char':
        embedding_length = 10125


    w2v_path = model_dir+"%s2vec_gensim%d.csv"%(dtype,w2v_length)        
    embedding_matrix = np.zeros((embedding_length, w2v_length))
    for no,line in enumerate(open(w2v_path,encoding="utf8")):
        embedding_matrix[no,:] = np.array([float(x) for x in line.split("#")[1].strip().split(",")])
    np.save(model_dir +"%s2vec_gensim%d.npy"%(dtype,w2v_length), embedding_matrix)


def transform_weight_npy2(cfg,ebed_type):
    model_type,dtype,input_length,w2v_length,n_hidden,n_epoch = cfg

    w2v_path = model_dir+"%s2vec_%s%d.vec"%(dtype,ebed_type,w2v_length)
    with open(w2v_path,'r',encoding='utf8') as file:
        line = file.readline()
        tokens = []
        embedding_length, w2v_length = map(int,line.split())
        embedding_matrix = np.zeros((embedding_length, w2v_length))
        print(line)
        for index,line in enumerate(file):
            data = line.strip().split(" ")
            assert(len(data) == 301)
            tokens.append(data[0])
            embedding_matrix[index,:] = np.array([float(x) for x in data[1:]])
      
        open(model_dir+"fasttext/%s2index_%s%d.csv"%(dtype,ebed_type,w2v_length), 'w', encoding="utf8").write("\n".join(["%s\t%d"%(token,index) for index,token in enumerate(tokens)]))
        np.save(model_dir +"fasttext/%s2vec_%s%d.npy"%(dtype,ebed_type,w2v_length), embedding_matrix)


def read_save():
    import pandas as pd
    csv = pd.read_csv(model_dir+"atec_nlp_sim_train.csv",sep="\t",header=None,encoding='utf8')
    csv.columns=["lino","sent1","sent2","label"]
    csv.to_csv(model_dir + "foo.csv",sep="\t",header=None,index=False,encoding='utf8')
    for no,x in enumerate(open(model_dir + "foo.csv",'r',encoding='utf8')):
        print(x)
        if no>3:
            break

def save_to_file(df1,df2,df3,df4):
    #df1 df2 df3 df4类型为: pandas.core.frame.DataFrame.分别引用输入桩数据
    #topai(1, df1)函数把df1内容写入第一个输出桩
    df1.to_csv(model_dir+'char2index.csv',sep="#",header=None,index=Fasle,encoding='utf8',)

    w2 = df2.set_index("index",inplace=False)
    w2.sort_index(inplace=True)
    w2.to_csv(model_dir+'char2vec_gensim256.csv',sep="#",header=None,index=True,encoding='utf8',)

    df3.to_csv(model_dir+'word2index.csv',sep="#",header=None,index=Fasle,encoding='utf8',)

    w4 = df4.set_index("index",inplace=False)
    w4.sort_index(inplace=True)
    w4.to_csv(model_dir+'word2vec_gensim256.csv',sep="#",header=None,index=True,encoding='utf8',)

    for filename in [
        'char2index.csv',
        'char2vec_gensim256.csv',
        'word2index.csv',
        'word2vec_gensim256.csv',
        'atec_nlp_sim_train.csv',
        ]:
        with open(model_dir + filename,'r',encoding='utf8') as f:
            for no,line in enumerate(f):
                print(line)
                if no>3:
                    print(filename,"  ok")
                    break


def transform_weight_merge():
    w2v_path = "model/sgns.merge.char.txt"
    chars = []
    # with open(w2v_path,'r',encoding='utf8') as file:
    #     line = file.readline()
    #     fout = open(model_dir + )
    #     print(line)
    #     for line in file:
    #         data = line.strip().split(" ")
    #         assert(len(data) == 301)
    #         chars.append(data[0])

    #         break
    # return
    # dtype, w2v_length = 'char',300
    # open("model/%s2index_merge%s.txt"%(dtype,w2v_length),'w',encoding='utf8').write('\n'.join([char+'\t'+str(i) for i,char in enumerate(chars)]))
    
    # embedding_length = len(old_weights.index2word)
    # print(dtype,embedding_length)
        
    # with open(model_dir+w2v_path+".csv",'w',encoding="utf8") as out:
    #     indexs = [(kv.split("\t")[0],kv.split("\t")[1]) for kv in open("model/%s2index%s.txt"%(dtype,w2v_length),"r",encoding='utf8').read().split("\n")]
    #     for name,index in indexs:
    #         out.write(index + "#" +  ",".join(map(str,old_weights[name]))+"\n")


def transform_wiki():
    import re
    for i in range(11):
        with open("resources/wiki_corpus/wiki%02d"%i,"r",encoding="utf8") as wiki_in:
            with open("resources/wiki_corpus/wiki%02d.csv"%i,"w",encoding="utf8") as wiki_out:
                for line in wiki_in:
                    title, doc = line.strip().split("|")
                    if len(doc)>=10:
                        wiki_out.write(title+"|"+doc+"\n")

def transfrom_wiki2ids(dtype):
    n_tokens = 1000000
    with open("resources/wiki_corpus/wiki.csv","r",encoding="utf8") as file:
        train_array, val_array = [],[]
        with open(model_dir + "lm_train.csv", 'w', encoding="utf8") as train_file:
            for line in file:
                title, doc = line.strip().split("|")
                sentenses = doc.split("#")
                train_file.write()
                break

def main():
    # transform_weight(cfgs[0])
    # transform_weight(cfgs[1])
    # transform_weight_npy(cfgs[0])
    # transform_weight_npy(cfgs[1])
    # transform_weight_merge()  # 下载的混合词嵌入，
    # transform_weight_merge_npy()  # 下载的混合词嵌入，
    # transform_weight_fasttext_sgns()  # 自己训练的fasttext词嵌入
    # transform_weight_fasttext_sgns()  # 自己训练的fasttext词嵌入
    # transform_weight_fasttext_sgns_npy()  # 自己训练的fasttext词嵌入
    # transform_weight_fasttext_sgns_npy()  # 自己训练的fasttext词嵌入
    # read_save()
    # transform_wiki()
    # [transform_weight_npy2(cfg,ebed_type) for cfg in cfgs for ebed_type in ["fastskip", "fastcbow"]]
    transfrom_wiki2ids("char")

if __name__ == '__main__':
    main()