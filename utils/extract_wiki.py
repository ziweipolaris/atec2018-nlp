import codecs
import re

import bz2file
import jieba_fast as jieba
from gensim.corpora.wikicorpus import extract_pages, filter_wiki
# from gensim.corpora import WikiCorpus
from tqdm import tqdm


def get_wiki():
    from opencc import OpenCC
    # 参考这篇博客注释
    # https://kexue.fm/archives/4176
    opencc1 = OpenCC("t2s")
    resub1 = re.compile(':*{\|[\s\S]*?\|}')  
    resub2 = re.compile('<gallery>[\s\S]*?</gallery>')  
    resub3 = re.compile('(.){{([^{}\n]*?\|[^{}\n]*?)}}')  
    resub4 = re.compile('\* *\n|\'{2,}')  
    resub5 = re.compile('\n+')  
    resub6 = re.compile('\n[:;]|\n +')  
    resub7 = re.compile('\n==')

    refind1 = re.compile('^[a-zA-Z]+:')
    refind2 = re.compile('^#')

    p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
    p2 = re.compile(r'[（\(][，；。？！\s]*[）\)]')
    p3 = re.compile(r'[「『]')
    p4 = re.compile(r'[」』]')

    def wiki_replace(s):
        s = filter_wiki(s)  
        s = resub1.sub('', s)  
        s = resub2.sub('', s)  
        s = resub3.sub('\\1[[\\2]]', s)  
        s = resub4.sub('', s)  
        s = resub5.sub('\n', s)  
        s = resub6.sub('\n', s)  
        s = resub7.sub('\n\n==', s)
        s = p1.sub(r'\2', s)
        s = p2.sub(r'', s)
        s = p3.sub(r'“', s)
        s = p4.sub(r'”', s)
        return opencc1.convert(s).strip()
    
    wiki = extract_pages(bz2file.open('zhwiki-latest-pages-articles.xml.bz2'))

    # wiki=WikiCorpus('zhwiki-latest-pages-articles.xml.bz2',lemmatize=False,dictionary={})

    with codecs.open('wiki.txt', 'w', encoding='utf-8') as f:
        i = 0
        filelist = []
        for d in tqdm(wiki):
            
            print(d[0])
            print(d[1])

            i+=1
            
            if i == 5:break

            continue
            if not refind1.findall(d[0]) and d[0] and not refind2.findall(d[1]):
                filelist.append(d[0]+"\n"+d[1])
                line = d[1]

                i += 1  
                if i % 100 == 0:  
                    s = wiki_replace("\n\n".join(filelist))
                    f.write(s)  
                    filelist = []

def get_cut_std_wiki():
    with open("cut_std_wiki.txt","w",encoding="utf8") as output:
        with open("std_wiki.txt","r",encoding="utf8") as file:
            for line in tqdm(file): 
                output.write(" ".join(list(jieba.cut(line))))

def get_wiki2():
    reobj1 = re.compile(r"[ `~!@#$%^&*\(\)-_=+\[\]\{\}\\\|;:\'\",<.>/?a-zA-Z\d]+")
    reobj2 = re.compile(r"\n+")
    reobj3 = re.compile("(（）)|(“”)|(「」)|(《》)|(“”)|(‘’)|(【】)|[，。？——！]{2,}")
    reuseful = re.compile('^[a-zA-Z]+:')
    redirect = re.compile(r"^#")
    def wiki_replace(s):
        s = filter_wiki(s)  
        s = reobj1.sub("", s)     # 为上传阿里云剔除竖线(|)符号
        s = reobj2.sub("#",s)
        s = reobj3.sub("",s)
        return s

    wiki = extract_pages(bz2file.open('zhwiki-latest-pages-articles.xml.bz2'))
    with codecs.open('wiki-tw.csv', 'w', encoding='utf-8') as f:
        i = 0
        filelist = []
        for d in tqdm(wiki):
            if not reuseful.findall(d[0]) and not redirect.findall(d[1]):
                i+=1
                filelist.append(reobj1.sub("",d[0])+"|"+wiki_replace(d[1])+"\n")
                if i % 1000 == 0:  
                    s = ("".join(filelist))
                    f.write(s)
                    filelist = []
        if filelist:
            s = ("".join(filelist))
            f.write(s)
        
def wiki_error():
    for no,line in enumerate(open("wiki_1.csv",'r', encoding="utf8")):
        pair = line.split("|")
        if len(pair)>2:
            print(no,pair[0],pair[1])

if __name__ == '__main__':
    # get_wiki2() # 繁体转简体 + 特殊符号处理
    wiki_error()