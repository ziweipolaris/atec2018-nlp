# https://pai4contest.cloud.alipay.com/experiment.htm?Lang=zh_CN&lang=zh_CN&etag=iZbp10tfg72g1zj2tnd6rwZ&experimentId=3508

import time
start_time = time.time()
import gensim
import numpy as np
import pandas as pd
import os
import re
try:
    import jieba_fast as jieba
except Exception as e:
    import jieba
from sklearn.model_selection import train_test_split


try:
    print(model_dir)
    test_size = 0.025
    online=True
except:
    model_dir = "pai_model/"
    test_size = 0.05
    online=False

random_state = 42

cfgs = [
    ("siamese","char",24,300,[64, 64, 64],90),
    ("siamese","word",20,300,[80, 64, 64],90),
]

new_words = "支付宝 付款码 二维码 收钱码 转账 退款 退钱 余额宝 运费险 还钱 还款 花呗 借呗 蚂蚁花呗 蚂蚁借呗 蚂蚁森林 小黄车 飞猪 微客 宝卡 芝麻信用 亲密付 淘票票 饿了么 摩拜 滴滴 滴滴出行".split(" ")
for word in new_words:
    jieba.add_word(word)

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


# start imports###################################################################
from enum import IntEnum
from timeit import default_timer as timer
import copy
import math
from abc import abstractmethod
import contextlib
from glob import glob, iglob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import chain
from functools import partial
from functools import wraps
from collections import Iterable, Counter, OrderedDict
import pickle
# end imports####################################################################

# start torch imports ############################################################
from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler, BatchSampler
from torch.utils.data import Dataset
from distutils.version import LooseVersion
import torch
from torch import nn, cuda, backends, FloatTensor, LongTensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset
from torch.nn.init import kaiming_uniform, kaiming_normal
import queue
import collections,sys,traceback,threading
import warnings
warnings.filterwarnings('ignore', message='Implicit dimension choice', category=UserWarning)

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
if IS_TORCH_04:
    from torch.nn.init import kaiming_uniform_ as kaiming_uniform
    from torch.nn.init import kaiming_normal_ as kaiming_normal

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())
def save_model(m, p): torch.save(m.state_dict(), p)
def load_model(m, p):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    names = set(m.state_dict().keys())
    for n in list(sd.keys()): # list "detatches" the iterator
        if n not in names and n+'_raw' in names:
            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
            del sd[n]
    m.load_state_dict(sd)
# end torch imports ##############################################################

# start fastai.core###############################################################
def sum_geom(a,r,n): return a*n if r==1 else math.ceil(a*(1-r**n)/(1-r))

def is_listy(x): return isinstance(x, (list,tuple))
def is_iter(x): return isinstance(x, collections.Iterable)
def map_over(x, f): return [f(o) for o in x] if is_listy(x) else f(x)
def map_none(x, f): return None if x is None else f(x)
def delistify(x): return x[0] if is_listy(x) else x
def listify(x, y):
    if not is_iter(x): x=[x]
    n = y if type(y)==int else len(y)
    if len(x)==1: x = x * n
    return x

def datafy(x):
    if is_listy(x): return [o.data for o in x]
    else:           return x.data

conv_dict = {np.dtype('int8'): torch.LongTensor, np.dtype('int16'): torch.LongTensor,
    np.dtype('int32'): torch.LongTensor, np.dtype('int64'): torch.LongTensor,
    np.dtype('float32'): torch.FloatTensor, np.dtype('float64'): torch.FloatTensor}

def A(*a):
    """convert iterable object into numpy array"""
    return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

def T(a, half=False, cuda=True):
    """
    Convert numpy array into a pytorch tensor. 
    if Cuda is available and USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = torch.cuda.HalfTensor(a) if half else torch.FloatTensor(a)
        else: raise NotImplementedError(a.dtype)
    if cuda: a = to_gpu(a, async=True)
    return a

def create_variable(x, volatile, requires_grad=False):
    if type (x) != Variable:
        if IS_TORCH_04: x = Variable(T(x), requires_grad=requires_grad)
        else:           x = Variable(T(x), requires_grad=requires_grad, volatile=volatile)
    return x

def V_(x, requires_grad=False, volatile=False):
    '''equivalent to create_variable, which creates a pytorch tensor'''
    return create_variable(x, volatile=volatile, requires_grad=requires_grad)
def V(x, requires_grad=False, volatile=False):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, lambda o: V_(o, requires_grad, volatile))

def VV_(x): 
    '''creates a volatile tensor, which does not require gradients. '''
    return create_variable(x, True)

def VV(x):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, VV_)

def to_np(v):
    '''returns an np.array object given an input of np.array, list, tuple, torch variable or tensor.'''
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, Variable): v=v.data
    if torch.cuda.is_available():
        if isinstance(v, torch.cuda.HalfTensor): v=v.float()
    if isinstance(v, torch.FloatTensor): v=v.float()
    return v.cpu().numpy()

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')
USE_GPU = torch.cuda.is_available()
def to_gpu(x, *args, **kwargs):
    '''puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x

def noop(*args, **kwargs): return

def split_by_idxs(seq, idxs):
    '''A generator that returns sequence pieces, seperated by indexes specified in idxs. '''
    last = 0
    for idx in idxs:
        if not (-len(seq) <= idx < len(seq)):
          raise KeyError(f'Idx {idx} is out-of-bounds')
        yield seq[last:idx]
        last = idx
    yield seq[last:]

def trainable_params_(m):
    '''Returns a list of trainable parameters in the model m. (i.e., those that require gradients.)'''
    return [p for p in m.parameters() if p.requires_grad]

def chain_params(p):
    if is_listy(p):
        return list(chain(*[trainable_params_(o) for o in p]))
    return trainable_params_(p)

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module): f(m)
    if len(c)>0:
        for l in c: apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def SGD_Momentum(momentum):
    return lambda *args, **kwargs: optim.SGD(*args, momentum=momentum, **kwargs)

def one_hot(a,c): return np.eye(c)[a]

def partition(a, sz): 
    """splits iterables a in equal parts of size sz"""
    return [a[i:i+sz] for i in range(0, len(a), sz)]

def partition_by_cores(a):
    return partition(a, len(a)//num_cpus() + 1)

def num_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


class BasicModel():
    def __init__(self,model,name='unnamed'): self.model,self.name = model,name
    def get_layer_groups(self, do_fc=False): return children(self.model)

class SingleModel(BasicModel):
    def get_layer_groups(self): return [self.model]

class SimpleNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)


def save(fn, a): 
    """Utility function that savess model, function, etc as pickle"""    
    pickle.dump(a, open(fn,'wb'))
def load(fn): 
    """Utility function that loads model, function, etc as pickle"""
    return pickle.load(open(fn,'rb'))
def load2(fn):
    """Utility funciton allowing model piclking across Python2 and Python3"""
    return pickle.load(open(fn,'rb'), encoding='iso-8859-1')

def load_array(fname): 
    '''
    Load array using bcolz, which is based on numpy, for fast array saving and loading operations. 
    https://github.com/Blosc/bcolz
    '''
    return bcolz.open(fname)[:]


def chunk_iter(iterable, chunk_size):
    '''A generator that yields chunks of iterable, chunk_size at a time. '''
    while True:
        chunk = []
        try:
            for _ in range(chunk_size): chunk.append(next(iterable))
            yield chunk
        except StopIteration:
            if chunk: yield chunk
            break

def set_grad_enabled(mode): return torch.set_grad_enabled(mode) if IS_TORCH_04 else contextlib.suppress()

def no_grad_context(): return torch.no_grad() if IS_TORCH_04 else contextlib.suppress()
# end fastai.core#################################################################


# start fastai.transforms#################################################################

# end fastai.transforms################################################################


# start fastai.dataloader#################################################################
string_classes = (str, bytes)


def get_tensor(batch, pin, half=False):
    if isinstance(batch, (np.ndarray, np.generic)):
        batch = T(batch, half=half, cuda=False).contiguous()
        if pin: batch = batch.pin_memory()
        return to_gpu(batch)
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.Mapping):
        return {k: get_tensor(sample, pin, half) for k, sample in batch.items()}
    elif isinstance(batch, collections.Sequence):
        return [get_tensor(sample, pin, half) for sample in batch]
    raise TypeError(f"batch must contain numbers, dicts or lists; found {type(batch)}")


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, pad_idx=0,
                 num_workers=None, pin_memory=False, drop_last=False, pre_pad=True, half=False,
                 transpose=False, transpose_y=False):
        self.dataset,self.batch_size,self.num_workers = dataset,batch_size,num_workers
        self.pin_memory,self.drop_last,self.pre_pad = pin_memory,drop_last,pre_pad
        self.transpose,self.transpose_y,self.pad_idx,self.half = transpose,transpose_y,pad_idx,half

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler is mutually exclusive with '
                                 'batch_size, shuffle, sampler, and drop_last')

        if sampler is not None and shuffle:
            raise ValueError('sampler is mutually exclusive with shuffle')

        if batch_sampler is None:
            if sampler is None:
                sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        if num_workers is None:
            self.num_workers = num_cpus()

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self): return len(self.batch_sampler)

    def jag_stack(self, b):
        if len(b[0].shape) not in (1,2): return np.stack(b)
        ml = max(len(o) for o in b)
        if min(len(o) for o in b)==ml: return np.stack(b)
        res = np.zeros((len(b), ml), dtype=b[0].dtype) + self.pad_idx
        for i,o in enumerate(b):
            if self.pre_pad: res[i, -len(o):] = o
            else:            res[i,  :len(o)] = o
        return res

    def np_collate(self, batch):
        b = batch[0]
        if isinstance(b, (np.ndarray, np.generic)): return self.jag_stack(batch)
        elif isinstance(b, (int, float)): return np.array(batch)
        elif isinstance(b, string_classes): return batch
        elif isinstance(b, collections.Mapping):
            return {key: self.np_collate([d[key] for d in batch]) for key in b}
        elif isinstance(b, collections.Sequence):
            return [self.np_collate(samples) for samples in zip(*batch)]
        raise TypeError(("batch must contain numbers, dicts or lists; found {}".format(type(b))))

    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        if self.transpose:   res[0] = res[0].T
        if self.transpose_y: res[1] = res[1].T
        return res

    def __iter__(self):
        if self.num_workers==0:
            for batch in map(self.get_batch, iter(self.batch_sampler)):
                yield get_tensor(batch, self.pin_memory, self.half)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
                for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
                    for batch in e.map(self.get_batch, c):
                        yield get_tensor(batch, self.pin_memory, self.half)
# end fastai.dataloader#################################################################


# start fastai.dataset#################################################################
def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    """ Get a list of index values for Validation set from a dataset
    
    Arguments:
        n : int, Total number of elements in the data set.
        cv_idx : int, starting index [idx_start = cv_idx*int(val_pct*n)] 
        val_pct : (int, float), validation set percentage 
        seed : seed value for RandomState
        
    Returns:
        list of indexes 
    """
    np.random.seed(seed)
    n_val = int(val_pct*n)
    idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start+n_val]

def path_for(root_path, new_path, targ):
    return os.path.join(root_path, new_path, str(targ))

def resize_img(fname, targ, path, new_path, fn=None):
    """
    Enlarge or shrink a single image to scale, such that the smaller of the height or width dimension is equal to targ.
    """
    if fn is None:
        fn = resize_fn(targ)
    dest = os.path.join(path_for(path, new_path, targ), fname)
    if os.path.exists(dest): return
    im = Image.open(os.path.join(path, fname)).convert('RGB')
    os.makedirs(os.path.split(dest)[0], exist_ok=True)
    fn(im).save(dest)

def resize_fn(targ):
    def resize(im):
        r,c = im.size
        ratio = targ/min(r,c)
        sz = (scale_to(r, ratio, targ), scale_to(c, ratio, targ))
        return im.resize(sz, Image.LINEAR)
    return resize


def resize_imgs(fnames, targ, path, new_path, resume=True, fn=None):
    """
    Enlarge or shrink a set of images in the same directory to scale, such that the smaller of the height or width dimension is equal to targ.
    Note: 
    -- This function is multithreaded for efficiency. 
    -- When destination file or folder already exist, function exists without raising an error. 
    """
    target_path = path_for(path, new_path, targ)
    if resume:
        subdirs = {os.path.dirname(p) for p in fnames}
        subdirs = {s for s in subdirs if os.path.exists(os.path.join(target_path, s))}
        already_resized_fnames = set()
        for subdir in subdirs:
            files = [os.path.join(subdir, file) for file in os.listdir(os.path.join(target_path, subdir))]
            already_resized_fnames.update(set(files))
        original_fnames = set(fnames)
        fnames = list(original_fnames - already_resized_fnames)
    
    errors = {}
    def safely_process(fname):
        try:
            resize_img(fname, targ, path, new_path, fn=fn)
        except Exception as ex:
            errors[fname] = str(ex)

    if len(fnames) > 0:
        with ThreadPoolExecutor(num_cpus()) as e:
            ims = e.map(lambda fname: safely_process(fname), fnames)
            for _ in tqdm(ims, total=len(fnames), leave=False): pass
    if errors:
        print('Some images failed to process:')
        print(json.dumps(errors, indent=2))
    return os.path.join(path,new_path,str(targ))

def read_dir(path, folder):
    """ Returns a list of relative file paths to `path` for all files within `folder` """
    full_path = os.path.join(path, folder)
    fnames = glob(f"{full_path}/*.*")
    directories = glob(f"{full_path}/*/")
    if any(fnames):
        return [os.path.relpath(f,path) for f in fnames]
    elif any(directories):
        raise FileNotFoundError("{} has subdirectories but contains no files. Is your directory structure is correct?".format(full_path))
    else:
        raise FileNotFoundError("{} folder doesn't exist or is empty".format(full_path))

def read_dirs(path, folder):
    '''
    Fetches name of all files in path in long form, and labels associated by extrapolation of directory names. 
    '''
    lbls, fnames, all_lbls = [], [], []
    full_path = os.path.join(path, folder)
    for lbl in sorted(os.listdir(full_path)):
        if lbl not in ('.ipynb_checkpoints','.DS_Store'):
            all_lbls.append(lbl)
            for fname in os.listdir(os.path.join(full_path, lbl)):
                if fname not in ('.DS_Store'):
                    fnames.append(os.path.join(folder, lbl, fname))
                    lbls.append(lbl)
    return fnames, lbls, all_lbls

def n_hot(ids, c):
    '''
    one hot encoding by index. Returns array of length c, where all entries are 0, except for the indecies in ids
    '''
    res = np.zeros((c,), dtype=np.float32)
    res[ids] = 1
    return res

def folder_source(path, folder):
    """
    Returns the filenames and labels for a folder within a path
    
    Returns:
    -------
    fnames: a list of the filenames within `folder`
    all_lbls: a list of all of the labels in `folder`, where the # of labels is determined by the # of directories within `folder`
    lbl_arr: a numpy array of the label indices in `all_lbls`
    """
    fnames, lbls, all_lbls = read_dirs(path, folder)
    lbl2idx = {lbl:idx for idx,lbl in enumerate(all_lbls)}
    idxs = [lbl2idx[lbl] for lbl in lbls]
    lbl_arr = np.array(idxs, dtype=int)
    return fnames, lbl_arr, all_lbls

def parse_csv_labels(fn, skip_header=True, cat_separator = ' '):
    """Parse filenames and label sets from a CSV file.

    This method expects that the csv file at path :fn: has two columns. If it
    has a header, :skip_header: should be set to True. The labels in the
    label set are expected to be space separated.

    Arguments:
        fn: Path to a CSV file.
        skip_header: A boolean flag indicating whether to skip the header.

    Returns:
        a two-tuple of (
            image filenames,
            a dictionary of filenames and corresponding labels
        )
    .
    :param cat_separator: the separator for the categories column
    """
    df = pd.read_csv(fn, index_col=0, header=0 if skip_header else None, dtype=str)
    fnames = df.index.values
    df.iloc[:,0] = df.iloc[:,0].str.split(cat_separator)
    return fnames, list(df.to_dict().values())[0]

def nhot_labels(label2idx, csv_labels, fnames, c):
                
    all_idx = {k: n_hot([label2idx[o] for o in ([] if type(v) == float else v)], c)
               for k,v in csv_labels.items()}
    return np.stack([all_idx[o] for o in fnames])

def csv_source(folder, csv_file, skip_header=True, suffix='', continuous=False, cat_separator=' '):
    fnames,csv_labels = parse_csv_labels(csv_file, skip_header, cat_separator)
    return dict_source(folder, fnames, csv_labels, suffix, continuous)

def dict_source(folder, fnames, csv_labels, suffix='', continuous=False):
    all_labels = sorted(list(set(p for o in csv_labels.values() for p in ([] if type(o) == float else o))))
    full_names = [os.path.join(folder,str(fn)+suffix) for fn in fnames]
    if continuous:
        label_arr = np.array([np.array(csv_labels[i]).astype(np.float32)
                for i in fnames])
    else:
        label2idx = {v:k for k,v in enumerate(all_labels)}
        label_arr = nhot_labels(label2idx, csv_labels, fnames, len(all_labels))
        is_single = np.all(label_arr.sum(axis=1)==1)
        if is_single: label_arr = np.argmax(label_arr, axis=1)
    return full_names, label_arr, all_labels

class BaseDataset(Dataset):
    """An abstract class representing a fastai dataset. Extends torch.utils.data.Dataset."""
    def __init__(self, transform=None):
        self.transform = transform
        self.n = self.get_n()
        self.c = self.get_c()
        self.sz = self.get_sz()

    def get1item(self, idx):
        x,y = self.get_x(idx),self.get_y(idx)
        return self.get(self.transform, x, y)

    def __getitem__(self, idx):
        if isinstance(idx,slice):
            xs,ys = zip(*[self.get1item(i) for i in range(*idx.indices(self.n))])
            return np.stack(xs),ys
        return self.get1item(idx)

    def __len__(self): return self.n

    def get(self, tfm, x, y):
        return (x,y) if tfm is None else tfm(x,y)

    @abstractmethod
    def get_n(self):
        """Return number of elements in the dataset == len(self)."""
        raise NotImplementedError

    @abstractmethod
    def get_c(self):
        """Return number of classes in a dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_sz(self):
        """Return maximum size of an image in a dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_x(self, i):
        """Return i-th example (image, wav, etc)."""
        raise NotImplementedError

    @abstractmethod
    def get_y(self, i):
        """Return i-th label."""
        raise NotImplementedError

    @property
    def is_multi(self):
        """Returns true if this data set contains multiple labels per sample."""
        return False

    @property
    def is_reg(self):
        """True if the data set is used to train regression models."""
        return False

def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn) and not str(fn).startswith("http"):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn) and not str(fn).startswith("http"):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            if str(fn).startswith("http"):
                req = urllib.urlopen(str(fn))
                image = np.asarray(bytearray(req.read()), dtype="uint8")
                im = cv2.imdecode(image, flags).astype(np.float32)/255
            else:
                im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

class FilesDataset(BaseDataset):
    def __init__(self, fnames, transform, path):
        self.path,self.fnames = path,fnames
        super().__init__(transform)
    def get_sz(self): return self.transform.sz
    def get_x(self, i): return open_image(os.path.join(self.path, self.fnames[i]))
    def get_n(self): return len(self.fnames)

    def resize_imgs(self, targ, new_path, resume=True, fn=None):
        """
        resize all images in the dataset and save them to `new_path`
        
        Arguments:
        targ (int): the target size
        new_path (string): the new folder to save the images
        resume (bool): if true (default), allow resuming a partial resize operation by checking for the existence
        of individual images rather than the existence of the directory
        fn (function): custom resizing function Img -> Img
        """
        dest = resize_imgs(self.fnames, targ, self.path, new_path, resume, fn)
        return self.__class__(self.fnames, self.y, self.transform, dest)

    def denorm(self,arr):
        """Reverse the normalization done to a batch of images.

        Arguments:
            arr: of shape/size (N,3,sz,sz)
        """
        if type(arr) is not np.ndarray: arr = to_np(arr)
        if len(arr.shape)==3: arr = arr[None]
        return self.transform.denorm(np.rollaxis(arr,1,4))


class FilesArrayDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y=y
        assert(len(fnames)==len(y))
        super().__init__(fnames, transform, path)
    def get_y(self, i): return self.y[i]
    def get_c(self):
        return self.y.shape[1] if len(self.y.shape)>1 else 0

class FilesIndexArrayDataset(FilesArrayDataset):
    def get_c(self): return int(self.y.max())+1


class FilesNhotArrayDataset(FilesArrayDataset):
    @property
    def is_multi(self): return True


class FilesIndexArrayRegressionDataset(FilesArrayDataset):
    def is_reg(self): return True

class ArraysDataset(BaseDataset):
    def __init__(self, x, y, transform):
        self.x,self.y=x,y
        assert(len(x)==len(y))
        super().__init__(transform)
    def get_x(self, i): return self.x[i]
    def get_y(self, i): return self.y[i]
    def get_n(self): return len(self.y)
    def get_sz(self): return self.x.shape[1]


class ArraysIndexDataset(ArraysDataset):
    def get_c(self): return int(self.y.max())+1
    def get_y(self, i): return self.y[i]


class ArraysIndexRegressionDataset(ArraysIndexDataset):
    def is_reg(self): return True


class ArraysNhotDataset(ArraysDataset):
    def get_c(self): return self.y.shape[1]
    @property
    def is_multi(self): return True


class ModelData():
    """Encapsulates DataLoaders and Datasets for training, validation, test. Base class for fastai *Data classes."""
    def __init__(self, path, trn_dl, val_dl, test_dl=None):
        self.path,self.trn_dl,self.val_dl,self.test_dl = path,trn_dl,val_dl,test_dl

    @classmethod
    def from_dls(cls, path,trn_dl,val_dl,test_dl=None):
        #trn_dl,val_dl = DataLoader(trn_dl),DataLoader(val_dl)
        #if test_dl: test_dl = DataLoader(test_dl)
        return cls(path, trn_dl, val_dl, test_dl)

    @property
    def is_reg(self): return self.trn_ds.is_reg
    @property
    def is_multi(self): return self.trn_ds.is_multi
    @property
    def trn_ds(self): return self.trn_dl.dataset
    @property
    def val_ds(self): return self.val_dl.dataset
    @property
    def test_ds(self): return self.test_dl.dataset
    @property
    def trn_y(self): return self.trn_ds.y
    @property
    def val_y(self): return self.val_ds.y


class ImageData(ModelData):
    def __init__(self, path, datasets, bs, num_workers, classes):
        trn_ds,val_ds,fix_ds,aug_ds,test_ds,test_aug_ds = datasets
        self.path,self.bs,self.num_workers,self.classes = path,bs,num_workers,classes
        self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl,self.test_dl,self.test_aug_dl = [
            self.get_dl(ds,shuf) for ds,shuf in [
                (trn_ds,True),(val_ds,False),(fix_ds,False),(aug_ds,False),
                (test_ds,False),(test_aug_ds,False)
            ]
        ]

    def get_dl(self, ds, shuffle):
        if ds is None: return None
        return DataLoader(ds, batch_size=self.bs, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=False)

    @property
    def sz(self): return self.trn_ds.sz
    @property
    def c(self): return self.trn_ds.c

    def resized(self, dl, targ, new_path, resume = True, fn=None):
        """
        Return a copy of this dataset resized
        """
        return dl.dataset.resize_imgs(targ, new_path, resume=resume, fn=fn) if dl else None

    def resize(self, targ_sz, new_path='tmp', resume=True, fn=None):
        """
        Resizes all the images in the train, valid, test folders to a given size.

        Arguments:
        targ_sz (int): the target size
        new_path (str): the path to save the resized images (default tmp)
        resume (bool): if True, check for images in the DataSet that haven't been resized yet (useful if a previous resize
        operation was aborted)
        fn (function): optional custom resizing function
        """
        new_ds = []
        dls = [self.trn_dl,self.val_dl,self.fix_dl,self.aug_dl]
        if self.test_dl: dls += [self.test_dl, self.test_aug_dl]
        else: dls += [None,None]
        t = tqdm_notebook(dls)
        for dl in t: new_ds.append(self.resized(dl, targ_sz, new_path, resume, fn))
        t.close()
        return self.__class__(new_ds[0].path, new_ds, self.bs, self.num_workers, self.classes)

    @staticmethod
    def get_ds(fn, trn, val, tfms, test=None, **kwargs):
        res = [
            fn(trn[0], trn[1], tfms[0], **kwargs), # train
            fn(val[0], val[1], tfms[1], **kwargs), # val
            fn(trn[0], trn[1], tfms[1], **kwargs), # fix
            fn(val[0], val[1], tfms[0], **kwargs)  # aug
        ]
        if test is not None:
            if isinstance(test, tuple):
                test_lbls = test[1]
                test = test[0]
            else:
                if len(trn[1].shape) == 1:
                    test_lbls = np.zeros((len(test),1))
                else:
                    test_lbls = np.zeros((len(test),trn[1].shape[1]))
            res += [
                fn(test, test_lbls, tfms[1], **kwargs), # test
                fn(test, test_lbls, tfms[0], **kwargs)  # test_aug
            ]
        else: res += [None,None]
        return res


class ImageClassifierData(ImageData):
    @classmethod
    def from_arrays(cls, path, trn, val, bs=64, tfms=(None,None), classes=None, num_workers=4, test=None, continuous=False):
        """ Read in images and their labels given as numpy arrays

        Arguments:
            path: a root path of the data (used for storing trained models, precomputed values, etc)
            trn: a tuple of training data matrix and target label/classification array (e.g. `trn=(x,y)` where `x` has the
                shape of `(5000, 784)` and `y` has the shape of `(5000,)`)
            val: a tuple of validation data matrix and target label/classification array.
            bs: batch size
            tfms: transformations (for data augmentations). e.g. output of `tfms_from_model`
            classes: a list of all labels/classifications
            num_workers: a number of workers
            test: a matrix of test data (the shape should match `trn[0]`)

        Returns:
            ImageClassifierData
        """
        f = ArraysIndexRegressionDataset if continuous else ArraysIndexDataset
        datasets = cls.get_ds(f, trn, val, tfms, test=test)
        return cls(path, datasets, bs, num_workers, classes=classes)

    @classmethod
    def from_paths(cls, path, bs=64, tfms=(None,None), trn_name='train', val_name='valid', test_name=None, test_with_labels=False, num_workers=8):
        """ Read in images and their labels given as sub-folder names

        Arguments:
            path: a root path of the data (used for storing trained models, precomputed values, etc)
            bs: batch size
            tfms: transformations (for data augmentations). e.g. output of `tfms_from_model`
            trn_name: a name of the folder that contains training images.
            val_name:  a name of the folder that contains validation images.
            test_name:  a name of the folder that contains test images.
            num_workers: number of workers

        Returns:
            ImageClassifierData
        """
        assert not(tfms[0] is None or tfms[1] is None), "please provide transformations for your train and validation sets"
        trn,val = [folder_source(path, o) for o in (trn_name, val_name)]
        if test_name:
            test = folder_source(path, test_name) if test_with_labels else read_dir(path, test_name)
        else: test = None
        datasets = cls.get_ds(FilesIndexArrayDataset, trn, val, tfms, path=path, test=test)
        return cls(path, datasets, bs, num_workers, classes=trn[2])

    @classmethod
    def from_csv(cls, path, folder, csv_fname, bs=64, tfms=(None,None),
               val_idxs=None, suffix='', test_name=None, continuous=False, skip_header=True, num_workers=8, cat_separator=' '):
        """ Read in images and their labels given as a CSV file.

        This method should be used when training image labels are given in an CSV file as opposed to
        sub-directories with label names.

        Arguments:
            path: a root path of the data (used for storing trained models, precomputed values, etc)
            folder: a name of the folder in which training images are contained.
            csv_fname: a name of the CSV file which contains target labels.
            bs: batch size
            tfms: transformations (for data augmentations). e.g. output of `tfms_from_model`
            val_idxs: index of images to be used for validation. e.g. output of `get_cv_idxs`.
                If None, default arguments to get_cv_idxs are used.
            suffix: suffix to add to image names in CSV file (sometimes CSV only contains the file name without file
                    extension e.g. '.jpg' - in which case, you can set suffix as '.jpg')
            test_name: a name of the folder which contains test images.
            continuous: TODO
            skip_header: skip the first row of the CSV file.
            num_workers: number of workers
            cat_separator: Labels category separator

        Returns:
            ImageClassifierData
        """
        assert not (tfms[0] is None or tfms[1] is None), "please provide transformations for your train and validation sets"
        assert not (os.path.isabs(folder)), "folder needs to be a relative path"
        fnames,y,classes = csv_source(folder, csv_fname, skip_header, suffix, continuous=continuous, cat_separator=cat_separator)
        return cls.from_names_and_array(path, fnames, y, classes, val_idxs, test_name,
                num_workers=num_workers, suffix=suffix, tfms=tfms, bs=bs, continuous=continuous)

    @classmethod
    def from_path_and_array(cls, path, folder, y, classes=None, val_idxs=None, test_name=None,
            num_workers=8, tfms=(None,None), bs=64):
        """ Read in images given a sub-folder and their labels given a numpy array

        Arguments:
            path: a root path of the data (used for storing trained models, precomputed values, etc)
            folder: a name of the folder in which training images are contained.
            y: numpy array which contains target labels ordered by filenames.
            bs: batch size
            tfms: transformations (for data augmentations). e.g. output of `tfms_from_model`
            val_idxs: index of images to be used for validation. e.g. output of `get_cv_idxs`.
                If None, default arguments to get_cv_idxs are used.
            test_name: a name of the folder which contains test images.
            num_workers: number of workers

        Returns:
            ImageClassifierData
        """
        assert not (tfms[0] is None or tfms[1] is None), "please provide transformations for your train and validation sets"
        assert not (os.path.isabs(folder)), "folder needs to be a relative path"
        fnames = np.core.defchararray.add(f'{folder}/', sorted(os.listdir(f'{path}{folder}')))
        return cls.from_names_and_array(path, fnames, y, classes, val_idxs, test_name,
                num_workers=num_workers, tfms=tfms, bs=bs)

    @classmethod
    def from_names_and_array(cls, path, fnames, y, classes, val_idxs=None, test_name=None,
            num_workers=8, suffix='', tfms=(None,None), bs=64, continuous=False):
        val_idxs = get_cv_idxs(len(fnames)) if val_idxs is None else val_idxs
        ((val_fnames,trn_fnames),(val_y,trn_y)) = split_by_idx(val_idxs, np.array(fnames), y)

        test_fnames = read_dir(path, test_name) if test_name else None
        if continuous: f = FilesIndexArrayRegressionDataset
        else:
            f = FilesIndexArrayDataset if len(trn_y.shape)==1 else FilesNhotArrayDataset
        datasets = cls.get_ds(f, (trn_fnames,trn_y), (val_fnames,val_y), tfms,
                               path=path, test=test_fnames)
        return cls(path, datasets, bs, num_workers, classes=classes)

def split_by_idx(idxs, *a):
    """
    Split each array passed as *a, to a pair of arrays like this (elements selected by idxs,  the remaining elements)
    This can be used to split multiple arrays containing training data to validation and training set.

    :param idxs [int]: list of indexes selected
    :param a list: list of np.array, each array should have same amount of elements in the first dimension
    :return: list of tuples, each containing a split of corresponding array from *a.
            First element of each tuple is an array composed from elements selected by idxs,
            second element is an array of remaining elements.
    """
    mask = np.zeros(len(a[0]),dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask],o[~mask]) for o in a]
# end fastai.dataset################################################################


# start fastai.layer_optimizer#################################################################
def opt_params(parm, lr, wd):
    return {'params': chain_params(parm), 'lr':lr, 'weight_decay':wd}

class LayerOptimizer():
    def __init__(self, opt_fn, layer_groups, lrs, wds=None):
        if not isinstance(layer_groups, (list,tuple)): layer_groups=[layer_groups]
        lrs = listify(lrs, layer_groups)
        if wds is None: wds=0.
        wds = listify(wds, layer_groups)
        self.layer_groups,self.lrs,self.wds = layer_groups,lrs,wds
        self.opt = opt_fn(self.opt_params())

    def opt_params(self):
        assert len(self.layer_groups) == len(self.lrs), f'size mismatch, expected {len(self.layer_groups)} lrs, but got {len(self.lrs)}'
        assert len(self.layer_groups) == len(self.wds), f'size mismatch, expected {len(self.layer_groups)} wds, but got {len(self.wds)}'
        params = list(zip(self.layer_groups,self.lrs,self.wds))
        return [opt_params(*p) for p in params]

    @property
    def lr(self): return self.lrs[-1]

    @property
    def mom(self):
        if 'betas' in self.opt.param_groups[0]:
            return self.opt.param_groups[0]['betas'][0]
        else:
            return self.opt.param_groups[0]['momentum']

    def set_lrs(self, lrs):
        lrs = listify(lrs, self.layer_groups)
        set_lrs(self.opt, lrs)
        self.lrs=lrs

    def set_wds_out(self, wds):
        wds = listify(wds, self.layer_groups)
        set_wds_out(self.opt, wds)
        set_wds(self.opt, [0] * len(self.layer_groups))
        self.wds=wds

    def set_wds(self, wds):
        wds = listify(wds, self.layer_groups)
        set_wds(self.opt, wds)
        set_wds_out(self.opt, [0] * len(self.layer_groups))
        self.wds=wds
    
    def set_mom(self,momentum):
        if 'betas' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: pg['betas'] = (momentum, pg['betas'][1])
        else:
            for pg in self.opt.param_groups: pg['momentum'] = momentum
    
    def set_beta(self,beta):
        if 'betas' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: pg['betas'] = (pg['betas'][0],beta)
        elif 'alpha' in self.opt.param_groups[0]:
            for pg in self.opt.param_groups: pg['alpha'] = beta

    def set_opt_fn(self, opt_fn):
        if type(self.opt) != type(opt_fn(self.opt_params())):
            self.opt = opt_fn(self.opt_params())

def zip_strict_(l, r):
    assert len(l) == len(r), f'size mismatch, expected {len(l)} r, but got {len(r)} r'
    return zip(l, r)

def set_lrs(opt, lrs):
    lrs = listify(lrs, opt.param_groups)
    for pg,lr in zip_strict_(opt.param_groups,lrs): pg['lr'] = lr

def set_wds_out(opt, wds):
    wds = listify(wds, opt.param_groups)
    assert len(opt.param_groups) == len(wds), f'size mismatch, expected {len(opt.param_groups)} wds, but got {len(wds)}'
    for pg,wd in zip_strict_(opt.param_groups,wds): pg['wd'] = wd

def set_wds(opt, wds):
    wds = listify(wds, opt.param_groups)
    assert len(opt.param_groups) == len(wds), f'size mismatch, expected {len(opt.param_groups)} wds, but got {len(wds)}'
    for pg,wd in zip_strict_(opt.param_groups,wds): pg['weight_decay'] = wd
# end fastai.layer_optimizer################################################################


# start fastai.sgdr#################################################################

class Callback:
    '''
    An abstract class that all callback(e.g., LossRecorder) classes extends from. 
    Must be extended before usage.
    '''
    def on_train_begin(self): pass
    def on_epoch_begin(self): pass
    def on_batch_begin(self): pass
    def on_phase_begin(self): pass
    def on_epoch_end(self, metrics): pass
    def on_phase_end(self): pass
    def on_batch_end(self, metrics): pass
    def on_train_end(self): pass

# Useful for maintaining status of a long-running job.
# 
# Usage:
# learn.fit(0.01, 1, callbacks = [LoggingCallback(save_path="/tmp/log")])
class LoggingCallback(Callback):
    '''
    A class useful for maintaining status of a long-running job.
    e.g.: learn.fit(0.01, 1, callbacks = [LoggingCallback(save_path="/tmp/log")])
    '''
    def __init__(self, save_path):
        super().__init__()
        self.save_path=save_path
    def on_train_begin(self):
        self.batch = 0
        self.epoch = 0
        self.phase = 0
        self.f = open(self.save_path, "a", 1)
        self.log("\ton_train_begin")
    def on_batch_begin(self):
        self.log(str(self.batch)+"\ton_batch_begin")
    def on_phase_begin(self):
        self.log(str(self.phase)+"\ton_phase_begin")
    def on_epoch_end(self, metrics):
        self.log(str(self.epoch)+"\ton_epoch_end: "+str(metrics))
        self.epoch += 1
    def on_phase_end(self):
        self.log(str(self.phase)+"\ton_phase_end")
        self.phase+=1
    def on_batch_end(self, metrics):
        self.log(str(self.batch)+"\ton_batch_end: "+str(metrics))
        self.batch += 1
    def on_train_end(self):
        self.log("\ton_train_end")
        self.f.close()
    def log(self, string):
        self.f.write(time.strftime("%Y-%m-%dT%H:%M:%S")+"\t"+string+"\n")
        
class LossRecorder(Callback):
    '''
    Saves and displays loss functions and other metrics. 
    Default sched when none is specified in a learner. 
    '''
    def __init__(self, layer_opt, save_path='', record_mom=False, metrics=[]):
        super().__init__()
        self.layer_opt=layer_opt
        self.init_lrs=np.array(layer_opt.lrs)
        self.save_path, self.record_mom, self.metrics = save_path, record_mom, metrics

    def on_train_begin(self):
        self.losses,self.lrs,self.iterations,self.epochs,self.times = [],[],[],[],[]
        self.start_at = timer()
        self.val_losses, self.rec_metrics = [], []
        if self.record_mom:
            self.momentums = []
        self.iteration = 0
        self.epoch = 0

    def on_epoch_end(self, metrics):
        self.epoch += 1
        self.epochs.append(self.iteration)
        self.times.append(timer() - self.start_at)
        self.save_metrics(metrics)

    def on_batch_end(self, loss):
        self.iteration += 1
        self.lrs.append(self.layer_opt.lr)
        self.iterations.append(self.iteration)
        if isinstance(loss, list):
            self.losses.append(loss[0])
            self.save_metrics(loss[1:])
        else: self.losses.append(loss)
        if self.record_mom: self.momentums.append(self.layer_opt.mom)

    def save_metrics(self,vals):
        self.val_losses.append(delistify(vals[0]))
        if len(vals) > 2: self.rec_metrics.append(vals[1:])
        elif len(vals) == 2: self.rec_metrics.append(vals[1])

    def plot_loss(self, n_skip=10, n_skip_end=5):
        '''
        plots loss function as function of iterations. 
        When used in Jupyternotebook, plot will be displayed in notebook. Else, plot will be displayed in console and both plot and loss are saved in save_path. 
        '''
        if not in_ipynb(): plt.switch_backend('agg')
        plt.plot(self.iterations[n_skip:-n_skip_end], self.losses[n_skip:-n_skip_end])
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'loss_plot.png'))
            np.save(os.path.join(self.save_path, 'losses.npy'), self.losses[10:])

    def plot_lr(self):
        '''Plots learning rate in jupyter notebook or console, depending on the enviroment of the learner.'''
        if not in_ipynb():
            plt.switch_backend('agg')
        if self.record_mom:
            fig, axs = plt.subplots(1,2,figsize=(12,4))
            for i in range(0,2): axs[i].set_xlabel('iterations')
            axs[0].set_ylabel('learning rate')
            axs[1].set_ylabel('momentum')
            axs[0].plot(self.iterations,self.lrs)
            axs[1].plot(self.iterations,self.momentums)   
        else:
            plt.xlabel("iterations")
            plt.ylabel("learning rate")
            plt.plot(self.iterations, self.lrs)
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))


class LR_Updater(LossRecorder):
    '''
    Abstract class where all Learning Rate updaters inherit from. (e.g., CirularLR)
    Calculates and updates new learning rate and momentum at the end of each batch. 
    Have to be extended. 
    '''
    def on_train_begin(self):
        super().on_train_begin()
        self.update_lr()
        if self.record_mom:
            self.update_mom()

    def on_batch_end(self, loss):
        res = super().on_batch_end(loss)
        self.update_lr()
        if self.record_mom:
            self.update_mom()
        return res

    def update_lr(self):
        new_lrs = self.calc_lr(self.init_lrs)
        self.layer_opt.set_lrs(new_lrs)
    
    def update_mom(self):
        new_mom = self.calc_mom()
        self.layer_opt.set_mom(new_mom)

    @abstractmethod
    def calc_lr(self, init_lrs): raise NotImplementedError
    
    @abstractmethod
    def calc_mom(self): raise NotImplementedError


class LR_Finder(LR_Updater):
    '''
    Helps you find an optimal learning rate for a model, as per suggetion of 2015 CLR paper. 
    Learning rate is increased in linear or log scale, depending on user input, and the result of the loss funciton is retained and can be plotted later. 
    '''
    def __init__(self, layer_opt, nb, end_lr=10, linear=False, metrics = []):
        self.linear, self.stop_dv = linear, True
        ratio = end_lr/layer_opt.lr
        self.lr_mult = (ratio/nb) if linear else ratio**(1/nb)
        super().__init__(layer_opt,metrics=metrics)

    def on_train_begin(self):
        super().on_train_begin()
        self.best=1e9

    def calc_lr(self, init_lrs):
        mult = self.lr_mult*self.iteration if self.linear else self.lr_mult**self.iteration
        return init_lrs * mult

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics,list) else metrics
        if self.stop_dv and (math.isnan(loss) or loss>self.best*4):
            return True
        if (loss<self.best and self.iteration>10): self.best=loss
        return super().on_batch_end(metrics)

    def plot(self, n_skip=10, n_skip_end=5):
        '''
        Plots the loss function with respect to learning rate, in log scale. 
        '''
        plt.ylabel("validation loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip:-(n_skip_end+1)], self.losses[n_skip:-(n_skip_end+1)])
        plt.xscale('log')

class LR_Finder2(LR_Finder):
    """
        A variant of lr_find() that helps find the best learning rate. It doesn't do
        an epoch but a fixed num of iterations (which may be more or less than an epoch
        depending on your data).
    """
    def __init__(self, layer_opt, nb, end_lr=10, linear=False, metrics=[], stop_dv=True):
        self.nb, self.metrics = nb, metrics
        super().__init__(layer_opt, nb, end_lr, linear, metrics)
        self.stop_dv = stop_dv

    def on_batch_end(self, loss):
        if self.iteration == self.nb:
            return True
        return super().on_batch_end(loss)

    def plot(self, n_skip=10, n_skip_end=5, smoothed=True):
        if self.metrics is None: self.metrics = []
        n_plots = len(self.metrics)+2
        fig, axs = plt.subplots(n_plots,figsize=(6,4*n_plots))
        for i in range(0,n_plots): axs[i].set_xlabel('learning rate')
        axs[0].set_ylabel('training loss')
        axs[1].set_ylabel('validation loss')
        for i,m in enumerate(self.metrics): 
            axs[i+2].set_ylabel(m.__name__)
            if len(self.metrics) == 1:
                values = self.rec_metrics
            else:
                values = [rec[i] for rec in self.rec_metrics]
            if smoothed: values = smooth_curve(values,0.98)
            axs[i+2].plot(self.lrs[n_skip:-n_skip_end], values[n_skip:-n_skip_end])
        plt_val_l = smooth_curve(self.val_losses, 0.98) if smoothed else self.val_losses
        axs[0].plot(self.lrs[n_skip:-n_skip_end],self.losses[n_skip:-n_skip_end])
        axs[1].plot(self.lrs[n_skip:-n_skip_end],plt_val_l[n_skip:-n_skip_end])

class CosAnneal(LR_Updater):
    ''' Learning rate scheduler that implements a cosine annealation schedule. '''
    def __init__(self, layer_opt, nb, on_cycle_end=None, cycle_mult=1):
        self.nb,self.on_cycle_end,self.cycle_mult = nb,on_cycle_end,cycle_mult
        super().__init__(layer_opt)

    def on_train_begin(self):
        self.cycle_iter,self.cycle_count=0,0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        if self.iteration<self.nb/20:
            self.cycle_iter += 1
            return init_lrs/100.

        cos_out = np.cos(np.pi*(self.cycle_iter)/self.nb) + 1
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            self.nb *= self.cycle_mult
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return init_lrs / 2 * cos_out


class CircularLR(LR_Updater):
    '''
    A learning rate updater that implements the CircularLearningRate (CLR) scheme. 
    Learning rate is increased then decreased linearly. 
    '''
    def __init__(self, layer_opt, nb, div=4, cut_div=8, on_cycle_end=None, momentums=None):
        self.nb,self.div,self.cut_div,self.on_cycle_end = nb,div,cut_div,on_cycle_end
        if momentums is not None:
            self.moms = momentums
        super().__init__(layer_opt, record_mom=(momentums is not None))

    def on_train_begin(self):
        self.cycle_iter,self.cycle_count=0,0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        cut_pt = self.nb//self.cut_div
        if self.cycle_iter>cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt)/(self.nb - cut_pt)
        else: pct = self.cycle_iter/cut_pt
        res = init_lrs * (1 + pct*(self.div-1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res
    
    def calc_mom(self):
        cut_pt = self.nb//self.cut_div
        if self.cycle_iter>cut_pt:
            pct = (self.cycle_iter - cut_pt)/(self.nb - cut_pt)
        else: pct = 1 - self.cycle_iter/cut_pt
        res = self.moms[1] + pct * (self.moms[0] - self.moms[1])
        return res

class CircularLR_beta(LR_Updater):
    def __init__(self, layer_opt, nb, div=10, pct=10, on_cycle_end=None, momentums=None):
        self.nb,self.div,self.pct,self.on_cycle_end = nb,div,pct,on_cycle_end
        self.cycle_nb = int(nb * (1-pct/100) / 2)
        if momentums is not None:
            self.moms = momentums
        super().__init__(layer_opt, record_mom=(momentums is not None))

    def on_train_begin(self):
        self.cycle_iter,self.cycle_count=0,0
        super().on_train_begin()

    def calc_lr(self, init_lrs):
        if self.cycle_iter>2 * self.cycle_nb:
            pct = (self.cycle_iter - 2*self.cycle_nb)/(self.nb - 2*self.cycle_nb)
            res = init_lrs * (1 + (pct * (1-100)/100)) / self.div
        elif self.cycle_iter>self.cycle_nb:
            pct = 1 - (self.cycle_iter - self.cycle_nb)/self.cycle_nb
            res = init_lrs * (1 + pct*(self.div-1)) / self.div
        else:
            pct = self.cycle_iter/self.cycle_nb
            res = init_lrs * (1 + pct*(self.div-1)) / self.div
        self.cycle_iter += 1
        if self.cycle_iter==self.nb:
            self.cycle_iter = 0
            if self.on_cycle_end: self.on_cycle_end(self, self.cycle_count)
            self.cycle_count += 1
        return res

    def calc_mom(self):
        if self.cycle_iter>2*self.cycle_nb:
            res = self.moms[0]
        elif self.cycle_iter>self.cycle_nb:
            pct = 1 - (self.cycle_iter - self.cycle_nb)/self.cycle_nb
            res = self.moms[0] + pct * (self.moms[1] - self.moms[0])
        else:
            pct = self.cycle_iter/self.cycle_nb
            res = self.moms[0] + pct * (self.moms[1] - self.moms[0])
        return res


class SaveBestModel(LossRecorder):
    
    """ Save weights of the best model based during training.
        If metrics are provided, the first metric in the list is used to
        find the best model. 
        If no metrics are provided, the loss is used.
        
        Args:
            model: the fastai model
            lr: indicate to use test images; otherwise use validation images
            name: the name of filename of the weights without '.h5'
        
        Usage:
            Briefly, you have your model 'learn' variable and call fit.
            >>> learn.fit(lr, 2, cycle_len=2, cycle_mult=1, best_save_name='mybestmodel')
            ....
            >>> learn.load('mybestmodel')
            
            For more details see http://forums.fast.ai/t/a-code-snippet-to-save-the-best-model-during-training/12066
 
    """
    def __init__(self, model, layer_opt, metrics, name='best_model'):
        super().__init__(layer_opt)
        self.name = name
        self.model = model
        self.best_loss = None
        self.best_acc = None
        self.save_method = self.save_when_only_loss if metrics==None else self.save_when_acc
        
    def save_when_only_loss(self, metrics):
        loss = metrics[0]
        if self.best_loss == None or loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')
    
    def save_when_acc(self, metrics):
        loss, acc = metrics[0], metrics[1]
        if self.best_acc == None or acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.model.save(f'{self.name}')
        elif acc == self.best_acc and  loss < self.best_loss:
            self.best_loss = loss
            self.model.save(f'{self.name}')
        
    def on_epoch_end(self, metrics):
        super().on_epoch_end(metrics)
        if math.isnan(metrics[0]): return
        self.save_method(metrics)


class WeightDecaySchedule(Callback):
    def __init__(self, layer_opt, batch_per_epoch, cycle_len, cycle_mult, n_cycles, norm_wds=False, wds_sched_mult=None):
        """
        Implements the weight decay schedule as mentioned in https://arxiv.org/abs/1711.05101

        :param layer_opt: The LayerOptimizer
        :param batch_per_epoch: Num batches in 1 epoch
        :param cycle_len: Num epochs in initial cycle. Subsequent cycle_len = previous cycle_len * cycle_mult
        :param cycle_mult: Cycle multiplier
        :param n_cycles: Number of cycles to be executed
        """
        super().__init__()

        self.layer_opt = layer_opt
        self.batch_per_epoch = batch_per_epoch
        self.init_wds = np.array(layer_opt.wds)  # Weights as set by user
        self.init_lrs = np.array(layer_opt.lrs)  # Learning rates as set by user
        self.new_wds = None                      # Holds the new weight decay factors, calculated in on_batch_begin()
        self.iteration = 0
        self.epoch = 0
        self.wds_sched_mult = wds_sched_mult
        self.norm_wds = norm_wds
        self.wds_history = list()

        # Pre calculating the number of epochs in the cycle of current running epoch
        self.epoch_to_num_cycles, i = dict(), 0
        for cycle in range(n_cycles):
            for _ in range(cycle_len):
                self.epoch_to_num_cycles[i] = cycle_len
                i += 1
            cycle_len *= cycle_mult

    def on_train_begin(self):
        self.iteration = 0
        self.epoch = 0

    def on_batch_begin(self):
        # Prepare for decay of weights

        # Default weight decay (as provided by user)
        wdn = self.init_wds

        # Weight decay multiplier (The 'eta' in the paper). Optional.
        wdm = 1.0
        if self.wds_sched_mult is not None:
            wdm = self.wds_sched_mult(self)

        # Weight decay normalized. Optional.
        if self.norm_wds:
            wdn = wdn / np.sqrt(self.batch_per_epoch * self.epoch_to_num_cycles[self.epoch])

        # Final wds
        self.new_wds = wdm * wdn

        # Set weight_decay with zeros so that it is not applied in Adam, we will apply it outside in on_batch_end()
        self.layer_opt.set_wds_out(self.new_wds)
        # We have to save the existing weights before the optimizer changes the values
        self.iteration += 1

    def on_epoch_end(self, metrics):
        self.epoch += 1

class DecayType(IntEnum):
    ''' Data class, each decay type is assigned a number. '''
    NO = 1
    LINEAR = 2
    COSINE = 3
    EXPONENTIAL = 4
    POLYNOMIAL = 5

class DecayScheduler():
    '''Given initial and endvalue, this class generates the next value depending on decay type and number of iterations. (by calling next_val().) '''

    def __init__(self, dec_type, num_it, start_val, end_val=None, extra=None):
        self.dec_type, self.nb, self.start_val, self.end_val, self.extra = dec_type, num_it, start_val, end_val, extra
        self.it = 0
        if self.end_val is None and not (self.dec_type in [1,4]): self.end_val = 0
    
    def next_val(self):
        self.it += 1
        if self.dec_type == DecayType.NO:
            return self.start_val
        elif self.dec_type == DecayType.LINEAR:
            pct = self.it/self.nb
            return self.start_val + pct * (self.end_val-self.start_val)
        elif self.dec_type == DecayType.COSINE:
            cos_out = np.cos(np.pi*(self.it)/self.nb) + 1
            return self.end_val + (self.start_val-self.end_val) / 2 * cos_out
        elif self.dec_type == DecayType.EXPONENTIAL:
            ratio = self.end_val / self.start_val
            return self.start_val * (ratio **  (self.it/self.nb))
        elif self.dec_type == DecayType.POLYNOMIAL:
            return self.end_val + (self.start_val-self.end_val) * (1 - self.it/self.nb)**self.extra
        

class TrainingPhase():
    '''
    Object with training information for each phase, when multiple phases are involved during training.  
    Used in fit_opt_sched in learner.py
    '''
    def __init__(self, epochs=1, opt_fn=optim.SGD, lr=1e-2, lr_decay=DecayType.NO, momentum=0.9,
                momentum_decay=DecayType.NO, beta=None, wds=None, wd_loss=True):
        """
        Creates an object containing all the relevant informations for one part of a model training.

        Args
        epochs: number of epochs to train like this
        opt_fn: an optimizer (example optim.Adam)
        lr: one learning rate or a tuple of the form (start_lr,end_lr)
          each of those can be a list/numpy array for differential learning rates
        lr_decay: a DecayType object specifying how the learning rate should change
        momentum: one momentum (or beta1 in case of Adam), or a tuple of the form (start_mom,end_mom)
        momentum_decay: a DecayType object specifying how the momentum should change
        beta: beta2 parameter of Adam or alpha parameter of RMSProp
        wds: weight decay (can be an array for differential wds)
        """
        self.epochs, self.opt_fn, self.lr, self.momentum, self.beta, self.wds = epochs, opt_fn, lr, momentum, beta, wds
        if isinstance(lr_decay,tuple): self.lr_decay, self.extra_lr = lr_decay
        else: self.lr_decay, self.extra_lr = lr_decay, None
        if isinstance(momentum_decay,tuple): self.mom_decay, self.extra_mom = momentum_decay
        else: self.mom_decay, self.extra_mom = momentum_decay, None
        self.wd_loss = wd_loss

    def phase_begin(self, layer_opt, nb_batches):
        self.layer_opt = layer_opt
        if isinstance(self.lr, tuple): start_lr,end_lr = self.lr
        else: start_lr, end_lr = self.lr, None
        self.lr_sched = DecayScheduler(self.lr_decay, nb_batches * self.epochs, start_lr, end_lr, extra=self.extra_lr)
        if isinstance(self.momentum, tuple): start_mom,end_mom = self.momentum
        else: start_mom, end_mom = self.momentum, None
        self.mom_sched = DecayScheduler(self.mom_decay, nb_batches * self.epochs, start_mom, end_mom, extra=self.extra_mom)
        self.layer_opt.set_opt_fn(self.opt_fn)
        self.layer_opt.set_lrs(start_lr)
        self.layer_opt.set_mom(start_mom)
        if self.beta is not None: self.layer_opt.set_beta(self.beta)
        if self.wds is not None:
            if self.wd_loss: self.layer_opt.set_wds(self.wds)
            else: self.layer_opt.set_wds_out(self.wds)
    
    def update(self):
        new_lr, new_mom = self.lr_sched.next_val(), self.mom_sched.next_val()
        self.layer_opt.set_lrs(new_lr)
        self.layer_opt.set_mom(new_mom)
    

class OptimScheduler(LossRecorder):
    '''Learning rate Scheduler for training involving multiple phases.'''

    def __init__(self, layer_opt, phases, nb_batches, stop_div = False):
        self.phases, self.nb_batches, self.stop_div = phases, nb_batches, stop_div
        super().__init__(layer_opt, record_mom=True)

    def on_train_begin(self):
        super().on_train_begin()
        self.phase,self.best=0,1e9

    def on_batch_end(self, metrics):
        loss = metrics[0] if isinstance(metrics,list) else metrics
        if self.stop_div and (math.isnan(loss) or loss>self.best*4):
            return True
        if (loss<self.best and self.iteration>10): self.best=loss
        super().on_batch_end(metrics)
        self.phases[self.phase].update()
    
    def on_phase_begin(self):
        self.phases[self.phase].phase_begin(self.layer_opt, self.nb_batches[self.phase])

    def on_phase_end(self):
        self.phase += 1

    def plot_lr(self, show_text=True, show_moms=True):
        """Plots the lr rate/momentum schedule"""
        phase_limits = [0]
        for nb_batch, phase in zip(self.nb_batches, self.phases):
            phase_limits.append(phase_limits[-1] + nb_batch * phase.epochs)
        if not in_ipynb():
            plt.switch_backend('agg')
        np_plts = 2 if show_moms else 1
        fig, axs = plt.subplots(1,np_plts,figsize=(6*np_plts,4))
        if not show_moms: axs = [axs]
        for i in range(np_plts): axs[i].set_xlabel('iterations')
        axs[0].set_ylabel('learning rate')
        axs[0].plot(self.iterations,self.lrs)
        if show_moms:
            axs[1].set_ylabel('momentum')
            axs[1].plot(self.iterations,self.momentums)
        if show_text:   
            for i, phase in enumerate(self.phases):
                text = phase.opt_fn.__name__
                if phase.wds is not None: text+='\nwds='+str(phase.wds)
                if phase.beta is not None: text+='\nbeta='+str(phase.beta)
                for k in range(np_plts):
                    if i < len(self.phases)-1:
                        draw_line(axs[k], phase_limits[i+1])
                    draw_text(axs[k], (phase_limits[i]+phase_limits[i+1])/2, text) 
        if not in_ipynb():
            plt.savefig(os.path.join(self.save_path, 'lr_plot.png'))
    
    def plot(self, n_skip=10, n_skip_end=5, linear=None):
        if linear is None: linear = self.phases[-1].lr_decay == DecayType.LINEAR
        plt.ylabel("loss")
        plt.plot(self.lrs[n_skip:-n_skip_end], self.losses[n_skip:-n_skip_end])
        if linear: plt.xlabel("learning rate")
        else:
            plt.xlabel("learning rate (log scale)")
            plt.xscale('log')

def draw_line(ax,x):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.plot([x,x],[ymin,ymax], color='red', linestyle='dashed')

def draw_text(ax,x, text):
    xmin, xmax, ymin, ymax = ax.axis()
    ax.text(x,(ymin+ymax)/2,text, horizontalalignment='center', verticalalignment='center', fontsize=14, alpha=0.5)

def smooth_curve(vals, beta):
    avg_val = 0
    smoothed = []
    for (i,v) in enumerate(vals):
        avg_val = beta * avg_val + (1-beta) * v
        smoothed.append(avg_val/(1-beta**(i+1)))
    return smoothed
# end fastai.sgdr################################################################


# start fastai.layers#################################################################

# end fastai.layers################################################################


# start fastai.metrics#################################################################
# There are 2 versions of each metrics function, depending on the type of the prediction tensor:
# *    torch preds/log_preds
# *_np numpy preds/log_preds
#

def accuracy(preds, targs):
    preds = torch.max(preds, dim=1)[1]
    return (preds==targs).float().mean()

def accuracy_np(preds, targs):
    preds = np.argmax(preds, 1)
    return (preds==targs).mean()

def accuracy_thresh(thresh):
    return lambda preds,targs: accuracy_multi(preds, targs, thresh)

def accuracy_multi(preds, targs, thresh):
    return ((preds>thresh).float()==targs).float().mean()

def accuracy_multi_np(preds, targs, thresh):
    return ((preds>thresh)==targs).mean()

def recall(log_preds, targs, thresh=0.5, epsilon=1e-8):
    preds = torch.exp(log_preds)
    pred_pos = torch.max(preds > thresh, dim=1)[1]
    tpos = torch.mul((targs.byte() == pred_pos.byte()), targs.byte())
    return tpos.sum()/(targs.sum() + epsilon)

def recall_np(preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte()).float()
    return tpos.sum().float()/(targs.sum() + epsilon)

def precision(log_preds, targs, thresh=0.5, epsilon=1e-8):
    preds = torch.exp(log_preds)
    pred_pos = torch.max(preds > thresh, dim=1)[1]
    tpos = torch.mul((targs.byte() == pred_pos.byte()), targs.byte())
    return tpos.sum()/(pred_pos.sum() + epsilon)

def precision_np(preds, targs, thresh=0.5, epsilon=1e-8):
    pred_pos = preds > thresh
    tpos = torch.mul((targs.byte() == pred_pos), targs.byte())
    return tpos.sum().float()/(pred_pos.sum().float() + epsilon)

def fbeta(log_preds, targs, beta, thresh=0.5, epsilon=1e-8):
    """Calculates the F-beta score (the weighted harmonic mean of precision and recall).
    This is the micro averaged version where the true positives, false negatives and
    false positives are calculated globally (as opposed to on a per label basis).

    beta == 1 places equal weight on precision and recall, b < 1 emphasizes precision and
    beta > 1 favors recall.
    """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall(log_preds, targs, thresh)
    prec = precision(log_preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec + epsilon)

def fbeta_np(preds, targs, beta, thresh=0.5, epsilon=1e-8):
    """ see fbeta """
    assert beta > 0, 'beta needs to be greater than 0'
    beta2 = beta ** 2
    rec = recall_np(preds, targs, thresh)
    prec = precision_np(preds, targs, thresh)
    return (1 + beta2) * prec * rec / (beta2 * prec + rec + epsilon)

def f1(log_preds, targs, thresh=0.5): return fbeta(log_preds, targs, 1, thresh)
def f1_np(preds, targs, thresh=0.5): return fbeta_np(preds, targs, 1, thresh)
# end fastai.metrics################################################################


# start fastai.losses#################################################################
def fbeta_torch(y_true, y_pred, beta, threshold, eps=1e-9):
    y_pred = (y_pred.float() > threshold).float()
    y_true = y_true.float()
    tp = (y_pred * y_true).sum(dim=1)
    precision = tp / (y_pred.sum(dim=1)+eps)
    recall = tp / (y_true.sum(dim=1)+eps)
    return torch.mean(
        precision*recall / (precision*(beta**2)+recall+eps) * (1+beta**2))
# end fastai.losses################################################################


# start fastai.swa#################################################################
"""
    From the paper:
        Averaging Weights Leads to Wider Optima and Better Generalization
        Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
        https://arxiv.org/abs/1803.05407
        2018
        
    Author's implementation: https://github.com/timgaripov/swa
"""
class SWA(Callback):
    def __init__(self, model, swa_model, swa_start):
        super().__init__()
        self.model,self.swa_model,self.swa_start=model,swa_model,swa_start
        
    def on_train_begin(self):
        self.epoch = 0
        self.swa_n = 0

    def on_epoch_end(self, metrics):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1
            
        self.epoch += 1
            
    def update_average_model(self):
        # update running average of parameters
        model_params = self.model.parameters()
        swa_params = self.swa_model.parameters()
        for model_param, swa_param in zip(model_params, swa_params):
            swa_param.data *= self.swa_n
            swa_param.data += model_param.data
            swa_param.data /= (self.swa_n + 1)            
    
def collect_bn_modules(module, bn_modules):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        bn_modules.append(module)

def fix_batchnorm(swa_model, train_dl):
    """
    During training, batch norm layers keep track of a running mean and
    variance of the previous layer's activations. Because the parameters
    of the SWA model are computed as the average of other models' parameters,
    the SWA model never sees the training data itself, and therefore has no
    opportunity to compute the correct batch norm statistics. Before performing 
    inference with the SWA model, we perform a single pass over the training data
    to calculate an accurate running mean and variance for each batch norm layer.
    """
    bn_modules = []
    swa_model.apply(lambda module: collect_bn_modules(module, bn_modules))

    if not bn_modules: return

    swa_model.train()
    for module in bn_modules:
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)
    
    momenta = [m.momentum for m in bn_modules]
    inputs_seen = 0

    with torch.no_grad():
        for (*x,y) in iter(train_dl):        
            xs = V(x)
            batch_size = xs[0].size(1)

            momentum = batch_size / (inputs_seen + batch_size)
            for module in bn_modules:
                module.momentum = momentum

            if len(xs)==1: swa_model(*xs)
            else: swa_model(xs)

            
            inputs_seen += batch_size
                
    for module, momentum in zip(bn_modules, momenta):
        module.momentum = momentum
# end fastai.swa################################################################


# start fastai.model#################################################################
def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]

def predict_to_bcolz(m, gen, arr, workers=4):
    arr.trim(len(arr))
    lock=threading.Lock()
    m.eval()
    for x,*_ in tqdm(gen):
        y = to_np(m(VV(x)).data)
        with lock:
            arr.append(y)
            arr.flush()

def num_features(m):
    c=children(m)
    if len(c)==0: return None
    for l in reversed(c):
        if hasattr(l, 'num_features'): return l.num_features
        res = num_features(l)
        if res is not None: return res

def torch_item(x): return x.item() if hasattr(x,'item') else x[0]

class Stepper():
    def __init__(self, m, opt, crit, clip=0, reg_fn=None, fp16=False, loss_scale=1):
        self.m,self.opt,self.crit,self.clip,self.reg_fn = m,opt,crit,clip,reg_fn
        self.fp16 = fp16
        self.reset(True)
        if self.fp16: self.fp32_params = copy_model_to_fp32(m, opt)
        self.loss_scale = loss_scale

    def reset(self, train=True):
        if train: apply_leaf(self.m, set_train_mode)
        else: self.m.eval()
        if hasattr(self.m, 'reset'):
            self.m.reset()
            if self.fp16: self.fp32_params = copy_model_to_fp32(self.m, self.opt)

    def step(self, xs, y, epoch):
        xtra = []
        if len(xs)==1:
            output = self.m(*xs)
        else:
            output = self.m(xs)

        if isinstance(output,tuple): output,*xtra = output
        if self.fp16: self.m.zero_grad()
        else: self.opt.zero_grad() 
        loss = raw_loss = self.crit(output, y)
        if self.loss_scale != 1: assert(self.fp16); loss = loss*self.loss_scale
        if self.reg_fn: loss = self.reg_fn(output, xtra, raw_loss)
        loss.backward()
        if self.fp16: update_fp32_grads(self.fp32_params, self.m)
        if self.loss_scale != 1:
            for param in self.fp32_params: param.grad.data.div_(self.loss_scale)
        if self.clip:   # Gradient clipping
            if IS_TORCH_04: nn.utils.clip_grad_norm_(trainable_params_(self.m), self.clip)
            else: nn.utils.clip_grad_norm(trainable_params_(self.m), self.clip)
        if 'wd' in self.opt.param_groups[0] and self.opt.param_groups[0]['wd'] != 0: 
            #Weight decay out of the loss. After the gradient computation but before the step.
            for group in self.opt.param_groups:
                lr, wd = group['lr'], group['wd']
                for p in group['params']:
                    if p.grad is not None: p.data = p.data.add(-wd * lr, p.data)
        self.opt.step()
        if self.fp16: 
            copy_fp32_to_model(self.m, self.fp32_params)
            torch.cuda.synchronize()
        return torch_item(raw_loss.data)

    def evaluate(self, xs, y):
        if len(xs)==1:
            preds = self.m(*xs)
        else:
            preds = self.m(xs)
        if isinstance(preds,tuple): preds=preds[0]
        return preds, self.crit(preds, y)

def set_train_mode(m):
    if (hasattr(m, 'running_mean') and (getattr(m,'bn_freeze',False)
              or not getattr(m,'trainable',False))): m.eval()
    elif (getattr(m,'drop_freeze',False) and hasattr(m, 'p')
          and ('drop' in type(m).__name__.lower())): m.eval()
    else: m.train()

def fit(model, data, n_epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper,
        swa_model=None, swa_start=None, swa_eval_freq=None, visualize=False, **kwargs):
    """ Fits a model

    Arguments:
       model (model): any pytorch module
           net = to_gpu(net)
       data (ModelData): see ModelData class and subclasses (can be a list)
       opts: an optimizer. Example: optim.Adam. 
       If n_epochs is a list, it needs to be the layer_optimizer to get the optimizer as it changes.
       n_epochs(int or list): number of epochs (or list of number of epochs)
       crit: loss function to optimize. Example: F.cross_entropy
    """

    seq_first = kwargs.pop('seq_first', False)
    all_val = kwargs.pop('all_val', False)
    get_ep_vals = kwargs.pop('get_ep_vals', False)
    validate_skip = kwargs.pop('validate_skip', 0)
    metrics = metrics or []
    callbacks = callbacks or []
    avg_mom=0.98
    batch_num,avg_loss=0,0.
    for cb in callbacks: cb.on_train_begin()
    names = ["epoch", "trn_loss", "val_loss"] + [f.__name__ for f in metrics]
    if swa_model is not None:
        swa_names = ['swa_loss'] + [f'swa_{f.__name__}' for f in metrics]
        names += swa_names
        # will use this to call evaluate later
        swa_stepper = stepper(swa_model, None, crit, **kwargs)

    layout = "{!s:10} " * len(names)
    if not isinstance(n_epochs, Iterable): n_epochs=[n_epochs]
    if not isinstance(data, Iterable): data = [data]
    if len(data) == 1: data = data * len(n_epochs)
    for cb in callbacks: cb.on_phase_begin()
    model_stepper = stepper(model, opt.opt if hasattr(opt,'opt') else opt, crit, **kwargs)
    ep_vals = collections.OrderedDict()
    tot_epochs = int(np.ceil(np.array(n_epochs).sum()))
    cnt_phases = np.array([ep * len(dat.trn_dl) for (ep,dat) in zip(n_epochs,data)]).cumsum()
    phase = 0
    for epoch in range(tot_epochs):
        print(f'Epoch:{epoch+1} / {tot_epochs}')
        for cb in callbacks: cb.on_epoch_begin()
        if phase >= len(n_epochs): break #Sometimes cumulated errors make this append.
        model_stepper.reset(True)
        cur_data = data[phase]
        if hasattr(cur_data, 'trn_sampler'): cur_data.trn_sampler.set_epoch(epoch)
        if hasattr(cur_data, 'val_sampler'): cur_data.val_sampler.set_epoch(epoch)
        num_batch = len(cur_data.trn_dl)
        t = iter(cur_data.trn_dl)
        if all_val: val_iter = IterBatch(cur_data.val_dl)

        for index,(*x,y) in enumerate(t):
            batch_num += 1
            for cb in callbacks: cb.on_batch_begin()
            loss = model_stepper.step(V(x),V(y), epoch)
            avg_loss = avg_loss * avg_mom + loss * (1-avg_mom)
            debias_loss = avg_loss / (1 - avg_mom**batch_num)

            print(f"epoch {epoch} batch: {index+1} / {num_batch} loss={debias_loss}")

            # if index>3: break
            stop=False
            los = debias_loss if not all_val else [debias_loss] + validate_next(model_stepper,metrics, val_iter)
            for cb in callbacks: stop = stop or cb.on_batch_end(los)
            if stop: return
            if batch_num >= cnt_phases[phase]:
                for cb in callbacks: cb.on_phase_end()
                phase += 1
                if phase >= len(n_epochs):
                    t.close()
                    break
                for cb in callbacks: cb.on_phase_begin()
                if isinstance(opt, LayerOptimizer): model_stepper.opt = opt.opt
                if cur_data != data[phase]:
                    t.close()
                    break

        if not all_val:
            vals = validate(model_stepper, cur_data.val_dl, metrics, epoch, seq_first=seq_first, validate_skip = validate_skip)
            stop=False
            for cb in callbacks: stop = stop or cb.on_epoch_end(vals)
            if swa_model is not None:
                if (epoch + 1) >= swa_start and ((epoch + 1 - swa_start) % swa_eval_freq == 0 or epoch == tot_epochs - 1):
                    # fix_batchnorm(swa_model, cur_data.trn_dl)
                    swa_vals = validate(swa_stepper, cur_data.val_dl, metrics, epoch, validate_skip = validate_skip)
                    vals += swa_vals

            if epoch > 0: 
                print_stats(epoch, [debias_loss] + vals, visualize, prev_val)
            else:
                print(layout.format(*names))
                print_stats(epoch, [debias_loss] + vals, visualize)
            prev_val = [debias_loss] + vals
            ep_vals = append_stats(ep_vals, epoch, [debias_loss] + vals)
        if stop: break
    for cb in callbacks: cb.on_train_end()
    if get_ep_vals: return vals, ep_vals
    else: return vals

def append_stats(ep_vals, epoch, values, decimals=6):
    ep_vals[epoch]=list(np.round(values, decimals))
    return ep_vals

def print_stats(epoch, values, visualize, prev_val=[], decimals=6):
    layout = "{!s:^10}" + " {!s:10}" * len(values)
    values = [epoch] + list(np.round(values, decimals))
    sym = ""
    if visualize:
        if epoch == 0:                                             pass        
        elif values[1] > prev_val[0] and values[2] > prev_val[1]:  sym = " △ △"
        elif values[1] > prev_val[0] and values[2] < prev_val[1]:  sym = " △ ▼"            
        elif values[1] < prev_val[0] and values[2] > prev_val[1]:  sym = " ▼ △"            
        elif values[1] < prev_val[0] and values[2] < prev_val[1]:  sym = " ▼ ▼"
    print(layout.format(*values) + sym)

class IterBatch():
    def __init__(self, dl):
        self.idx = 0
        self.dl = dl
        self.iter = iter(dl)

    def __iter__(self): return self

    def next(self):
        res = next(self.iter)
        self.idx += 1
        if self.idx == len(self.dl):
            self.iter = iter(self.dl)
            self.idx=0
        return res

def validate_next(stepper, metrics, val_iter):
    """Computes the loss on the next minibatch of the validation set."""
    stepper.reset(False)
    with no_grad_context():
        (*x,y) = val_iter.next()
        preds,l = stepper.evaluate(VV(x), VV(y))
        res = [delistify(to_np(l))]
        res += [f(datafy(preds), datafy(y)) for f in metrics]
    stepper.reset(True)
    return res

def batch_sz(x, seq_first=False):
    if is_listy(x): x = x[0]
    return x.shape[1 if seq_first else 0]

def validate(stepper, dl, metrics, epoch, seq_first=False, validate_skip = 0):
    if epoch < validate_skip: return [float('nan')] + [float('nan')] * len(metrics)
    batch_cnts,loss,res = [],[],[]
    stepper.reset(False)
    with no_grad_context():
        for (*x,y) in iter(dl):
            y = VV(y)
            preds, l = stepper.evaluate(VV(x), y)
            batch_cnts.append(batch_sz(x, seq_first=seq_first))
            loss.append(to_np(l))
            res.append([f(datafy(preds), datafy(y)) for f in metrics])
    return [np.average(loss, 0, weights=batch_cnts)] + list(np.average(np.stack(res), 0, weights=batch_cnts))

def get_prediction(x):
    if is_listy(x): x=x[0]
    return x.data

def predict(m, dl):
    preda,_ = predict_with_targs_(m, dl)
    return np.concatenate(preda)

def predict_batch(m, x):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    return m(VV(x))

def predict_with_targs_(m, dl):
    m.eval()
    if hasattr(m, 'reset'): m.reset()
    res = []
    for *x,y in iter(dl): 
        if len(x)==1:
            res.append([get_prediction(to_np(m(*VV(x)))),to_np(y)])
        else:
            res.append([get_prediction(to_np(m(VV(x)))),to_np(y)])
    return zip(*res)

def predict_with_targs(m, dl):
    m.eval()
    preda,targa = predict_with_targs_(m, dl)
    return np.concatenate(preda), np.concatenate(targa)

# From https://github.com/ncullen93/torchsample
def model_summary(m, inputs):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if is_listy(output):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (not isinstance(module, nn.Sequential) and
           not isinstance(module, nn.ModuleList) and
           not (module == m)):
            hooks.append(module.register_forward_hook(hook))

    summary = OrderedDict()
    hooks = []
    m.apply(register_hook)
    xs = [to_gpu(Variable(x)) for x in inputs]
    m(*xs)

    for h in hooks: h.remove()
    return summary
# end fastai.model################################################################


# start fastai.fp16################################################################
IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')

class FP16(nn.Module):
    def __init__(self, module): 
        super().__init__()
        self.module = batchnorm_to_fp32(module.half())
        
    def forward(self, input):
        if is_float(input): input = input.half()
        return self.module(input)
    
    def load_state_dict(self, *inputs, **kwargs):
        self.module.load_state_dict(*inputs, **kwargs)

    def state_dict(self, *inputs, **kwargs):
        return self.module.state_dict(*inputs, **kwargs)
    
    def __getitem__(self, idx):
        return self.module[idx]

def is_float(tensor):
    if IS_TORCH_04: return tensor.is_floating_point()
    if isinstance(tensor, Variable): tensor = tensor.data
    return isinstance(tensor, torch.cuda.FloatTensor)

def batchnorm_to_fp32(module):
    '''
    BatchNorm layers to have parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_fp32(child)
    return module

def copy_model_to_fp32(m, optim):
    """  Creates a fp32 copy of model parameters and sets optimizer parameters
    """
    fp32_params = [m_param.clone().type(torch.cuda.FloatTensor).detach() for m_param in trainable_params_(m)]
    optim_groups = [group['params'] for group in optim.param_groups]
    iter_fp32_params = iter(fp32_params)
    for group_params in optim_groups:
        for i in range(len(group_params)):
            if not group_params[i].requires_grad: continue # only update trainable_params_
            fp32_param = next(iter_fp32_params)
            assert(fp32_param.shape == group_params[i].shape)
            fp32_param.requires_grad = group_params[i].requires_grad
            group_params[i] = fp32_param
    return fp32_params

def copy_fp32_to_model(m, fp32_params):
    m_params = trainable_params_(m)
    assert(len(m_params) == len(fp32_params))
    for fp32_param, m_param in zip(fp32_params, m_params):
        m_param.data.copy_(fp32_param.data)

def update_fp32_grads(fp32_params, m):
    m_params = trainable_params_(m)
    assert(len(m_params) == len(fp32_params))
    for fp32_param, m_param in zip(fp32_params, m_params):
        if fp32_param.grad is None:
            fp32_param.grad = nn.Parameter(fp32_param.data.new().resize_(*fp32_param.data.size()))
        fp32_param.grad.data.copy_(m_param.grad.data)

# end fastai.fp16###############################################################


# start fastai.learner###############################################################
def apply_lsuv_init(model, data, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True, cuda=True):
    model.eval();
    if cuda:
        model=model.cuda()
        data=data.cuda()
    else:
        model=model.cpu()
        data=data.cpu()        
        
    model.apply(count_conv_fc_layers)
    if do_orthonorm:
        model.apply(orthogonal_weights_init)
        if cuda:
            model=model.cuda()
    for layer_idx in range(gg['total_fc_conv_layers']):
        model.apply(add_current_hook)
        out = model(data)
        current_std = gg['act_dict'].std()
        attempts = 0
        while (np.abs(current_std - needed_std) > std_tol):
            gg['current_coef'] =  needed_std / (current_std  + 1e-8);
            gg['correction_needed'] = True
            model.apply(apply_weights_correction)
            if cuda:
                model=model.cuda()
            out = model(data)
            current_std = gg['act_dict'].std()
            attempts+=1
            if attempts > max_attempts:
                print(f'Cannot converge in {max_attempts} iterations')
                break
        if gg['hook'] is not None:
           gg['hook'].remove()
        gg['done_counter']+=1
        gg['counter_to_apply_correction'] = 0
        gg['hook_position'] = 0
        gg['hook']  = None
    if not cuda:
        model=model.cpu()
    return model



class Learner():
    def __init__(self, data, models, opt_fn=None, tmp_name='tmp', models_name='models', metrics=None, clip=None, crit=None):
        """
        Combines a ModelData object with a nn.Module object, such that you can train that
        module.
        data (ModelData): An instance of ModelData.
        models(module): chosen neural architecture for solving a supported problem.
        opt_fn(function): optimizer function, uses SGD with Momentum of .9 if none.
        tmp_name(str): output name of the directory containing temporary files from training process
        models_name(str): output name of the directory containing the trained model
        metrics(list): array of functions for evaluating a desired metric. Eg. accuracy.
        clip(float): gradient clip chosen to limit the change in the gradient to prevent exploding gradients Eg. .3
        """
        self.data_,self.models,self.metrics = data,models,metrics
        self.sched=None
        self.wd_sched = None
        self.clip = None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = tmp_name if os.path.isabs(tmp_name) else os.path.join(self.data.path, tmp_name)
        self.models_path = models_name if os.path.isabs(models_name) else os.path.join(self.data.path, models_name)
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        self.crit = crit if crit else self._get_crit(data)
        self.reg_fn = None
        self.fp16 = False
        self.swa_model = copy.deepcopy(self.model)

    @classmethod
    def from_model_data(cls, m, data, **kwargs):
        self = cls(data, BasicModel(to_gpu(m)), **kwargs)
        self.unfreeze()
        return self

    def __getitem__(self,i): return self.children[i]

    @property
    def children(self): return children(self.model)

    @property
    def model(self): return self.models.model

    @property
    def data(self): return self.data_

    def summary(self): return model_summary(self.model, [torch.rand(3, 3, self.data.sz,self.data.sz)])

    def __repr__(self): return self.model.__repr__()
    
    def lsuv_init(self, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=False):         
        x = V(next(iter(self.data.trn_dl))[0])
        self.models.model=apply_lsuv_init(self.model, x, needed_std=needed_std, std_tol=std_tol,
                            max_attempts=max_attempts, do_orthonorm=do_orthonorm, 
                            cuda=USE_GPU and torch.cuda.is_available())

    def set_bn_freeze(self, m, do_freeze):
        if hasattr(m, 'running_mean'): m.bn_freeze = do_freeze

    def bn_freeze(self, do_freeze):
        apply_leaf(self.model, lambda m: self.set_bn_freeze(m, do_freeze))

    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

    def freeze_all_but(self, n):
        c=self.get_layer_groups()
        for l in c: set_trainable(l, False)
        set_trainable(c[n], True)
        
    def freeze_groups(self, groups):
        c = self.get_layer_groups()
        self.unfreeze()
        for g in groups:
            set_trainable(c[g], False)
            
    def unfreeze_groups(self, groups):
        c = self.get_layer_groups()
        for g in groups:
            set_trainable(c[g], True)

    def unfreeze(self): self.freeze_to(0)

    def get_model_path(self, name): return os.path.join(self.models_path,name)+'.h5'
    
    def save(self, name): 
        save_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'): save_model(self.swa_model, self.get_model_path(name)[:-3]+'-swa.h5')
                       
    def load(self, name): 
        load_model(self.model, self.get_model_path(name))
        if hasattr(self, 'swa_model'): load_model(self.swa_model, self.get_model_path(name)[:-3]+'-swa.h5')

    def set_data(self, data): self.data_ = data

    def get_cycle_end(self, name):
        if name is None: return None
        return lambda sched, cycle: self.save_cycle(name, cycle)

    def save_cycle(self, name, cycle): self.save(f'{name}_cyc_{cycle}')
    def load_cycle(self, name, cycle): self.load(f'{name}_cyc_{cycle}')

    def half(self):
        if self.fp16: return
        self.fp16 = True
        if type(self.model) != FP16: self.models.model = FP16(self.model)
    def float(self):
        if not self.fp16: return
        self.fp16 = False
        if type(self.model) == FP16: self.models.model = self.model.module
        self.model.float()

    def fit_gen(self, model, data, layer_opt, n_cycle, cycle_len=None, cycle_mult=1, cycle_save_name=None, best_save_name=None,
                use_clr=None, use_clr_beta=None, metrics=None, callbacks=None, use_wd_sched=False, norm_wds=False,             
                wds_sched_mult=None, use_swa=True, swa_start=1, swa_eval_freq=1, **kwargs):

        """Method does some preparation before finally delegating to the 'fit' method for
        fitting the model. Namely, if cycle_len is defined, it adds a 'Cosine Annealing'
        scheduler for varying the learning rate across iterations.

        Method also computes the total number of epochs to fit based on provided 'cycle_len',
        'cycle_mult', and 'n_cycle' parameters.

        Args:
            model (Learner):  Any neural architecture for solving a supported problem.
                Eg. ResNet-34, RNN_Learner etc.

            data (ModelData): An instance of ModelData.

            layer_opt (LayerOptimizer): An instance of the LayerOptimizer class

            n_cycle (int): number of cycles

            cycle_len (int):  number of cycles before lr is reset to the initial value.
                E.g if cycle_len = 3, then the lr is varied between a maximum
                and minimum value over 3 epochs.

            cycle_mult (int): additional parameter for influencing how the lr resets over
                the cycles. For an intuitive explanation, please see
                https://github.com/fastai/fastai/blob/master/courses/dl1/lesson1.ipynb

            cycle_save_name (str): use to save the weights at end of each cycle (requires
                use_clr, use_clr_beta or cycle_len arg)

            best_save_name (str): use to save weights of best model during training.

            metrics (function): some function for evaluating a desired metric. Eg. accuracy.

            callbacks (list(Callback)): callbacks to apply during the training.

            use_wd_sched (bool, optional): set to True to enable weight regularization using
                the technique mentioned in https://arxiv.org/abs/1711.05101. When this is True
                alone (see below), the regularization is detached from gradient update and
                applied directly to the weights.

            norm_wds (bool, optional): when this is set to True along with use_wd_sched, the
                regularization factor is normalized with each training cycle.

            wds_sched_mult (function, optional): when this is provided along with use_wd_sched
                as True, the value computed by this function is multiplied with the regularization
                strength. This function is passed the WeightDecaySchedule object. And example
                function that can be passed is:
                            f = lambda x: np.array(x.layer_opt.lrs) / x.init_lrs
                            
            use_swa (bool, optional): when this is set to True, it will enable the use of
                Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407). The learner will
                include an additional model (in the swa_model attribute) for keeping track of the 
                average weights as described in the paper. All testing of this technique so far has
                been in image classification, so use in other contexts is not guaranteed to work.
                
            swa_start (int, optional): if use_swa is set to True, then this determines the epoch
                to start keeping track of the average weights. It is 1-indexed per the paper's
                conventions.
                
            swa_eval_freq (int, optional): if use_swa is set to True, this determines the frequency
                at which to evaluate the performance of the swa_model. This evaluation can be costly
                for models using BatchNorm (requiring a full pass through the data), which is why the
                default is not to evaluate after each epoch.

        Returns:
            None
        """

        if cycle_save_name:
            assert use_clr or use_clr_beta or cycle_len, "cycle_save_name argument requires either of the following arguments use_clr, use_clr_beta, cycle_len"

        if callbacks is None: callbacks=[]
        if metrics is None: metrics=self.metrics

        if use_wd_sched:
            # This needs to come before CosAnneal() because we need to read the initial learning rate from
            # layer_opt.lrs - but CosAnneal() alters the layer_opt.lrs value initially (divides by 100)
            if np.sum(layer_opt.wds) == 0:
                print('fit() warning: use_wd_sched is set to True, but weight decay(s) passed are 0. Use wds to '
                      'pass weight decay values.')
            batch_per_epoch = len(data.trn_dl)
            cl = cycle_len if cycle_len else 1
            self.wd_sched = WeightDecaySchedule(layer_opt, batch_per_epoch, cl, cycle_mult, n_cycle,
                                                norm_wds, wds_sched_mult)
            callbacks += [self.wd_sched]

        if use_clr is not None:
            clr_div,cut_div = use_clr[:2]
            moms = use_clr[2:] if len(use_clr) > 2 else None
            cycle_end = self.get_cycle_end(cycle_save_name)
            assert cycle_len, "use_clr requires cycle_len arg"
            total_iterators = len(data.trn_dl)*cycle_len
            self.sched = CircularLR(layer_opt, total_iterators, on_cycle_end=cycle_end, div=clr_div, cut_div=cut_div,
                                    momentums=moms)
        elif use_clr_beta is not None:
            div,pct = use_clr_beta[:2]
            moms = use_clr_beta[2:] if len(use_clr_beta) > 3 else None
            cycle_end = self.get_cycle_end(cycle_save_name)
            assert cycle_len, "use_clr_beta requires cycle_len arg"
            total_iterators = len(data.trn_dl)*cycle_len
            self.sched = CircularLR_beta(layer_opt, total_iterators, on_cycle_end=cycle_end, div=div,
                                    pct=pct, momentums=moms)
        elif cycle_len:
            cycle_end = self.get_cycle_end(cycle_save_name)
            cycle_batches = len(data.trn_dl)*cycle_len
            self.sched = CosAnneal(layer_opt, cycle_batches, on_cycle_end=cycle_end, cycle_mult=cycle_mult)
        elif not self.sched: self.sched=LossRecorder(layer_opt)
        callbacks+=[self.sched]

        if best_save_name is not None:
            callbacks+=[SaveBestModel(self, layer_opt, metrics, best_save_name)]

        n_epoch = int(sum_geom(cycle_len if cycle_len else 1, cycle_mult, n_cycle))

        if use_swa or not use_swa:
            # make a copy of the model to track average weights
            # self.swa_model = copy.deepcopy(model)
            # swa_start = 1 if n_epoch < 10 else n_epoch-9       # 保存最后10个的平均值
            callbacks+=[SWA(model, self.swa_model, swa_start)]

        return fit(model, data, n_epoch, layer_opt.opt, self.crit,
            metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, fp16=self.fp16,
            swa_model=self.swa_model if use_swa else None, swa_start=swa_start, 
            swa_eval_freq=swa_eval_freq, **kwargs)

    def get_layer_groups(self): return self.models.get_layer_groups()

    def get_layer_opt(self, lrs, wds):

        """Method returns an instance of the LayerOptimizer class, which
        allows for setting differential learning rates for different
        parts of the model.

        An example of how a model maybe differentiated into different parts
        for application of differential learning rates and weight decays is
        seen in ../.../courses/dl1/fastai/conv_learner.py, using the dict
        'model_meta'. Currently, this seems supported only for convolutional
        networks such as VGG-19, ResNet-XX etc.

        Args:
            lrs (float or list(float)): learning rate(s) for the model

            wds (float or list(float)): weight decay parameter(s).

        Returns:
            An instance of a LayerOptimizer
        """
        return LayerOptimizer(self.opt_fn, self.get_layer_groups(), lrs, wds)

    def fit(self, lrs, n_cycle, wds=None, **kwargs):

        """Method gets an instance of LayerOptimizer and delegates to self.fit_gen(..)

        Note that one can specify a list of learning rates which, when appropriately
        defined, will be applied to different segments of an architecture. This seems
        mostly relevant to ImageNet-trained models, where we want to alter the layers
        closest to the images by much smaller amounts.

        Likewise, a single or list of weight decay parameters can be specified, which
        if appropriate for a model, will apply variable weight decay parameters to
        different segments of the model.

        Args:
            lrs (float or list(float)): learning rate for the model

            n_cycle (int): number of cycles (or iterations) to fit the model for

            wds (float or list(float)): weight decay parameter(s).

            kwargs: other arguments

        Returns:
            None
        """
        self.sched = None
        layer_opt = self.get_layer_opt(lrs, wds)
        return self.fit_gen(self.model, self.data, layer_opt, n_cycle, **kwargs)

    def warm_up(self, lr, wds=None):
        layer_opt = self.get_layer_opt(lr/4, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), lr, linear=True)
        return self.fit_gen(self.model, self.data, layer_opt, 1)

    def lr_find(self, start_lr=1e-5, end_lr=10, wds=None, linear=False, **kwargs):
        """Helps you find an optimal learning rate for a model.

         It uses the technique developed in the 2015 paper
         `Cyclical Learning Rates for Training Neural Networks`, where
         we simply keep increasing the learning rate from a very small value,
         until the loss starts decreasing.

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            wds (iterable/float)

        Examples:
            As training moves us closer to the optimal weights for a model,
            the optimal learning rate will be smaller. We can take advantage of
            that knowledge and provide lr_find() with a starting learning rate
            1000x smaller than the model's current learning rate as such:

            >> learn.lr_find(lr/1000)

            >> lrs = np.array([ 1e-4, 1e-3, 1e-2 ])
            >> learn.lr_find(lrs / 1000)

        Notes:
            lr_find() may finish before going through each batch of examples if
            the loss decreases enough.

        .. _Cyclical Learning Rates for Training Neural Networks:
            http://arxiv.org/abs/1506.01186

        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder(layer_opt, len(self.data.trn_dl), end_lr, linear=linear)
        self.fit_gen(self.model, self.data, layer_opt, 1, **kwargs)
        self.load('tmp')

    def lr_find2(self, start_lr=1e-5, end_lr=10, num_it = 100, wds=None, linear=False, stop_dv=True, **kwargs):
        """A variant of lr_find() that helps find the best learning rate. It doesn't do
        an epoch but a fixed num of iterations (which may be more or less than an epoch
        depending on your data).
        At each step, it computes the validation loss and the metrics on the next
        batch of the validation data, so it's slower than lr_find().

        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            num_it : the number of iterations you want it to run
            wds (iterable/float)
            stop_dv : stops (or not) when the losses starts to explode.
        """
        self.save('tmp')
        layer_opt = self.get_layer_opt(start_lr, wds)
        self.sched = LR_Finder2(layer_opt, num_it, end_lr, linear=linear, metrics=self.metrics, stop_dv=stop_dv)
        self.fit_gen(self.model, self.data, layer_opt, num_it//len(self.data.trn_dl) + 1, all_val=True, **kwargs)
        self.load('tmp')

    def predict(self, is_test=False, use_swa=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        m = self.swa_model if use_swa else self.model
        return predict(m, dl)

    def predict_with_targs(self, is_test=False, use_swa=False):
        dl = self.data.test_dl if is_test else self.data.val_dl
        m = self.swa_model if use_swa else self.model
        return predict_with_targs(m, dl)

    def predict_dl(self, dl, use_swa=False): 
        m = self.swa_model if use_swa else self.model
        return predict_with_targs(m, dl)[0]

    def predict_array(self, arr):
        self.model.eval()
        return to_np(self.model(to_gpu(V(T(arr)))))

    def TTA(self, n_aug=4, is_test=False):
        """ Predict with Test Time Augmentation (TTA)

        Additional to the original test/validation images, apply image augmentation to them
        (just like for training images) and calculate the mean of predictions. The intent
        is to increase the accuracy of predictions by examining the images using multiple
        perspectives.

        Args:
            n_aug: a number of augmentation images to use per original image
            is_test: indicate to use test images; otherwise use validation images

        Returns:
            (tuple): a tuple containing:

                log predictions (numpy.ndarray): log predictions (i.e. `np.exp(log_preds)` will return probabilities)
                targs (numpy.ndarray): target values when `is_test==False`; zeros otherwise.
        """
        dl1 = self.data.test_dl     if is_test else self.data.val_dl
        dl2 = self.data.test_aug_dl if is_test else self.data.aug_dl
        preds1,targs = predict_with_targs(self.model, dl1)
        preds1 = [preds1]*math.ceil(n_aug/4)
        preds2 = [predict_with_targs(self.model, dl2)[0] for i in tqdm(range(n_aug), leave=False)]
        return np.stack(preds1+preds2), targs

    def fit_opt_sched(self, phases, cycle_save_name=None, best_save_name=None, stop_div=False, data_list=None, callbacks=None, 
                      cut = None, use_swa=False, swa_start=1, swa_eval_freq=5, **kwargs):
        """Wraps us the content of phases to send them to model.fit(..)

        This will split the training in several parts, each with their own learning rates/
        wds/momentums/optimizer detailed in phases.

        Additionaly we can add a list of different data objets in data_list to train
        on different datasets (to change the size for instance) for each of these groups.

        Args:
            phases: a list of TrainingPhase objects
            stop_div: when True, stops the training if the loss goes too high
            data_list: a list of different Data objects.
            kwargs: other arguments
            use_swa (bool, optional): when this is set to True, it will enable the use of
                Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407). The learner will
                include an additional model (in the swa_model attribute) for keeping track of the 
                average weights as described in the paper. All testing of this technique so far has
                been in image classification, so use in other contexts is not guaranteed to work. 
            swa_start (int, optional): if use_swa is set to True, then this determines the epoch
                to start keeping track of the average weights. It is 1-indexed per the paper's
                conventions.
            swa_eval_freq (int, optional): if use_swa is set to True, this determines the frequency
                at which to evaluate the performance of the swa_model. This evaluation can be costly
                for models using BatchNorm (requiring a full pass through the data), which is why the
                default is not to evaluate after each epoch.
        Returns:
            None
        """
        if data_list is None: data_list=[]
        if callbacks is None: callbacks=[]
        layer_opt = LayerOptimizer(phases[0].opt_fn, self.get_layer_groups(), 1e-2, phases[0].wds)
        if len(data_list) == 0: nb_batches = [len(self.data.trn_dl)] * len(phases)
        else: nb_batches = [len(data.trn_dl) for data in data_list] 
        self.sched = OptimScheduler(layer_opt, phases, nb_batches, stop_div)
        callbacks.append(self.sched)
        metrics = self.metrics
        if best_save_name is not None:
            callbacks+=[SaveBestModel(self, layer_opt, metrics, best_save_name)]
        if use_swa:
            # make a copy of the model to track average weights
            self.swa_model = copy.deepcopy(self.model)
            callbacks+=[SWA(self.model, self.swa_model, swa_start)]
        n_epochs = [phase.epochs for phase in phases] if cut is None else cut
        if len(data_list)==0: data_list = [self.data]
        return fit(self.model, data_list, n_epochs,layer_opt, self.crit,
            metrics=metrics, callbacks=callbacks, reg_fn=self.reg_fn, clip=self.clip, fp16=self.fp16,
            swa_model=self.swa_model if use_swa else None, swa_start=swa_start, 
            swa_eval_freq=swa_eval_freq, **kwargs)

    def _get_crit(self, data): return F.mse_loss

# end fastai.learner#################################################################

# start fastai.text###############################################################
class TextDataset(Dataset):
    def __init__(self, x, y, backwards=False, sos=None, eos=None):
        self.x,self.y,self.backwards,self.sos,self.eos = x,y,backwards,sos,eos

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.backwards: x = list(reversed(x))
        if self.eos is not None: x = x + [self.eos]
        if self.sos is not None: x = [self.sos]+x
        return np.array(x),self.y[idx]

    def __len__(self): return len(self.x)


class SortSampler(Sampler):
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range(len(self.data_source)), key=self.key, reverse=True))


class SortishSampler(Sampler):
    """Returns an iterator that traverses the the data in randomly ordered batches that are approximately the same size.
    The max key size batch is always returned in the first call because of pytorch cuda memory allocation sequencing.
    Without that max key returned first multiple buffers may be allocated when the first created isn't large enough
    to hold the next in the sequence.
    """
    def __init__(self, data_source, key, bs):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        sz = self.bs*50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:]))
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)


class LanguageModelLoader():
    """ Returns a language model iterator that iterates through batches that are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """
    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs,self.bptt,self.backwards = bs,bptt,backwards
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self): return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards: data=data[::-1]
        return T(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)


class LanguageModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [*zip(m.rnns, m.dropouths), (self.model[1], m.dropouti)]

class LanguageModelData():
    def __init__(self, path, pad_idx, n_tok, trn_dl, val_dl, test_dl=None, **kwargs):
        self.path,self.pad_idx,self.n_tok = path,pad_idx,n_tok
        self.trn_dl,self.val_dl,self.test_dl = trn_dl,val_dl,test_dl

    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        m = get_language_model(self.n_tok, emb_sz, n_hid, n_layers, self.pad_idx, **kwargs)
        model = LanguageModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)

class RNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.cross_entropy
    def fit(self, *args, **kwargs): return super().fit(*args, **kwargs, seq_first=True)

    def save_encoder(self, name): 
        save_model(self.model[0], self.get_model_path(name))
        if hasattr(self, "swa_model"): save_model(self.swa_model[0], self.get_model_path(name[:-4]+"-swa"+name[-4:]))
    def load_encoder(self, name): load_model(self.model[0], self.get_model_path(name))


class TextModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [(m.encoder, m.dropouti), *zip(m.rnns, m.dropouths), (self.model[1])]
# end fastai.text#################################################################


# start fastai.rnn_reg############################################################
def dropout_mask(x, sz, dropout):
    """ Applies a dropout mask whose size is determined by passed argument 'sz'.
    Args:
        x (nn.Variable): A torch Variable object
        sz (tuple(int, int, int)): The expected size of the new tensor
        dropout (float): The dropout fraction to apply

    This method uses the bernoulli distribution to decide which activations to keep.
    Additionally, the sampled activations is rescaled is using the factor 1/(1 - dropout).

    In the example given below, one can see that approximately .8 fraction of the
    returned tensors are zero. Rescaling with the factor 1/(1 - 0.8) returns a tensor
    with 5's in the unit places.

    The official link to the pytorch bernoulli function is here:
        http://pytorch.org/docs/master/torch.html#torch.bernoulli

    Examples:
        >>> a_Var = torch.autograd.Variable(torch.Tensor(2, 3, 4).uniform_(0, 1), requires_grad=False)
        >>> a_Var
            Variable containing:
            (0 ,.,.) =
              0.6890  0.5412  0.4303  0.8918
              0.3871  0.7944  0.0791  0.5979
              0.4575  0.7036  0.6186  0.7217
            (1 ,.,.) =
              0.8354  0.1690  0.1734  0.8099
              0.6002  0.2602  0.7907  0.4446
              0.5877  0.7464  0.4257  0.3386
            [torch.FloatTensor of size 2x3x4]
        >>> a_mask = dropout_mask(a_Var.data, (1,a_Var.size(1),a_Var.size(2)), dropout=0.8)
        >>> a_mask
            (0 ,.,.) =
              0  5  0  0
              0  0  0  5
              5  0  5  0
            [torch.FloatTensor of size 1x3x4]
    """
    return x.new(*sz).bernoulli_(1-dropout)/(1-dropout)


class LockedDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or not self.p: return x
        m = dropout_mask(x.data, (1, x.size(1), x.size(2)), self.p)
        return Variable(m, requires_grad=False) * x


class WeightDrop(torch.nn.Module):
    """A custom torch layer that serves as a wrapper on another torch layer.
    Primarily responsible for updating the weights in the wrapped module based
    on a specified dropout.
    """
    def __init__(self, module, dropout, weights=['weight_hh_l0']):
        """ Default constructor for the WeightDrop module

        Args:
            module (torch.nn.Module): A pytorch layer being wrapped
            dropout (float): a dropout value to apply
            weights (list(str)): the parameters of the wrapped **module**
                which should be fractionally dropped.
        """
        super().__init__()
        self.module,self.weights,self.dropout = module,weights,dropout
        self._setup()

    def _setup(self):
        """ for each string defined in self.weights, the corresponding
        attribute in the wrapped module is referenced, then deleted, and subsequently
        registered as a new parameter with a slightly modified name.

        Args:
            None

         Returns:
             None
        """
        if isinstance(self.module, torch.nn.RNNBase): self.module.flatten_parameters = noop
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))


    def _setweights(self):
        """ Uses pytorch's built-in dropout function to apply dropout to the parameters of
        the wrapped module.

        Args:
            None
        Returns:
            None
        """
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            if hasattr(self.module, name_w):
                delattr(self.module, name_w)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        """ updates weights and delegates the propagation of the tensor to the wrapped module's
        forward method

        Args:
            *args: supplied arguments

        Returns:
            tensor obtained by running the forward method on the wrapped module.
        """
        self._setweights()
        return self.module.forward(*args)

class EmbeddingDropout(nn.Module):

    """ Applies dropout in the embedding layer by zeroing out some elements of the embedding vector.
    Uses the dropout_mask custom layer to achieve this.

    Args:
        embed (torch.nn.Embedding): An embedding torch layer
        words (torch.nn.Variable): A torch variable
        dropout (float): dropout fraction to apply to the embedding weights
        scale (float): additional scaling to apply to the modified embedding weights

    Returns:
        tensor of size: (batch_size x seq_length x embedding_size)

    Example:

    >> embed = torch.nn.Embedding(10,3)
    >> words = Variable(torch.LongTensor([[1,2,4,5] ,[4,3,2,9]]))
    >> words.size()
        (2,4)
    >> embed_dropout_layer = EmbeddingDropout(embed)
    >> dropout_out_ = embed_dropout_layer(embed, words, dropout=0.40)
    >> dropout_out_
        Variable containing:
        (0 ,.,.) =
          1.2549  1.8230  1.9367
          0.0000 -0.0000  0.0000
          2.2540 -0.1299  1.5448
          0.0000 -0.0000 -0.0000

        (1 ,.,.) =
          2.2540 -0.1299  1.5448
         -4.0457  2.4815 -0.2897
          0.0000 -0.0000  0.0000
          1.8796 -0.4022  3.8773
        [torch.FloatTensor of size 2x4x3]
    """

    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, words, dropout=0.1, scale=None):
        if dropout:
            size = (self.embed.weight.size(0),1)
            mask = Variable(dropout_mask(self.embed.weight.data, size, dropout))
            masked_embed_weight = mask * self.embed.weight
        else: masked_embed_weight = self.embed.weight

        if scale: masked_embed_weight = scale * masked_embed_weight

        padding_idx = self.embed.padding_idx
        if padding_idx is None: padding_idx = -1

        
        if IS_TORCH_04:
            X = F.embedding(words,
                masked_embed_weight, padding_idx, self.embed.max_norm,
                self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse)
        else:
            X = self.embed._backend.Embedding.apply(words,
                masked_embed_weight, padding_idx, self.embed.max_norm,
                self.embed.norm_type, self.embed.scale_grad_by_freq, self.embed.sparse)

        return X
# end fastai.rnn_reg############################################################


# start fastai.lm_rnn############################################################
def seq2seq_reg(output, xtra, loss, alpha=0, beta=0):
    hs,dropped_hs = xtra
    if alpha:  # Activation Regularization
        loss = loss + (alpha * dropped_hs[-1].pow(2).mean()).sum()
    if beta:   # Temporal Activation Regularization (slowness)
        h = hs[-1]
        if len(h)>1: loss = loss + (beta * (h[1:] - h[:-1]).pow(2).mean()).sum()
    return loss


def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    if IS_TORCH_04: return h.detach() if type(h) == torch.Tensor else tuple(repackage_var(v) for v in h)
    else: return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)


class RNN_Encoder(nn.Module):

    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM or QRNN layers to drive the network, and
        - variational dropouts in the embedding and LSTM/QRNN layers

        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange=0.1

    def __init__(self, ntoken, emb_sz, n_hid, n_layers, pad_token, bidir=False,
                 dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5, qrnn=False):
        """ Default constructor for the RNN_Encoder class

            Args:
                bs (int): batch size of input data
                ntoken (int): number of vocabulary (or tokens) in the source dataset
                emb_sz (int): the embedding size to use to encode each token
                n_hid (int): number of hidden activation per LSTM layer
                n_layers (int): number of LSTM layers to use in the architecture
                pad_token (int): the int value used for padding text.
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                dropoute (float): dropout to apply to the embedding layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.

            Returns:
                None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs, self.qrnn = 1, qrnn
        self.encoder = nn.Embedding(ntoken, emb_sz, padding_idx=pad_token)
        self.encoder_with_dropout = EmbeddingDropout(self.encoder)
        if self.qrnn:
            #Using QRNN requires cupy: https://github.com/cupy/cupy
            from .torchqrnn.qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(n_layers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.linear = WeightDrop(rnn.linear, wdrop, weights=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                1, bidirectional=bidir) for l in range(n_layers)]
            if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)

        self.emb_sz,self.n_hid,self.n_layers,self.dropoute = emb_sz,n_hid,n_layers,dropoute
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(n_layers)])

    def forward(self, input):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)

        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl,bs = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        with set_grad_enabled(self.training):
            emb = self.encoder_with_dropout(input, dropout=self.dropoute if self.training else 0)
            emb = self.dropouti(emb)
            raw_output = emb
            new_hidden,raw_outputs,outputs = [],[],[]
            for l, (rnn,drop) in enumerate(zip(self.rnns, self.dropouths)):
                current_input = raw_output
                raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.n_layers - 1: raw_output = drop(raw_output)
                outputs.append(raw_output)

            self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        if IS_TORCH_04: return Variable(self.weights.new(self.ndir, self.bs, nh).zero_())
        else: return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset(self):
        if self.qrnn: [r.reset() for r in self.rnns]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self.one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.n_layers)]


class MultiBatchRNN(RNN_Encoder):
    def __init__(self, bptt, max_seq, *args, **kwargs):
        self.max_seq,self.bptt = max_seq,bptt
        super().__init__(*args, **kwargs)

    def concat(self, arrs):
        return [torch.cat([l[si] for l in arrs]) for si in range(len(arrs[0]))]

    def forward(self, input):
        sl,bs = input.size()
        for l in self.hidden:
            for h in l: h.data.zero_()
        raw_outputs, outputs = [],[]
        for i in range(0, sl, self.bptt):
            r, o = super().forward(input[i: min(i+self.bptt, sl)])
            if i>(sl-self.max_seq):
                raw_outputs.append(r)
                outputs.append(o)
        return self.concat(raw_outputs), self.concat(outputs)

class LinearDecoder(nn.Module):
    initrange=0.1
    def __init__(self, n_out, n_hid, dropout, tie_encoder=None, bias=False):
        super().__init__()
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.dropout = LockedDropout(dropout)
        if bias: self.decoder.bias.data.zero_()
        if tie_encoder: self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        raw_outputs, outputs = input
        output = self.dropout(outputs[-1])
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        result = decoded.view(-1, decoded.size(1))
        return result, raw_outputs, outputs


class LinearBlock(nn.Module):
    def __init__(self, ni, nf, drop):
        super().__init__()
        self.lin = nn.Linear(ni, nf)
        self.drop = nn.Dropout(drop)
        self.bn = nn.BatchNorm1d(ni)

    def forward(self, x): return self.lin(self.drop(self.bn(x)))


class PoolingLinearClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, input):
        raw_outputs, outputs = input
        output = outputs[-1]
        sl,bs,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        x = torch.cat([output[-1], mxpool, avgpool], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x, raw_outputs, outputs


class SequentialRNN(nn.Sequential):
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()


def get_language_model(n_tok, emb_sz, n_hid, n_layers, pad_token,
                 dropout=0.4, dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, tie_weights=True, qrnn=False, bias=False):
    """Returns a SequentialRNN model.

    A RNN_Encoder layer is instantiated using the parameters provided.

    This is followed by the creation of a LinearDecoder layer.

    Also by default (i.e. tie_weights = True), the embedding matrix used in the RNN_Encoder
    is used to  instantiate the weights for the LinearDecoder layer.

    The SequentialRNN layer is the native torch's Sequential wrapper that puts the RNN_Encoder and
    LinearDecoder layers sequentially in the model.

    Args:
        n_tok (int): number of unique vocabulary words (or tokens) in the source dataset
        emb_sz (int): the embedding size to use to encode each token
        n_hid (int): number of hidden activation per LSTM layer
        n_layers (int): number of LSTM layers to use in the architecture
        pad_token (int): the int value used for padding text.
        dropouth (float): dropout to apply to the activations going from one LSTM layer to another
        dropouti (float): dropout to apply to the input layer.
        dropoute (float): dropout to apply to the embedding layer.
        wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
        tie_weights (bool): decide if the weights of the embedding matrix in the RNN encoder should be tied to the
            weights of the LinearDecoder layer.
        qrnn (bool): decide if the model is composed of LSTMS (False) or QRNNs (True).
        bias (bool): decide if the decoder should have a bias layer or not.
    Returns:
        A SequentialRNN model
    """
    rnn_enc = RNN_Encoder(n_tok, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,
                 dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)
    enc = rnn_enc.encoder if tie_weights else None
    return SequentialRNN(rnn_enc, LinearDecoder(n_tok, emb_sz, dropout, tie_encoder=enc, bias=bias))


def get_rnn_classifier(bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                      dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, qrnn=False):
    rnn_enc = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)
    return SequentialRNN(rnn_enc, PoolingLinearClassifier(layers, drops))
# end fastai.lm_rnn#############################################################


# start sampled_sm ###############################################################
def resample_vocab(itos, trn, val, sz):
    freqs = Counter(trn)
    itos2 = [o for o,p in freqs.most_common()][:sz]
    itos2.insert(0,1)
    itos2.insert(0,0)
    stoi2 = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos2)})

    trn = np.array([stoi2[o] for o in trn])
    val = np.array([stoi2[o] for o in val])

    itos3 = [itos[o] for o in itos2]
    stoi3 = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos3)})
    return trn,val,itos3,stoi3


def get_prs(c, nt):
    uni_counter = Counter(c)
    uni_counts = np.array([uni_counter[o] for o in range(nt)])
    return uni_counts/uni_counts.sum()


def pt_sample(pr, ns):
    w = -torch.log(cuda.FloatTensor(len(pr)).uniform_())/(pr+1e-10)
    return torch.topk(w, ns, largest=False)[1]


class CrossEntDecoder(nn.Module):
    initrange=0.1
    def __init__(self, prs, decoder, n_neg=4000, sampled=True):
        super().__init__()
        self.prs,self.decoder,self.sampled = T(prs).cuda(),decoder,sampled
        self.set_n_neg(n_neg)

    def set_n_neg(self, n_neg): self.n_neg = n_neg

    def get_rand_idxs(self): return pt_sample(self.prs, self.n_neg)

    def sampled_softmax(self, input, target):
        idxs = V(self.get_rand_idxs())
        dw = self.decoder.weight
        #db = self.decoder.bias
        output = input @ dw[idxs].t() #+ db[idxs]
        max_output = output.max()
        output = output - max_output
        num = (dw[target] * input).sum(1) - max_output
        negs = torch.exp(num) + (torch.exp(output)*2).sum(1)
        return (torch.log(negs) - num).mean()

    def forward(self, input, target):
        if self.decoder.training:
            if self.sampled: return self.sampled_softmax(input, target)
            else: input = self.decoder(input)
        return F.cross_entropy(input, target)

def get_learner(drops, n_neg, sampled, md, em_sz, nh, nl, opt_fn, prs):

    class _LinearDecoder(nn.Module):
        initrange=0.1
        def __init__(self, n_out, nhid, dropout, tie_encoder=None, decode_train=True):
            super().__init__()
            self.decode_train = decode_train
            self.decoder = nn.Linear(nhid, n_out, bias=False)
            self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
            self.dropout = LockedDropout(dropout)
            if tie_encoder: self.decoder.weight = tie_encoder.weight

        def forward(self, input):
            raw_outputs, outputs = input
            output = self.dropout(outputs[-1])
            output = output.view(output.size(0)*output.size(1), output.size(2))
            if self.decode_train or not self.training:
                decoded = self.decoder(output)
                output = decoded.view(-1, decoded.size(1))
            return output, raw_outputs, outputs

    def _get_language_model(n_tok, em_sz, nhid, nlayers, pad_token, decode_train=True, dropouts=None):
        if dropouts is None: dropouts = [0.5,0.4,0.5,0.05,0.3]
        rnn_enc = RNN_Encoder(n_tok, em_sz, n_hid=nhid, n_layers=nlayers, pad_token=pad_token,
                     dropouti=dropouts[0], wdrop=dropouts[2], dropoute=dropouts[3], dropouth=dropouts[4])
        rnn_dec = _LinearDecoder(n_tok, em_sz, dropouts[1], decode_train=decode_train, tie_encoder=rnn_enc.encoder)
        return SequentialRNN(rnn_enc, rnn_dec)

    m = to_gpu(_get_language_model(md.n_tok, em_sz, nh, nl, md.pad_idx, decode_train=False, dropouts=drops))
    crit = CrossEntDecoder(prs, m[1].decoder, n_neg=n_neg, sampled=sampled).cuda()
    learner = RNN_Learner(md, LanguageModel(m), opt_fn=opt_fn)
    crit.dw = learner.model[0].encoder.weight
    learner.crit = crit
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip=0.3
    return learner,crit

# end sampled_sm  #################################################################




























































Punctuation = set("。？！，、；：「」『』‘’“”（〔〕【】—…–．《》〈〉|#")
re_num = re.compile(r"\d+")
re_star = re.compile(r"\*{3}")
num = "X"
def split_sentence(sentence,dtype):
    if dtype == "char":
        return [char for char in sentence if char and (0x4E00<= ord(char) <= 0x9FA5 or char in Punctuation or char==num) ]
    elif dtype == "word":
        return [word for word in list(jieba.cut(sentence)) if word and (0x4E00<= ord(word[0]) <= 0x9FA5 or word[0] in Punctuation or word==num)]

class WikiCorpus(object):
    def __init__(self):pass
    def __iter__(self):
        with open(model_dir + "wiki.csv",'r',encoding="utf8") as wiki:
            for line in wiki:
                line = re_num.sub(num,line)
                yield line

class AtecCorpus(object):
    def __init__(self):pass
    def __iter__(self):
        with open(model_dir + "atec_nlp_sim_train.csv",'r',encoding="utf8") as atec:
            for line in atec:
                lineno, s1, s2, label=line.strip().split("\t")
                s1 = re_star.sub(num,s1)
                s2 = re_star.sub(num,s2)
                yield s1, s2

def create_lm_csv(corpus, articles = 5000):
    train_array, val_array = [],[]
    if corpus=="wiki":data = WikiCorpus()
    elif corpus=="atec":data = AtecCorpus()

    os.makedirs(model_dir + corpus, exist_ok=True)
    len_trn, len_val = 0, 0
    train_file = open(model_dir + corpus + "/lm_train.csv", 'w', encoding="utf8")
    valid_file = open(model_dir + corpus + "/lm_valid.csv", 'w', encoding="utf8")
    if corpus=="wiki":
        np.random.seed(random_state)
        random_choice = np.random.rand(articles)
        for index, line in enumerate(data):
            if index>=articles: break
            if random_choice[index]<0.01:
                valid_file.write(line)
                len_val += 1
            else:
                train_file.write(line)
                len_trn += 1
    elif corpus=="atec":
        data = [[s1,s2] for s1,s2 in data]
        train, val = train_test_split(data, test_size=test_size, random_state=random_state)
        for s1,s2 in train:
            train_file.write(s1+"\n")
            train_file.write(s2+"\n")
            len_trn += 2
        for s1,s2 in val:
            valid_file.write(s1+"\n")
            valid_file.write(s2+"\n")
            len_val += 2

    train_file.flush()
    train_file.close()
    valid_file.flush()
    valid_file.close()
    print("create_lm_csv", len_trn, len_val)

BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag

def process_sentence(s, dtype, backwards):
    tokens = split_sentence(s,dtype)
    if backwards:tokens = list(reversed(tokens))
    return ['\n', BOS, FLD, '1'] + tokens


def create_toks(dir_path, dtype='char', backwards=False):
    print(f'=== create_toks: dir_path {dir_path} backwards {backwards} dtype {dtype}')

    assert os.path.exists(dir_path), f'Error: {dir_path} does not exist.'

    tmp_path = dir_path + "/" + 'tmp'
    os.makedirs(tmp_path, exist_ok=True)
    get_all = lambda filename: [process_sentence(line,dtype,backwards) for line in open(filename,'r',encoding="utf8")]

    tok_trn = get_all(dir_path + "/" + f"lm_train.csv")
    tok_val = get_all(dir_path + "/" + f"lm_valid.csv")

    BWD = '_bwd' if backwards else ""
    np.save(tmp_path + "/" + f'tok_trn_{dtype}{BWD}.npy', tok_trn)
    np.save(tmp_path + "/" + f'tok_val_{dtype}{BWD}.npy', tok_val)
    print("create_toks", len(tok_trn), len(tok_val))


def tok2id(dir_path, max_vocab, min_freq=1, dtype="char", backwards=False):
    print(f'=== tok2id: dir_path {dir_path} max_vocab {max_vocab} min_freq {min_freq}')
    p = dir_path
    assert os.path.exists(p), f'Error: {p} does not exist.'
    tmp_path = p + "/" + 'tmp'
    assert os.path.exists(tmp_path), f'Error: {tmp_path} does not exist.'
    BWD = '_bwd' if backwards else ""

    trn_tok = np.load(tmp_path + "/" + f'tok_trn_{dtype}{BWD}.npy')
    val_tok = np.load(tmp_path + "/" + f'tok_val_{dtype}{BWD}.npy')

    freq = Counter(p for o in trn_tok for p in o)
    print(freq.most_common(25))
    itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    print("vocab len",len(itos))

    trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
    val_lm = np.array([[stoi[o] for o in p] for p in val_tok])

    np.save(tmp_path + "/" + f'trn_ids_{dtype}{BWD}.npy', trn_lm)
    np.save(tmp_path + "/" + f'val_ids_{dtype}{BWD}.npy', val_lm)
    # 前向和后向的词典相同
    pickle.dump(itos, open(tmp_path + "/" + f'itos_{dtype}.pkl', 'wb'))
    print("tok2id", len(trn_lm), len(val_lm))

def sent_token2id(sents, dir_path, dtype, backwards):
    sents = [process_sentence(s,dtype,backwards) for s in sents]
    tmp_path = dir_path + "/" + 'tmp'
    itos = pickle.load(open(tmp_path + "/" + f'itos_{dtype}.pkl', 'rb'))
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
    return np.array([[stoi[o] for o in p] for p in sents])

def create_ids_lbls():
    with open(model_dir + "atec_nlp_sim_train.csv",'r',encoding="utf8") as atec:
        labels = []
        for line in atec:
            lineno, s1, s2, label=line.strip().split("\t")
            labels.append(int(label))
    train, valid = train_test_split(labels, test_size=test_size, random_state=random_state)
    np.save(model_dir + f"atec/tmp/lbl_trn.npy", train)
    np.save(model_dir + f"atec/tmp/lbl_val.npy", valid)
    print("create_ids_lbls", len(train), len(valid))


class EarlyStopping(Callback):
    def __init__(self, learner, save_path, enc_path=None, patience=5):
        super().__init__()
        self.learner=learner
        self.save_path=save_path
        self.enc_path=enc_path
        self.patience=patience
    def on_train_begin(self):
        self.best_val_loss=100
        self.num_epochs_no_improvement=0
    def on_epoch_end(self, metrics):
        val_loss = metrics[0]
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.num_epochs_no_improvement = 0
            self.learner.save(self.save_path)
            if self.enc_path is not None:
                self.learner.save_encoder(self.enc_path)
        else:
            self.num_epochs_no_improvement += 1
        if self.num_epochs_no_improvement > self.patience:
            print(f'Stopping - no improvement after {self.patience+1} epochs')
            return True
    def on_train_end(self):
        print(f'Loading best model from {self.save_path}')
        self.learner.load(self.save_path)


class TimerStop(Callback):
    """docstring for TimerStop"""
    def __init__(self, start_time, total_seconds):
        super(TimerStop, self).__init__()
        self.start_time = start_time
        self.total_seconds = total_seconds
        self.epoch_seconds = []
        self.is_stoped = False

    def on_epoch_begin(self):
        self.epoch_start = time.time()

    def on_epoch_end(self, metrics):
        self.epoch_seconds.append(time.time() - self.epoch_start)

        mean_epoch_seconds = sum(self.epoch_seconds)/len(self.epoch_seconds)
        if time.time() + mean_epoch_seconds > self.start_time + self.total_seconds:
            self.is_stoped = True
            return True

    def on_train_end(self):
        if self.is_stoped:
            print('timer stopping')

def pretrain_lm(dir_path, cuda_id, cl=1, bs=64, backwards=False, lr=3e-4, sampled=True, early_stopping=True,
             preload = False, pretrain_id='',dtype="char"):
    print(f'=== pretrain_lm: dir_path {dir_path}; cuda_id {cuda_id}; cl {cl}; bs {bs}; '
          f'backwards {backwards}; lr {lr}; sampled {sampled}; early stopping {early_stopping}; '
          f'preload {preload}; pretrain_id {pretrain_id}; dtype {dtype}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)
    PRE = 'bwd_' if backwards else 'fwd_'
    BWD = '_bwd' if backwards else ""
    p = dir_path
    assert os.path.exists(p), f'Error: {p} does not exist.'
    bptt=70
    em_sz,nh,nl = 400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    trn_lm = np.load(p + "/" + f'tmp/trn_ids_{dtype}{BWD}.npy')
    val_lm = np.load(p + "/" + f'tmp/val_ids_{dtype}{BWD}.npy')
    trn_lm = np.concatenate(trn_lm)
    val_lm = np.concatenate(val_lm)

    itos = pickle.load(open(p + "/" + f'tmp/itos_{dtype}.pkl', 'rb'))
    vs = len(itos)

    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs//5 if sampled else bs, bptt)
    md = LanguageModelData(p, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    tprs = get_prs(trn_lm, vs)
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.5
    n_neg = 1000 if dtype == "char" else 15000
    learner,crit = get_learner(drops, n_neg, sampled, md, em_sz, nh, nl, opt_fn, tprs)
    wd=1e-7
    learner.metrics = [accuracy]

    lrs = np.array([lr/6,lr/3,lr,lr])
    #lrs=lr

    lm_path = f'{PRE}{pretrain_id}_{dtype}'
    enc_path = f'{PRE}{pretrain_id}_{dtype}_enc'
    callbacks = []
    if False and early_stopping:
        callbacks.append(EarlyStopping(learner, lm_path, enc_path, patience=5))
        print('Using early stopping...')
    if preload: learner.load(lm_path)
    learner.fit(lrs, 1, wds=wd, use_clr=(32,10), cycle_len=cl, callbacks=callbacks)
    learner.save(lm_path)
    learner.save_encoder(enc_path)

def finetune_lm(dir_path, pretrain_path, cuda_id=0, cl=25, pretrain_id='', lm_id='', bs=64,
             dropmult=1.0, backwards=False, lr=4e-3, preload=True, bpe=False, startat=0, 
             use_clr=True, use_regular_schedule=False, use_discriminative=True, notrain=False, joined=False,
             train_file_id='', early_stopping=True, dtype="char"):
    print(f'=== finetune_lm: dir_path {dir_path}; pretrain_path {pretrain_path}; cuda_id {cuda_id}; '
          f'pretrain_id {pretrain_id}; cl {cl}; bs {bs}; backwards {backwards}; '
          f'dropmult {dropmult}; lr {lr}; preload {preload}; bpe {bpe};'
          f'startat {startat}; use_clr {use_clr}; notrain {notrain}; joined {joined} '
          f'early stopping {early_stopping}; dtype {dtype}')

    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE  = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    BWD = '_bwd' if backwards else ""
    train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    lm_path=f'{PRE}{lm_id}lm_{dtype}'
    enc_path=f'{PRE}{lm_id}lm_{dtype}_enc'

    pre_lm_path = pretrain_path + "/" + 'models' + "/" + f'{PRE}{pretrain_id}_{dtype}.h5'
    for p in [dir_path, pretrain_path, pre_lm_path]:
        assert os.path.exists(p), f'Error: {p} does not exist.'

    bptt=70
    em_sz,nh,nl = 400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    trn_lm_path = dir_path + "/" + 'tmp' + "/" + f'trn_{IDS}{train_file_id}_{dtype}{BWD}.npy'
    val_lm_path = dir_path + "/" + 'tmp' + "/" + f'val_{IDS}_{dtype}{BWD}.npy'

    print(f'Loading {trn_lm_path} and {val_lm_path}')
    trn_lm = np.load(trn_lm_path)
    trn_lm = np.concatenate(trn_lm)
    val_lm = np.load(val_lm_path)
    val_lm = np.concatenate(val_lm)

    if bpe:
        vs=30002
    else:
        itos = pickle.load(open(dir_path + "/" + 'tmp' + "/" + f'itos_{dtype}.pkl', 'rb'))
        vs = len(itos)

    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs, bptt)
    md = LanguageModelData(dir_path, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*dropmult

    learner = md.get_model(opt_fn, em_sz, nh, nl,
        dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip=0.3
    learner.metrics = [accuracy]
    wd=1e-7

    lrs = np.array([lr/6,lr/3,lr,lr/2]) if use_discriminative else lr
    if preload and startat == 0:
        wgts = torch.load(pre_lm_path, map_location=lambda storage, loc: storage)
        if bpe:
            learner.model.load_state_dict(wgts)
        else:
            print(f'Loading pretrained weights...')
            ew = to_np(wgts['0.encoder.weight'])
            row_m = ew.mean(0)

            itos2 = pickle.load(open(pretrain_path + "/" + 'tmp' + "/" + f'itos_{dtype}.pkl', 'rb'))
            stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
            nw = np.zeros((vs, em_sz), dtype=np.float32)
            nb = np.zeros((vs,), dtype=np.float32)
            for i,w in enumerate(itos):
                r = stoi2[w]
                if r>=0:
                    nw[i] = ew[r]
                else:
                    nw[i] = row_m

            wgts['0.encoder.weight'] = T(nw)
            wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
            wgts['1.decoder.weight'] = T(np.copy(nw))
            learner.model.load_state_dict(wgts)
            #learner.freeze_to(-1)
            #learner.fit(lrs, 1, wds=wd, use_clr=(6,4), cycle_len=1)
    elif preload:
        print('Loading LM that was already fine-tuned on the target data...')
        learner.load(lm_path)

    if not notrain:
        learner.unfreeze()
        if use_regular_schedule:
            print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
            use_clr = None
            n_cycles=cl
            cl=None
        else:
            n_cycles=1
        callbacks = []
        if False and early_stopping:
            callbacks.append(EarlyStopping(learner, lm_path, enc_path, patience=5))
            print('Using early stopping...')
        learner.fit(lrs, n_cycles, wds=wd, use_clr=(32,10) if use_clr else None, cycle_len=cl,
                    callbacks=callbacks)
        learner.save(lm_path)
        learner.save_encoder(enc_path)
    else:
        print('No more fine-tuning used. Saving original LM...')
        learner.save(lm_path)
        learner.save_encoder(enc_path)

def freeze_all_but(learner, n):
    c=learner.get_layer_groups()
    for l in c: set_trainable(l, False)
    set_trainable(c[n], True)

def repackage_var(h):
    """Wraps h in new Variables, to detach them from their history."""
    if IS_TORCH_04: return h.detach() if type(h) == torch.Tensor else tuple(repackage_var(v) for v in h)
    else: return Variable(h.data) if type(h) == Variable else tuple(repackage_var(v) for v in h)






class Siamese_Baseline(nn.Module):
    def __init__(self, cfg):
        super(Siamese_Baseline,self).__init__()

        self.ndir = 1
        self.n_h = 100
        self.bs, self.qrnn = 1, False
        self.shared_lstm = nn.LSTM(400,self.n_h)

    def one_hidden(self):   # 与weights变量在同一个cuda设备上，且数据类型相同
        return Variable(self.weights.new(self.ndir, self.bs, self.n_h).zero_())

    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = (self.one_hidden(), self.one_hidden())    # 初始化值 h0 和 c0

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward_once(self, x):
        sl,bs,_ = x.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        with set_grad_enabled(self.training):
            output, _ = self.shared_lstm(x, self.hidden)    # 输出 _ 包含(h_n,c_n)的tuple
            sl,bs,_ = output.size()
            avgpool = self.pool(output, bs, False)
            mxpool = self.pool(output, bs, True)
            x = torch.cat([output[-1], mxpool, avgpool], 1)
        return x

    def forward(self, input):
        l_output, r_output = input
        with set_grad_enabled(self.training):
            l_raw_outputs, l_outputs = l_output
            r_raw_outputs, r_outputs = r_output

            encoded_left = l_outputs[-1]
            encoded_right = r_outputs[-1]

            left_output = self.forward_once(encoded_left)
            right_output = self.forward_once(encoded_right)

            # 距离函数 exponent_neg_manhattan_distance
            x = [left_output, right_output]
            malstm_distance = torch.exp(-torch.sum(torch.abs(x[0] - x[1]), dim=1, keepdim=True))
            
        return malstm_distance.view(-1), l_raw_outputs, l_outputs







class Siamese(nn.Module):
    def __init__(self, cfg):
        super(Siamese,self).__init__()
        self.lstm = nn.LSTM(400,64,2,bidirectional=False)

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward_once(self, x):
        x.requires_grad_()
        x,_ = self.lstm(x)
        sl,bs,_ = x.size()
        avgpool = self.pool(x, bs, False)
        mxpool = self.pool(x, bs, True)
        x = torch.cat([x[-1], mxpool, avgpool], 1)
        return x

    def forward(self, input):
        l_output, r_output = input
        with set_grad_enabled(self.training):
            l_raw_outputs, l_outputs = l_output
            r_raw_outputs, r_outputs = r_output

            encoded_left = l_outputs[-1]
            encoded_right = r_outputs[-1]

            left_output = self.forward_once(encoded_left)
            right_output = self.forward_once(encoded_right)
            
            # 距离函数 exponent_neg_manhattan_distance
            malstm_distance = torch.exp(-torch.sum(torch.abs(left_output - right_output), dim=1, keepdim=True))
        
        return malstm_distance.view(-1), l_raw_outputs, l_outputs

def soft_attention_alignment(a, b):
    "Align text representation with neural soft attention"
    e_ij = torch.bmm(a.permute(1,0,2), b.permute(1,2,0))
    w_att_b = F.softmax(e_ij, dim=2)                   # a~ = softmax(attention,dim=1) * b_
    w_att_a = F.softmax(e_ij, dim=1).transpose(1,2)    # b~ = softmax(e_ij,dim=2) * a_
    a_aligned = torch.bmm(w_att_b, b.transpose(0,1)).transpose(0,1)
    b_aligned = torch.bmm(w_att_a, a.transpose(0,1)).transpose(0,1)
    return a_aligned, b_aligned

class ESIM(nn.Module):
    """docstring for ESIM"""
    def __init__(self, cfg):
        super(ESIM, self).__init__()
        self.bn = nn.BatchNorm1d(400)
        self.encode = nn.LSTM(400,300)
        self.compose = nn.LSTM(1200,300)
        self.lin = nn.ModuleList([nn.Linear(1200, 300),nn.Linear(300, 300),nn.Linear(300, 1)])
        self.drop = [nn.Dropout(0.5),nn.Dropout(0.5)]
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(1200), nn.BatchNorm1d(300), nn.BatchNorm1d(300)])
        self.bs = 1

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, input):
        l_output, r_output = input
        with set_grad_enabled(self.training):
            l_raw_outputs, l_outputs = l_output
            r_raw_outputs, r_outputs = r_output

            # embedding
            q1_embed = l_outputs[-1]
            q2_embed = r_outputs[-1]
            # print("encoded_left", type(encoded_left), encoded_left.size())

            sl,bs,_ = q1_embed.size()
            if bs!=self.bs: self.bs=bs

            # Encode
            self.encode.flatten_parameters()
            q1_encoded, _ = self.encode(q1_embed)
            q2_encoded, _ = self.encode(q2_embed)

            # print("q1_encoded", type(q1_encoded), q1_encoded.size())
            # Attention
            q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

            # print("q1_aligned", type(q1_aligned), q1_aligned.size())

            # Compose
            q1_combined = torch.cat([q1_encoded, q1_aligned, torch.sub(q1_encoded, q1_aligned), torch.mul(q1_encoded, q1_aligned)],dim=2)
            q2_combined = torch.cat([q2_encoded, q2_aligned, torch.sub(q2_encoded, q2_aligned), torch.mul(q2_encoded, q2_aligned)],dim=2)

            # print("q1_combined", type(q1_combined), q1_combined.size())

            self.compose.flatten_parameters()
            q1_compare, _ = self.compose(q1_combined)
            q2_compare, _ = self.compose(q2_combined)
    
            # print("q1_compare", type(q1_compare), q1_compare.size())

            # Aggregate
            q1_rep = torch.cat([self.pool(q1_compare, self.bs, False), self.pool(q1_compare, self.bs, True)], dim=1)
            q2_rep = torch.cat([self.pool(q2_compare, self.bs, False), self.pool(q2_compare, self.bs, True)], dim=1)

            # print("q1_rep", type(q1_rep), q1_rep.size())
            # Classifier
            dense = torch.cat([q1_rep, q2_rep],dim=1)


            # print("dense", type(dense), dense.size())
            dense = self.bn2[0](dense)
            dense = self.lin[0](dense)
            # print("dense0", type(dense), dense.size())
            dense = torch.nn.functional.elu(dense)
            dense = self.bn2[1](dense)
            dense = self.drop[0](dense)
            dense = self.lin[1](dense)
            # print("dense1", type(dense), dense.size())
            dense = torch.nn.functional.elu(dense)
            dense = self.bn2[2](dense)
            dense = self.drop[1](dense)
            dense = self.lin[2](dense)
            # print("dense2", type(dense), dense.size())
            output = torch.sigmoid(dense)
            # print("output", type(output), output.size())
            # torch.cuda.empty_cache() # 释放GPU内存很慢，一般不必要

        return output.view(-1), l_raw_outputs, l_outputs


def get_rnn_classifier_similar(similar, bptt, max_seq, n_class, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                      dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, qrnn=False,similar_cfg=None):

    class LM_Encoder(nn.Module):
        def __init__(self):
            super(LM_Encoder,self).__init__()
            self.rnn_enc = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                      dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)
            self.rnn_enc.reset()

        def forward(self, input):
            left_input, right_input = input
            l_output = self.rnn_enc(left_input)
            r_output = self.rnn_enc(right_input)
            return l_output, r_output


    return SequentialRNN(LM_Encoder(),similar(similar_cfg))

class Double_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.binary_cross_entropy  # F.mse_loss
    def fit(self, *args, **kwargs): return super().fit(*args, **kwargs, seq_first=True)

    def save_encoder(self, name): 
        save_model(self.model[0], self.get_model_path(name))
        if hasattr(self, "swa_model"): save_model(self.swa_model[0], self.get_model_path(name[:-4]+"-swa"+name[-4:]))
    def load_encoder(self, name): load_model(self.model[0].rnn_enc, self.get_model_path(name))


class Double_Model(BasicModel):
    def get_layer_groups(self):
        m = self.model[0].rnn_enc
        return [(m.encoder, m.dropouti), *zip(m.rnns, m.dropouths), (self.model[1])]

class Double_TextDataset(Dataset):
    def __init__(self, x1, x2, y, backwards=False, sos=None, eos=None):
        self.x1,self.x2,self.y,self.backwards,self.sos,self.eos = x1,x2,y,backwards,sos,eos

    def xi(self, i, idx):
        if i==1:x = self.x1[idx]
        elif i==2:x = self.x2[idx]

        if self.backwards: x = list(reversed(x))
        if self.eos is not None: x = x + [self.eos]
        if self.sos is not None: x = [self.sos]+x
        return x

    def __getitem__(self, idx):
        return np.array(self.xi(1,idx)),np.array(self.xi(2,idx)),self.y[idx]

    def __len__(self): return len(self.x1)

class Double_DataLoader(DataLoader):
    def get_batch(self, indices):
        res = self.np_collate([self.dataset[i] for i in indices])
        if self.transpose:   res[0] = res[0].T
        if self.transpose:   res[1] = res[1].T
        if self.transpose_y: res[2] = res[2].T
        return res

    def V(self,x):return get_tensor(x, self.pin_memory, self.half)
    def __iter__(self):
        if self.num_workers==0:
            for batch in map(self.get_batch, iter(self.batch_sampler)):
                yield get_tensor(batch, self.pin_memory, self.half)
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                # avoid py3.6 issue where queue is infinite and can result in memory exhaustion
                for c in chunk_iter(iter(self.batch_sampler), self.num_workers*10):
                    for batch in e.map(self.get_batch, c):
                        yield self.V(batch[0]), self.V(batch[1]), self.V(batch[2])


def train_clas(dir_path, similar, cuda_id, lm_id='', clas_id=None, bs=64, cl=1, backwards=False, startat=0, unfreeze=True,
               lr=0.01, dropmult=1.0, bpe=False, use_clr=True, early_stopping=True,
               use_regular_schedule=False, use_discriminative=True, last=False, chain_thaw=False,
               from_scratch=False, train_file_id='',dtype="char", load_learn=False):
    print(f'=== train_clas: dir_path {dir_path}; similar {similar}; cuda_id {cuda_id}; lm_id {lm_id}; clas_id {clas_id}; bs {bs}; cl {cl}; backwards {backwards}; '
        f'dropmult {dropmult} unfreeze {unfreeze} startat {startat}; bpe {bpe}; use_clr {use_clr}; early stopping {early_stopping}'
        f'use_regular_schedule {use_regular_schedule}; use_discriminative {use_discriminative}; last {last};'
        f'chain_thaw {chain_thaw}; from_scratch {from_scratch}; train_file_id {train_file_id};'
        f'dtype {dtype}; load_learn {load_learn}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    intermediate_clas_file = f'{PRE}{clas_id}clas_0_{dtype}'
    final_clas_file = f'{PRE}{clas_id}clas_1_{dtype}'
    if online:
        lm_file = f'{PRE}{lm_id}lm_{dtype}_enc'     # 
    else:
        lm_file = f'{PRE}{lm_id}lm_{dtype}-swa_enc'     # 本地 swa 版本更好
    lm_path = dir_path + "/" + 'models' + "/" + f'{lm_file}.h5'
    assert os.path.exists(lm_path), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh,nl = 70,400,1150,3
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    if backwards:
        trn_sent = np.load(dir_path + "/" + 'tmp' + "/" + f'trn_{IDS}{train_file_id}_{dtype}_bwd.npy')
        val_sent = np.load(dir_path + "/" + 'tmp' + "/" + f'val_{IDS}_{dtype}_bwd.npy')
    else:
        trn_sent = np.load(dir_path + "/" + 'tmp' + "/" + f'trn_{IDS}{train_file_id}_{dtype}.npy')
        val_sent = np.load(dir_path + "/" + 'tmp' + "/" + f'val_{IDS}_{dtype}.npy')

    trn_lbls = np.load(dir_path + "/" + 'tmp' + "/" + f'lbl_trn{train_file_id}.npy')
    val_lbls = np.load(dir_path + "/" + 'tmp' + "/" + f'lbl_val.npy')
    print(dir_path + "/" + 'tmp' + "/" + f'lbl_trn{train_file_id}_{dtype}.npy')
    print(dir_path + "/" + 'tmp' + "/" + f'lbl_val_{dtype}.npy')
    trn_sent, val_sent = trn_sent.reshape(-1,2), val_sent.reshape(-1,2)
    print(trn_sent.shape, val_sent.shape)
    print(trn_lbls.shape, val_lbls.shape)
    trn_lbls = trn_lbls.astype(np.float32)
    val_lbls = val_lbls.astype(np.float32)
    c=int(trn_lbls.max())+1

    if bpe: vs=30002
    else:
        itos = pickle.load(open(dir_path + "/" + 'tmp' + "/" + f'itos_{dtype}.pkl', 'rb'))
        vs = len(itos)

    trn_ds = Double_TextDataset(trn_sent[:,0], trn_sent[:,1], trn_lbls)
    val_ds = Double_TextDataset(val_sent[:,0], val_sent[:,1], val_lbls)
    trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs//2)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    trn_dl = Double_DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = Double_DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)

    md = ModelData(dir_path, trn_dl, val_dl)

    dps = np.array([0.4,0.5,0.05,0.3,0.4])*dropmult
    #dps = np.array([0.5, 0.4, 0.04, 0.3, 0.6])*dropmult
    #dps = np.array([0.65,0.48,0.039,0.335,0.34])*dropmult
    #dps = np.array([0.6,0.5,0.04,0.3,0.4])*dropmult

    m = get_rnn_classifier_similar(similar, bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    learn = Double_Learner(md, Double_Model(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip=25.
    learn.metrics = [f1_np]

    if load_learn:
        learn.load(final_clas_file)
        return learn

    lrm = 2.6
    if use_discriminative:
        lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
    else:
        lrs = lr
    wd = 1e-6
    if not from_scratch:
        learn.load_encoder(lm_file)
    else:
        print('Training classifier from scratch. LM encoder is not loaded.')
        use_regular_schedule = True

    if (startat<1) and not last and not chain_thaw and not from_scratch:
        learn.freeze_to(-1)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8, 3))
        learn.freeze_to(-2)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8, 3))
        learn.save(intermediate_clas_file)
    elif startat==1:
        learn.load(intermediate_clas_file)

    if chain_thaw:
        lrs = np.array([0.0001, 0.0001, 0.0001, 0.0001, 0.001])
        print('Using chain-thaw. Unfreezing all layers one at a time...')
        n_layers = len(learn.get_layer_groups())
        print('# of layers:', n_layers)
        # fine-tune last layer
        learn.freeze_to(-1)
        print('Fine-tuning last layer...')
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8,3))
        n = 0
        # fine-tune all layers up to the second-last one
        while n < n_layers-1:
            print('Fine-tuning layer #%d.' % n)
            freeze_all_but(learn, n)
            learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                      use_clr=None if use_regular_schedule or not use_clr else (8,3))
            n += 1

    if unfreeze:
        learn.unfreeze()
    else:
        learn.freeze_to(-3)

    if last:
        print('Fine-tuning only the last layer...')
        learn.freeze_to(-1)

    if use_regular_schedule:
        print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
        use_clr = None
        n_cycles = cl
        cl = None
    else:
        n_cycles = 1

    callbacks = []
    if False and early_stopping:
        callbacks.append(EarlyStopping(learn, final_clas_file, patience=5))
        print('Using early stopping...')
    if online:callbacks.append(TimerStop(start_time=start_time, total_seconds=7100))
    learn.fit(lrs, n_cycles, wds=wd, cycle_len=cl, use_clr=(8,8) if use_clr else None, callbacks=callbacks)

    print('Plotting lrs...')
    # learn.sched.plot_lr()
    learn.save(final_clas_file)


def r_f1_thresh(y_pred,y_true):
    from sklearn.metrics import f1_score
    import pandas as pd
    e = np.zeros((len(y_true),2))
    e[:,0] = y_pred.reshape(-1)
    e[:,1] = y_true
    f = pd.DataFrame(e)
    m1,m2,fact = 0,1000,1000
    x = np.array([f1_score(y_pred=f.loc[:,0]>thr/fact, y_true=f.loc[:,1]) for thr in range(m1,m2)])
    f1_, thresh = max(x),list(range(m1,m2))[x.argmax()]/fact
    return f.corr()[0][1], f1_, thresh


def eval_learn(learn):
    val_lbls = np.load(model_dir + "atec" + "/" + 'tmp' + "/" + f'lbl_val.npy')
    y_true = val_lbls.astype(np.float32)

    y_pred = learn.predict(use_swa=False)
    print(r_f1_thresh(y_pred,y_true))
    y_pred = learn.predict(use_swa=True)
    print(r_f1_thresh(y_pred,y_true))

def prepare_data():
    # pretrain language model
    articles = 25000
    if online: articles *= 5
    create_lm_csv("wiki", articles=articles)
    # finetune language model
    create_lm_csv("atec")
    for dtype in ["char", "word"]:
        for backwards in [True, False]:
            create_toks(model_dir+"wiki", dtype, backwards)
            tok2id(model_dir+"wiki", 300000, 5, dtype, backwards)
            create_toks(model_dir+"atec", dtype, backwards)
            tok2id(model_dir+"atec", 100000, 5, dtype, backwards)
    # train class model
    create_ids_lbls()

def main(similar_cls):
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

    
    def step1(dtype, backwards):    # 训练一个语言模型
        '''训练语言模型，精度：    4epoch，4*12391次迭代，char_bwd=0.350832'''
        '''online模型，精度： 2epoch char_bwd = 0.389244  char_fwd = 0.396402'''
        start = time.time()
        pretrain_lm(model_dir+"wiki", cuda_id=0,lr=1e-3, cl = 4, preload=False, dtype=dtype, backwards=backwards) # cl=12
        print("pretrain lm used:%f"%(time.time() - start))

    def step2(dtype, backwards):    # finetune当前语言模型
        '''finetune语言模型，精度：   25epoch 25*745次迭代， char_bwd=0.65192    char_fwd=0.652922'''
        '''online模型，      精度：    5epoch 5*3628次迭代， char_bwd=0.665007   char_fwd=0.675306'''
        start = time.time()
        finetune_lm(model_dir+"atec", model_dir+"wiki", cuda_id=0, cl=17, dtype=dtype, backwards=backwards, preload=True, startat=0) # cl=25
        print("finetune lm used:%f"%(time.time() - start))

    def step3(dtype, backwards):    # 基于语言模型训练分类器
        '''训练分类器， 精度：  2+6epoch 0.494 42min'''

        '''
        ESIM(25epoch 10548s)
        (0.009264956584760111, 0.2990871369294606, 0.0)
        (-0.0022201639439278402, 0.29918645193425203, 0.19)

        Siamese(25epoch 8866s)
        (0.46805806932410426, 0.5372168284789643, 0.293)
        (0.4387573102493323, 0.5076566125290023, 0.432)

        Siamese_Baseline(25epoch 8370s)
        (0.4641013241384524, 0.5223529411764706, 0.308)
        (0.3982938312050042, 0.4770240700218818, 0.491)

        Siamese_change(10epoch )
        '''
        start = time.time()
        train_clas(model_dir+"atec", similar_cls, cuda_id=0, cl=10, dtype=dtype, backwards=backwards, startat=0, load_learn=False) #cl=50
        print("train clas lm used:%f"%(time.time() - start))

    def find_best_thr(dtype, backwards):
        learn = train_clas(model_dir+"atec", similar_cls, cuda_id=0, cl=1, dtype=dtype, backwards=backwards, load_learn=True)
        eval_learn(learn)

    def get_result(dtype, backwards):
        if not online: df1 = pd.read_csv(model_dir+"atec_nlp_sim_train.csv",sep="\t", header=None, names =["id","sent1","sent2","label"], encoding="utf8")
        if online: best_threshold = 0.296
        else: best_threshold = 0.296

        val_sent = np.hstack([np.array(df1["sent1"]).reshape(-1,1), np.array(df1["sent2"]).reshape(-1,1)]).reshape(-1)
        val_sent = sent_token2id(val_sent, model_dir+"atec", dtype, backwards).reshape(-1, 2)
        val_ds = Double_TextDataset(val_sent[:,0], val_sent[:,1], np.ones(val_sent.shape[0]))
        val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
        val_dl = Double_DataLoader(val_ds, bs=64, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
        learn = train_clas(model_dir+"atec", similar_cls, cuda_id=0, cl=1, dtype=dtype, backwards=backwards, load_learn=True)
        result = learn.predict_dl(val_dl,use_swa=False)
        result = result > best_threshold
        print(sum(result))
        result = learn.predict_dl(val_dl,use_swa=True)
        result = result > best_threshold
        print(sum(result))
        df_output = pd.concat([df1["id"],pd.Series(result,name="label",dtype=np.int32)],axis=1)
        if online: topai(1,df_output)
        else: print(df_output)

    dtype, backwards = "char",True
    # dtype, backwards = "char",False
    # dtype, backwards = "word",True
    # dtype, backwards = "word",False
    
    step1(dtype,backwards)
    step2(dtype,backwards)
    step3(dtype,backwards)
    find_best_thr(dtype,backwards)
    get_result(dtype,backwards)



if __name__ == '__main__':  
    # prepare_data()
    
    similar_cls = Siamese
    main(similar_cls)
    # 清除不用的GPU缓存，使Keras有显存可用
    torch.cuda.empty_cache()