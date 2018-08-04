import fire
def train_clas(prefix, cuda_id, lm_id='', clas_id=None, bs=64, cl=1, backwards=False, startat=0, unfreeze=True,
				lr=0.01, dropmult=1.0, pretrain=True, bpe=False, use_clr=True,
				use_regular_schedule=False, use_discriminative=True, last=False, chain_thaw=False,
				from_scratch=False, train_file_id=''):
	pass

if __name__ == '__main__':
	fire.Fire(train_clas)
