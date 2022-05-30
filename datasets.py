import torch
from torch.utils.data import Dataset, BatchSampler, Sampler, DistributedSampler
import os
from utils import trim_batch, encode_file_bart, encode_scigen_mask_nums
from typing import Iterator, List
import random
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='test.txt', level=logging.DEBUG,
                    format='%(asctime)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class BartDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        **kwargs):
        super().__init__()

        self.char_level_representation = kwargs['char_level_representation']

        self.tokenizer = tokenizer
        source_data = os.path.join(kwargs['data_dir'], kwargs['type_path'] + ".source")
        target_data = os.path.join(kwargs['data_dir'], kwargs['type_path'] + ".source")
        self.source = encode_file_bart(tokenizer, source_data, kwargs['max_source_length'], kwargs['scientific_notation'], kwargs['char_level_representation'])
        self.target = encode_file_bart(tokenizer, target_data, kwargs['max_target_length'], kwargs['scientific_notation'], kwargs['char_level_representation'])

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):

        source_ids = self.source[index]["input_ids"].squeeze()
        target_ids = self.target[index]["input_ids"].squeeze()
        src_mask = self.source[index]["attention_mask"].squeeze()

        if self.char_level_representation:
            source_splitted_indices = self.source[index]['splitted_indices']
            
        else:
            source_splitted_indices = None
            
        return {"source_ids": source_ids, 
            "source_mask": src_mask, 
            "target_ids": target_ids,
            'source_splitted_indices': source_splitted_indices}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        source_splitted_indices = [x["source_splitted_indices"] for x in batch]        
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)

        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)

        _source_splitted_indices = []
        if self.char_level_representation:
            for indices in source_splitted_indices:
                _source_splitted_indices.append([ind for ind in indices if ind[1] <= source_ids.size()[1]])
        
        return {"source_ids": source_ids, 
            "source_mask": source_mask, 
            "target_ids": y, 
            'source_splitted_indices': _source_splitted_indices}


class SciGenMaskedEntity(Dataset):
    def __init__(
        self,
        tokenizer,
        contrastive=False,
        **kwargs):
        super().__init__()

        self.char_level_representation = kwargs['char_level_representation']
        self.tokenizer = tokenizer
        data_path_source = os.path.join(kwargs['data_dir'], kwargs['type_path'] + ".source")
        data_path_target = os.path.join(kwargs['data_dir'], kwargs['type_path'] + ".target")
        self.samples, self.pairs = encode_scigen_mask_nums(tokenizer, data_path_source, data_path_target, 
            kwargs['max_source_length'], kwargs['max_target_length'], kwargs['scientific_notation'], 
            self.char_level_representation, contrastive, kwargs['verbalized'])

        logger.info('Extraced %s masked samples from %s and %s' % (len(self.samples), data_path_source, data_path_target))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        source_ids = self.samples[index]['source']["input_ids"].squeeze()
        target_ids = self.samples[index]['target']["input_ids"].squeeze()
        src_mask = self.samples[index]['source']["attention_mask"].squeeze()        
        length_masked_target = self.samples[index]['length_masked_target']        
        masked_ids_target = self.samples[index]['masked_ids_target']
        source = self.samples[index]['original']
        
        if self.char_level_representation: 
            source_splitted_indices = self.samples[index]['source']['splitted_indices']
            
        else:
            source_splitted_indices = None
            
        return {"source_ids": source_ids,
            "source": source, 
            "source_mask": src_mask, 
            "target_ids": target_ids,            
            'source_splitted_indices': source_splitted_indices,                        
            'length_masked_target': length_masked_target,            
            'masked_ids_target': masked_ids_target}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        y = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, y

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])        
        pad_token_id = self.tokenizer.pad_token_id
        source_splitted_indices = [x["source_splitted_indices"] for x in batch]                
        length_masked_target = [x['length_masked_target'] for x in batch]                      
        masked_ids_target = [x["masked_ids_target"] for x in batch]   
        
        # for finding the embedding of the masked tokens after BERT pass, it is 
        # necessary that the y's, which are used as decoder input ids, have the 
        # same length as the original target (which contains the masked token and
        # is concatenated on the table as input)
        y = trim_batch(target_ids, pad_token_id, trim_pos=max(length_masked_target))

        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)

        _source_splitted_indices = []
                
        if self.char_level_representation:
            for indices in source_splitted_indices:
                _source_splitted_indices.append([ind for ind in indices if ind[1] <= source_ids.size()[1]])        
        
        return {"source_ids": source_ids, 
            "source_mask": source_mask, 
            "target_ids": y, 
            'source_splitted_indices': _source_splitted_indices,                                    
            'length_masked_target': length_masked_target,                        
            'masked_ids_target': masked_ids_target}

class BatchSampler(BatchSampler):

    def __init__(self, sampler: Sampler, batch_size, shuffle=True, drop_last=False):
        super().__init__(sampler, batch_size, shuffle)
        self.shuffle = shuffle 
        self.sampler = sampler

        # in this case, it is an distributed samples which doesn't have the field "data_source"
        if type(sampler) == DistributedSampler:
            self.pairs = self.sampler.dataset.pairs
        else:  
            self.pairs = self.sampler.data_source.pairs
        self.batch_size = batch_size        

        self.batches = self.create_batches()       
        self.length=len(self.batches)
        self.drop_last = drop_last

    def create_batches(self):

        batches = []
                        
        if self.shuffle:
            # random.shuffle(self.masked_indices_keys)
            random.shuffle(self.pairs)
        
        while len(self.pairs) > 0:            
            if len(self.pairs) < self.batch_size:
                self.batch_size = len(self.pairs)
            if len(self.pairs) < 2:
                break

            batch = []

            while len(batch) < self.batch_size:
                pair = self.pairs.pop()
                batch += [pair[0], pair[1]]

            if len(batch) >= 2:                                
                batches.append(batch)

        if not self.drop_last:
            if len(batch) >= 2:        
                batches.append(batch)

        return batches
    
    def __iter__(self) -> Iterator[List[int]]:        
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return self.length
