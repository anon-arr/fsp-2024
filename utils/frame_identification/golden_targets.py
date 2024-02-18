import pandas as pd
from Hydra.target_identification.candidate_identifier import LexicalUnitManager
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from nltk import sent_tokenize
import numpy as np
import torch

class GoldTargetFrameIDDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, tokenizer: AutoTokenizer):
        self.sentence = dataset.sentence

        # self.target_token_start = dataset.target_token_span.apply(lambda x: x[0])
        # self.target_token_end = dataset.target_token_span.apply(lambda x: x[1])
        self.target_token_span = None
        if 'target_token_span' in dataset.columns:
            self.target_token_span = torch.vstack([torch.Tensor(x).long() for x in dataset.target_token_span.values])

        self.frame = dataset.possible_frame
        # Only get first sentence of frame definition
        self.frame_definition = dataset.frame_definition.apply(lambda x: sent_tokenize(x)[0])

        self.lu = dataset.lu
        # Only get first sentence of LU definition
        self.lu_definition = dataset.lu_definition.apply(lambda x: sent_tokenize(x)[0])

        self.label = torch.tensor(dataset.label.values).type(torch.LongTensor)
        
        if 'frame_fes' in dataset.columns:
            self.frame_fes = dataset.frame_fes
            self.inputs = [f'{sentence} {tokenizer.sep_token} {frame_name} ({", ".join(frame_fes)}): {frame_def} {lu_name}: {lu_def}'
                        for sentence, frame_name, frame_def, lu_name, lu_def, frame_fes in
                        zip(self.sentence, self.frame, self.frame_definition, self.lu, self.lu_definition, self.frame_fes)]

        # Input format (RoBERTa)
        # Sentence <sep> frame_name: frame_def lu_name.pos: lu_def 
        
        # Baseline
        else:
            self.inputs = [f'{sentence} {tokenizer.sep_token} {frame_name}: {frame_def} {lu_name}: {lu_def}'
                            for sentence, frame_name, frame_def, lu_name, lu_def in
                            zip(self.sentence, self.frame, self.frame_definition, self.lu, self.lu_definition)]
        
        # Basline + fe names
        
        # Baseline w/o lu definition # (worse)
        # self.inputs = [f'{sentence} {tokenizer.sep_token} {frame_name}: {frame_def}'
        #                 for sentence, frame_name, frame_def, lu_name, lu_def in
        #                 zip(self.sentence, self.frame, self.frame_definition, self.lu, self.lu_definition)]
        
        # Tokenize inputs
        self.tokenized_inputs = tokenizer(self.inputs, padding=True, truncation=True, return_tensors='pt')
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Return values for huggingface trainer
        return {'input_ids': self.tokenized_inputs['input_ids'][idx],
                'attention_mask': self.tokenized_inputs['attention_mask'][idx],
                'labels': self.label[idx], 'target_token_span': self.target_token_span[idx] if self.target_token_span is not None else None}

def process_test_dataset(dataset: pd.DataFrame, frame_info: pd.DataFrame, 
                         frame_lu_defs: pd.DataFrame, lu_manager: LexicalUnitManager, 
                         use_pos: bool = False):
    # Drop duplicate rows, ignore tokens since they are unhashable
    dataset = dataset.drop_duplicates(['frame_name', 'sentence', 'target_token_span'])

    # Reset indices
    dataset = dataset.reset_index(drop=True)
    
    dataset = dataset.merge(frame_info[['name', 'definition', 'fes']], 
                            how='left', left_on='frame_name', right_on='name')
    
    dataset['fes'] = dataset.fes.apply(lambda x: list(x.keys()) if x is not np.nan else [])
    dataset = dataset.rename(columns={'definition':'frame_definition', 
                                      'fes':'frame_fes'}).drop(columns=['name'])
    
    dataset['lu_name_no_pos'] = dataset['lu_name'].apply(lambda x: x.split('.')[0]) 
    dataset = dataset.merge(frame_lu_defs, how='left', 
                            left_on=['frame_name', 'lu_name_no_pos'],
                            right_on=['frame_name', 'lu_no_pos'])
    
    dataset = dataset.rename(columns={'lu_def':'lu_definition'})
    dataset['label'] = 1
    dataset['possible_frame'] = dataset.frame_name

    return dataset


def process_dataset(dataset: pd.DataFrame, frame_info: pd.DataFrame, 
                    frame_lu_defs: pd.DataFrame, lu_manager: LexicalUnitManager, 
                    use_pos: bool = False):
    """
    Process dataset for frame identification task using golden targets.
    
    Args:
        dataset (pd.DataFrame): Dataframe containing the dataset.
        frame_info (pd.DataFrame): Dataframe containing frame information.
        lu_manager (utils.lu.LUManager): LUManager object containing LUs.
        use_pos (bool): Whether to use POS tags in LU definitions.
    """

    # Drop duplicate rows, ignore tokens since they are unhashable
    dataset = dataset.drop_duplicates(['frame_name', 'sentence', 'target_token_span'])

    # Reset indices
    dataset = dataset.reset_index(drop=True)

    # Add candidate frames to each sample
    # Add possible frames for each LU
    dataset['possible_frame'] = dataset.lu_name.apply(lambda x: lu_manager.find_frames_from_lu(x))

    # Convert each set of frames into individual rows
    dataset = dataset.explode('possible_frame')

    # Add definitions for each sample
    # Add frame definitions to samples
    dataset = dataset.merge(frame_info[['name', 'definition', 'fes']], 
                            how='left', left_on='possible_frame', right_on='name')

    dataset['fes'] = dataset.fes.apply(lambda x: list(x.keys()) if x is not np.nan else [])
    
    # Drop unnecessary columns and rename
    dataset = dataset.rename(columns={'definition':'frame_definition', 
                                      'fes':'frame_fes'}).drop(columns=['name'])
    
    # Add LU definitions to samples
    if use_pos:
        dataset = dataset.merge(frame_lu_defs, how='left', 
                                left_on=['possible_frame', 'lu_name'],
                                right_on=['frame_name', 'lu'])
    else:
        dataset['lu_name_no_pos'] = dataset['lu_name'].apply(lambda x: x.split('.')[0]) 
        dataset = dataset.merge(frame_lu_defs, how='left', 
                                left_on=['possible_frame', 'lu_name_no_pos'],
                                right_on=['frame_name', 'lu_no_pos'])

    # Drop unnecessary columns and rename
    dataset = dataset.rename(columns={'lu_def':'lu_definition', 'frame_name_x': 'frame_name'})

    # Set label for each row according to whether the candidate matches the actual frame
    # and if the lus match
    dataset['label'] = ((dataset.frame_name == dataset.possible_frame) & (dataset.lu == dataset.lu_name)).astype(int)

    # Get input_ids and attention_mask mapping
    # sent_tok_mapping = pd.DataFrame([(sentence, tokens['input_ids'], tokens['attention_mask'])
    #                                     for sentence, tokens in dataset[['sentence','tokenized_sentence']].values],
    #                                 columns=['sentence', 'input_ids', 'attention_maks']).drop_duplicates(['sentence']).reset_index(drop=True)
    # dataset = dataset.merge(train_sent_tok_mapping, how='left', left_on='sentence', right_on='sentence').drop(columns=['tokenized_sentence'])


    return dataset


def get_negative_samples(dataset: pd.DataFrame, frame_info: pd.DataFrame, frame_lu_defs: pd.DataFrame):
    # For each sentence, sample additional negative examples
    dataset_with_negatives = None

    for sentence_group in dataset.dropna().groupby('sentence'):
        sentence_df = sentence_group[1]

        # Get all lus used in sentence, we don't want to sample from these
        # May want to just generate all candidates in the future and use them instead
        sentence_frames = pd.concat((sentence_df[['frame_name', 'lu_name']], 
                                    sentence_df[['possible_frame', 'lu']].rename(
                                        columns={'possible_frame': 'frame_name', 
                                                'lu': 'lu_name'}))).drop_duplicates() 

        # Get all available lus
        available_lus = pd.concat((frame_lu_defs, 
                                sentence_frames.rename(columns={'lu_name':'lu'}))).drop_duplicates(['frame_name', 'lu'], 
                                                                                                    keep=False)

        # Get target groups
        for target_group in sentence_df.groupby(['sentence', 'target_token_span']):
            target_group_df = target_group[1]

            # Always have at least 5 samples
            negatives_to_sample = 5 - len(target_group_df)

            # Sample negative examples
            if negatives_to_sample > 0:
                negative_examples = available_lus.sample(n=negatives_to_sample, replace=False)
                available_lus = available_lus.drop(negative_examples.index)
                negative_examples = negative_examples.merge(frame_info[['name', 'definition']], 
                                        left_on='frame_name', right_on='name').drop('name', axis=1)
                negative_examples = negative_examples.rename(columns={'definition': 'frame_definition',
                                                                    'lu_def': 'lu_definition'})
            
                new_samples = pd.concat((target_group_df, negative_examples))
            else:
                new_samples = target_group_df

            # new_samples.target_token_span.fillna(target_group_df.target_token_span.iloc[0], inplace=True)
            new_samples.target_token_span = new_samples.target_token_span.apply(lambda x: target_group_df.target_token_span.iloc[0] if pd.isnull(x) else x)
            new_samples.sentence.fillna(target_group_df.sentence.iloc[0], inplace=True)
            new_samples.label.fillna(0, inplace=True)
            
            # Append to train dataset
            if dataset_with_negatives is None:
                dataset_with_negatives = new_samples
            else:
                dataset_with_negatives = pd.concat((dataset_with_negatives, new_samples))

    dataset_with_negatives.reset_index(drop=True, inplace=True)
    return dataset_with_negatives


def create_multiclass_dataset(dataset: pd.DataFrame, frame_info: pd.DataFrame, frame_lu_defs: pd.DataFrame, lu_manager: LexicalUnitManager, use_pos: bool = False):
    # Drop duplicate rows, ignore tokens since they are unhashable
    dataset = dataset.drop_duplicates(['frame_name', 'sentence', 'target_token_span'])

    # Reset indices
    dataset = dataset.reset_index(drop=True)

    # Convert frame_name to categorical label
    frame_info[frame_info.name == 'Becoming'].index[0]

    # Add definitions for each sample
    # Add frame definitions to samples
    dataset = dataset.merge(frame_info[['name', 'definition']], 
                            how='left', left_on='frame_name', right_on='name')

    # Drop unnecessary columns and rename
    dataset = dataset.rename(columns={'definition':'frame_definition'}).drop(columns=['name'])
    
    # Add LU definitions to samples
    if use_pos:
        dataset = dataset.merge(frame_lu_defs, how='left', 
                                left_on=['frame_name', 'lu_name'],
                                right_on=['frame_name', 'lu'])
    else:
        dataset['lu_name_no_pos'] = dataset['lu_name'].apply(lambda x: x.split('.')[0]) 
        dataset = dataset.merge(frame_lu_defs, how='left', 
                                left_on=['frame_name', 'lu_name_no_pos'],
                                right_on=['frame_name', 'lu_no_pos'])

    # Drop unnecessary columns and rename
    dataset = dataset.rename(columns={'lu_def':'lu_definition', 'frame_name_x': 'frame_name'})

    # Set label for each row according to whether the candidate matches the actual frame
    # and if the lus match
    dataset['label'] = ((dataset.frame_name == dataset.possible_frame) & (dataset.lu == dataset.lu_name)).astype(int)

    # Get input_ids and attention_mask mapping
    # sent_tok_mapping = pd.DataFrame([(sentence, tokens['input_ids'], tokens['attention_mask'])
    #                                     for sentence, tokens in dataset[['sentence','tokenized_sentence']].values],
    #                                 columns=['sentence', 'input_ids', 'attention_maks']).drop_duplicates(['sentence']).reset_index(drop=True)
    # dataset = dataset.merge(train_sent_tok_mapping, how='left', left_on='sentence', right_on='sentence').drop(columns=['tokenized_sentence'])


    return dataset