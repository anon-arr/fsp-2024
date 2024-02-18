import pandas as pd
import Hydra.config as config
from Hydra.utils.data import convert_lists_to_tuples
from nltk import sent_tokenize
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

def load_dataset(file_path, args, lu_manager, frame_info, frame_lu_defs):
    dataset = pd.read_json(f'{config.processed_path}/{file_path}')
    # dataset = pd.read_json(f'{config.processed_path}/train_fulltext_data_tokenized.json')

    cols_to_convert = ['target_merged', 'target_token_span', 'fe_span', 'fe_token_span']
    dataset = convert_lists_to_tuples(dataset, cols_to_convert)

    # Set relevant columns
    if args.gold_targets:
        # Columns for gold targets frame id
        relevant_columns = ['frame_name', 'sentence', 'target_merged', 'lu_name', 'tokenized_sentence', 'target_token_span']
    else:
        assert False, 'Candidate targets not implemented yet'
        
    # Remove irrelevant columns
    dataset = dataset[relevant_columns]

    # Drop duplicate rows, ignore tokens since they are unhashable
    dataset = dataset.drop_duplicates(['frame_name', 'sentence', 'target_token_span'])

    # Reset indices
    dataset = dataset.reset_index(drop=True)

    if args.use_negatives:
        # TODO: Get negative samples
        # Idea: Sample num_epochs * num_samples_per_epoch negative samples
        #      why: This way we are sampling different negatives each time
        # Idea: Slowly introduce negative samples, i.e., in epoch 1, sample 1 negative sample per positive sample
        #       in epoch 2, sample 2 negative samples per positive sample, etc.
        #      why: This way we are not overwhelming the model with negative samples
        #      Maybe on last epoch we can use all frames as negative samples somehow?
        # Idea: Sample negative samples from the same frame as the positive sample, but different LUs
        assert False, 'Negative sampling not implemented yet'
    else:
        dataset['possible_frame'] = dataset.lu_name.apply(lambda x: lu_manager.find_frames_from_lu(x))

        # Convert each set of frames into individual rows
        dataset = dataset.explode('possible_frame').reset_index(drop=True)

    if args.use_frame_definitions:
        # Add frame definitions to samples
        dataset = dataset.merge(frame_info[['name', 'definition']], 
                                how='left', left_on='possible_frame', right_on='name')
        dataset.rename(columns={'definition': 'frame_definition'}, inplace=True)
        dataset.drop(columns='name', inplace=True)
        dataset = dataset.dropna().reset_index(drop=True)

        # Clip frame definitions to first sentence
        if args.clip_frame_definitions:
            # Clip frame definitions to first sentence
            dataset['frame_definition'] = dataset.frame_definition.apply(lambda x: sent_tokenize(x)[0])
        
    if args.use_lu_definitions:
        # Add LU definitions to samples
        dataset = dataset.merge(frame_lu_defs[['frame_name', 'lu', 'lu_def']], how='left', 
                                left_on=['lu_name', 'possible_frame'], right_on=['lu', 'frame_name'])
        dataset.rename(columns={'lu_def': 'lu_definition', 'frame_name_x':'frame_name'}, inplace=True)
        dataset.drop(columns=['lu', 'frame_name_y'], inplace=True)

        # Fill NA values with a random LU definition from the same frame using frame_lu_defs
        a = dataset.groupby('possible_frame')['possible_frame'].transform(lambda x: frame_lu_defs[frame_lu_defs.frame_name == x.values[0]]['lu_def'].sample(1).values[0])
        dataset['lu_definition'] = dataset.lu_definition.fillna(a)
        dataset = dataset.dropna().reset_index(drop=True)

    if args.append_fe_names:
        dataset = dataset.merge(frame_info[['name', 'fes']], how='left', 
                                left_on='possible_frame', right_on='name')
        dataset.rename(columns={'fes': 'frame_fes'}, inplace=True)
        dataset.drop(columns='name', inplace=True)
        dataset['frame_fes'] = dataset.frame_fes.apply(lambda x: list(x.keys()))

    dataset['label'] = (dataset.frame_name == dataset.possible_frame).astype(int)
    
    return dataset


def create_input_string(args, tokenizer, sentence, possible_frame, 
                        frame_definition, lu_definition, frame_fes):
    """ Create input string for frame identification model

    Args:
        row (pd.Series): Row of dataset
        args (argparse.Namespace): Arguments
        tokenizer (transformers.AutoTokenizer): Tokenizer
    
    Returns:
        str: Input string for frame identification model
    """
    _frame_span = None
    _lu_span = None
    _fe_span = None

    if args.use_frame_definitions:
        _frame_span = f'{possible_frame}: {frame_definition.strip()}'
    else:
        _frame_span = f'{possible_frame}'

    if args.use_lu_definitions:
        _lu_span = f'{":".join(lu_definition.split(":")[1:]).strip()}'

    if args.append_fe_names:
        _fe_span = ' '.join(frame_fes)
    
    if args.add_fsp_tokens:
        _lu_span = f'<lu> {_lu_span.strip()} </lu>' if _lu_span is not None else None
        _fe_span = ' '.join([f'<e> {fe} </e>' for fe in frame_fes]) if _fe_span is not None else None

        if args.use_frame_definitions:
            _fr_def = frame_definition.strip()
            if frame_fes is not None:
                for fe in frame_fes:
                    _fr_def = _fr_def.replace(fe, f'<e> {fe} </e>')
            _frame_span = f'<f> {possible_frame} </f> : {_fr_def}'
        else:
            _frame_span = f'<f> {_frame_span.strip()} </f>'

    return f'{sentence} {tokenizer.sep_token} {_lu_span if _lu_span is not None else ""} {_frame_span if _frame_span is not None else ""} {_fe_span if _fe_span is not None else ""} {tokenizer.sep_token}'


class FrameIdentificationDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, tokenizer: AutoTokenizer, args):
        self.sentence = dataset.sentence

        self.target_token_span = None
        if 'target_token_span' in dataset.columns:
            self.target_token_span = torch.vstack([torch.Tensor(x).long() for x in dataset.target_token_span.values])

        self.frame = dataset.possible_frame
        self.frame_definition = dataset.frame_definition if args.use_frame_definitions else [None] * len(dataset)
        self.lu = dataset.lu_name
        self.lu_definition = dataset.lu_definition.apply(lambda x: sent_tokenize(x)[0]) if args.use_lu_definitions else [None] * len(dataset)
        self.label = torch.tensor(dataset.label.values).type(torch.LongTensor)
        self.frame_fes = dataset.frame_fes if args.append_fe_names else [None] * len(dataset)

        # Create inputs
        self.inputs = [create_input_string(args, tokenizer, sentence, possible_frame, frame_definition, lu_definition, frame_fes)
                        for sentence, possible_frame, frame_definition, lu_definition, frame_fes in
                        zip(self.sentence, self.frame, self.frame_definition, self.lu_definition, self.frame_fes)]
        
        # Tokenize inputs
        self.tokenized_inputs = tokenizer(self.inputs, padding=True, return_tensors='pt')
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Return values for huggingface trainer
        return {'input_ids': self.tokenized_inputs['input_ids'][idx],
                'attention_mask': self.tokenized_inputs['attention_mask'][idx],
                'labels': self.label[idx], 'target_token_span': self.target_token_span[idx] if self.target_token_span is not None else None}
