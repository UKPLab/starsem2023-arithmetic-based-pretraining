import torch
from torch.nn import CrossEntropyLoss

def MNRLoss(embeddings_a:torch.Tensor, embeddings_b:torch.Tensor, scale:float = 20.0):
    '''
    Calculate Multiple Negative Ranking Loss according to the implementation used in SBERT
    (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py)

    We use the implementation of the MNRLoss in S-BERT as reference (https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/MultipleNegativesRankingLoss.py) which uses sentence embeddings. The sentence embeddings are obtained using a mean pooling operation which takes the attention mask into account (https://www.sbert.net/examples/applications/computing-embeddings/README.html?highlight=sentence_embedding#sentence-embeddings-with-transformers). Since in our case all embeddings of a number's tokens matter, I think we're fine with just averaging the hidden states together. 
    '''

    loss = CrossEntropyLoss()

    # calculate cos sim
    a_norm = torch.nn.functional.normalize(embeddings_a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(embeddings_b, p=2, dim=1)
    scores = torch.mm(a_norm, b_norm.transpose(0, 1)) * scale
    # generate labels
    labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
    return loss(scores, labels)

def masked_loss(batch, lm_labels, predictions, prev_loss, step):

    loss = CrossEntropyLoss()

    if prev_loss is None:
        prev_loss = torch.zeros(1, requires_grad=True)

    if 'masked_ids_target' in batch and len(batch['masked_ids_target']) >= 2:
        # masked_ids_target -> positions of masked token in the target sequence
        # lm_labels -> target sequence; lm_label[i][j] -> ith target sequence and token at position j
        # this should already work with span masking (as long as the target sequence is tokenized in the same way 
        # as the source sequence)
        [print(batch['masked_ids_target'][i], lm_labels[i])
            for i in range(0, len(batch['masked_ids_target']), step)
            for j in batch['masked_ids_target'][i]]
        
        masked_labels = [lm_labels[i][j] 
            for i in range(0, len(batch['masked_ids_target']), step) 
            for j in batch['masked_ids_target'][i]]

        if len(masked_labels) > 0:
            # outputs[1] -> logits; outputs[1][i] -> logits for a specific sample from the batch
            # outputs[1][i][j] -> logits for the j-th item in sample i
            masked_labels = torch.stack(masked_labels)

            # same as with masked_labels
            masked_predictions = torch.stack([predictions[i][j] 
                for i in range(0, len(batch['masked_ids_target']), step)
                for j in batch['masked_ids_target'][i] 
                if len(predictions[i][j])])

            return loss(masked_predictions, masked_labels)

    return prev_loss
