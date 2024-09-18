from transformers import RobertaModel, RobertaTokenizer
import torch
from lama.modules.base_connector import *

class RobertaVocab(object):
    def __init__(self, roberta):
        self.roberta = roberta

    def __getitem__(self, arg):
        value = ""
        try:
            predicted_token_bpe = self.roberta.tokenizer.convert_ids_to_tokens([arg])[0]
            if predicted_token_bpe.strip() == ROBERTA_MASK or predicted_token_bpe.strip() == ROBERTA_START_SENTENCE:
                value = predicted_token_bpe.strip()
            else:
                value = self.roberta.tokenizer.decode([arg]).strip()
        except Exception as e:
            print(arg)
            print(predicted_token_bpe)
            print(value)
            print(f"Exception {e} for input {arg}")
        return value


class Roberta(Base_Connector):
    def __init__(self, args):
        super().__init__()
        roberta_model_dir = args.roberta_model_dir
        roberta_model_name = args.roberta_model_name
        
        # Load the RoBERTa model and tokenizer
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self._build_vocab()
        self._init_inverse_vocab()
        self.max_sentence_length = 128 # args.max_sentence_length

    def _cuda(self):
        self.model.cuda()

    def _build_vocab(self):
        self.vocab = []
        for key in range(self.tokenizer.vocab_size):
            predicted_token_bpe = self.tokenizer.convert_ids_to_tokens([key])[0]
            value = self.tokenizer.decode([key]).strip()
            
            # Check if value is not empty before accessing its first character
            if value and value[0] == " ":  # if the token starts with a whitespace
                value = value.strip()
            else:
                value = f"_{value}_"
            
            # Avoid duplicate entries in the vocab
            if value in self.vocab:
                value = f"{value}_{key}"

            self.vocab.append(value)

    def get_id(self, input_string):
        tokens = self.tokenizer.encode(input_string, add_special_tokens=False)
        return tokens

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self._cuda()

        tensor_list = []
        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        
        for masked_inputs_list in sentences_list:
            tokens_list = []
            for masked_input in masked_inputs_list:
                masked_input = masked_input.replace(MASK, self.tokenizer.mask_token)
                tokens = self.tokenizer.encode(masked_input, add_special_tokens=True)
                tokens_list.append(torch.tensor(tokens))

            tokens = torch.cat(tokens_list)[: self.max_sentence_length]
            output_tokens_list.append(tokens.long().cpu().numpy())

            if len(tokens) > max_len:
                max_len = len(tokens)
            tensor_list.append(tokens)
            masked_index = (tokens == self.tokenizer.mask_token_id).nonzero().numpy()
            for x in masked_index:
                masked_indices_list.append([x[0]])

        pad_id = self.tokenizer.pad_token_id
        tokens_list = []
        for tokens in tensor_list:
            pad_length = max_len - len(tokens)
            if pad_length > 0:
                pad_tensor = torch.full([pad_length], pad_id, dtype=torch.int)
                tokens = torch.cat((tokens, pad_tensor))
            tokens_list.append(tokens)

        batch_tokens = torch.stack(tokens_list)

        with torch.no_grad():
            self.model.eval()
            outputs = self.model(batch_tokens.to(self.model.device))
            log_probs = outputs.logits

        return log_probs.cpu(), output_tokens_list, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # To be implemented
        return None
