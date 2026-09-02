import torch
from torch import nn

from recon.vocab_utils import resize_token_embeddings_if_needed


class _Tokenizer:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class _TinyCausalLM:
    def __init__(self, vocab_size, hidden_size=3):
        self.input_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.output_embeddings = nn.Linear(hidden_size, vocab_size, bias=False)
        self.resize_calls = []

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    def resize_token_embeddings(self, new_size):
        self.resize_calls.append(new_size)
        old_input = self.input_embeddings
        old_output = self.output_embeddings
        self.input_embeddings = nn.Embedding(new_size, old_input.embedding_dim)
        self.output_embeddings = nn.Linear(old_output.in_features, new_size, bias=False)
        with torch.no_grad():
            input_rows = min(new_size, old_input.num_embeddings)
            output_rows = min(new_size, old_output.out_features)
            self.input_embeddings.weight[:input_rows].copy_(old_input.weight[:input_rows])
            self.output_embeddings.weight[:output_rows].copy_(old_output.weight[:output_rows])


def test_preserves_checkpoint_padding_and_action_rows():
    checkpoint_vocab = 152064
    tokenizer_vocab = 151646
    model = _TinyCausalLM(checkpoint_vocab)
    action_row = 151642
    with torch.no_grad():
        model.input_embeddings.weight[action_row].fill_(3.25)
        model.output_embeddings.weight[action_row].fill_(-2.5)
    input_row = model.input_embeddings.weight[action_row].clone()
    output_row = model.output_embeddings.weight[action_row].clone()

    result = resize_token_embeddings_if_needed(model, _Tokenizer(tokenizer_vocab))

    assert result == checkpoint_vocab
    assert model.resize_calls == []
    assert model.input_embeddings.num_embeddings == checkpoint_vocab
    assert model.output_embeddings.out_features == checkpoint_vocab
    assert torch.equal(model.input_embeddings.weight[action_row], input_row)
    assert torch.equal(model.output_embeddings.weight[action_row], output_row)


def test_grows_for_added_token_ids_without_losing_weights():
    model = _TinyCausalLM(vocab_size=8)
    with torch.no_grad():
        model.input_embeddings.weight.fill_(1.5)
        model.output_embeddings.weight.fill_(-0.75)

    result = resize_token_embeddings_if_needed(model, _Tokenizer(10))

    assert result == 10
    assert model.resize_calls == [10]
    assert model.input_embeddings.num_embeddings == 10
    assert model.output_embeddings.out_features == 10
    assert torch.all(model.input_embeddings.weight[:8] == 1.5)
    assert torch.all(model.output_embeddings.weight[:8] == -0.75)


def test_repairs_an_untied_output_head_that_is_too_small():
    model = _TinyCausalLM(vocab_size=8)
    model.output_embeddings = nn.Linear(3, 6, bias=False)

    result = resize_token_embeddings_if_needed(model, _Tokenizer(7))

    assert result == 8
    assert model.resize_calls == [8]
    assert model.input_embeddings.num_embeddings == 8
    assert model.output_embeddings.out_features == 8
