local cuda_device = -1;

local hidden_dim = 500;
local max_len = 100;
local num_layers = 2;
local dropout = 0.1;
local bidirectional = true;
local batch_size = 50;


{
  "dataset_reader": {
    "type": "seq2seq",
    "lazy": false,
    "source_tokenizer": {
        "type": "mecab",
        "special_tokens": ["@start@", "@end@"],
    },
    "target_tokenizer": {
        "type": "mecab",
        "special_tokens": ["@start@", "@end@"],
    },
    "source_max_tokens": max_len,
    "target_max_tokens": max_len,
  },
  "train_data_path": "https://storage.googleapis.com/tyoyo/jwtd/v1.0/train.tsv",
  "validation_data_path": "https://storage.googleapis.com/tyoyo/jwtd/v1.0/dev.tsv",
  "data_loader": {
      "type": "pytorch_dataloader",
      "batch_sampler": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": ["source_tokens", "target_tokens"],
      },
      "num_workers": 16,
  },
  "vocabulary": {
      "max_vocab_size": 10000
    // "type": "from_files",
    // "directory": "experiments/jwtd/vocab/mecab"
  }
}
