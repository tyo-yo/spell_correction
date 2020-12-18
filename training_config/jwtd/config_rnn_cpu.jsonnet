local hidden_dim = 64;
local max_len = 64;
local num_layers = 1;
local dropout = 0.0;
local bidirectional = true;
local batch_size = 32;

{
  "dataset_reader": {
    "type": "seq2seq",
    "source_tokenizer": {
        "type": "spacy",
        "language": "ja_core_news_sm"
    },
    "target_tokenizer": {
        "type": "spacy",
        "language": "ja_core_news_sm"
    },
    "source_max_tokens": max_len,
    "target_max_tokens": max_len,
    // sudachiのtokenizeによりデフォルトのstart_symbolの@start@が @, start, @ になってしまうための設定
    "start_symbol": "STARTSYMBOL",
    "end_symbol": "ENDSYMBOL",
  },
  "train_data_path": "https://storage.googleapis.com/tyoyo/jwtd/v1.0/train.tsv",
  "validation_data_path": "https://storage.googleapis.com/tyoyo/jwtd/v1.0/dev.tsv",
  "model": {
    "type": "composed_seq2seq",
    "source_text_embedder": {
      "type": "basic",
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": hidden_dim
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": hidden_dim,
      "hidden_size": hidden_dim,
      "num_layers": num_layers,
      "bidirectional": bidirectional
    },
    "decoder": {
      "type": "auto_regressive_seq_decoder",
      "max_decoding_steps": max_len,
      "target_embedder": {
         "embedding_dim": hidden_dim
      },
      "target_namespace": "target_tokens",
      "tie_output_embedding": true,
      "beam_size": 4,
      "tensor_based_metric": {
        "type": "bleu"
      },
      "token_based_metric": null,
      /* "label_smoothing_ratio": 0.1, */
      "decoder_net": {
        "type": "lstm_cell",
        "decoding_dim": hidden_dim,
        "target_embedding_dim": hidden_dim,
      }
    },
  },
  "trainer": {
    "num_epochs": 10,
    "patience": 2,
    "cuda_device": -1,
    "num_serialized_models_to_keep": 1,
    "grad_norm": 5.0,
    "grad_clipping": null,
    "validation_metric": "+BLEU",
    "summary_interval": 100, # 100
    "histogram_interval": 10000, # null
    "should_log_learning_rate": true,
    "optimizer": {
      "type": "adam",
      "eps": 1e-9,
      "betas": [
        0.9,
        0.98
      ],
      "weight_decay": 0.01 # 0.01
    },
    // "learning_rate_scheduler": {
    //   "type": "noam",
    //   "factor": 1,
    //   "model_size": hidden_dim,
    //   "warmup_steps": 4000
    // }
  },
  "data_loader": {
      "type": "pytorch_dataloader",
      "batch_size": batch_size,
      "sampler": {
        "type": "bucket",
        "sorting_keys": [["source_tokens", "num_tokens"]],
      },
  },
//   "vocabulary": {
//     "directory_path": "data/vocabulary",
//     "extend": true
//   }
}
