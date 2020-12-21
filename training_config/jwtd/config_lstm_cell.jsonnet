local cuda_device = 0;

local hidden_dim = 500;
local max_len = 64;
local num_layers = 2;
local dropout = 0.1;
local bidirectional = true;
local batch_size = 50;


{
  "dataset_reader": {
    "type": "seq2seq",
    "lazy": true,
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
    "tied_source_embedder_key": "tokens",
    "encoder": {
      "type": "lstm",
      "input_size": hidden_dim,
      "hidden_size": hidden_dim / 2,
      "num_layers": num_layers,
      "bidirectional": bidirectional
    },
    "decoder": {
      "type": "auto_regressive_seq_decoder",
      "max_decoding_steps": max_len,
      "target_embedder": {
         "embedding_dim": hidden_dim
      },
      "target_namespace": "tokens",
      "tie_output_embedding": true,
      "beam_size": 4,
      "tensor_based_metric": {
        "type": "bleu"
      },
      "token_based_metric": {
        "type": "levenshtein_distance",
        "avg_by_seq_len": false
      },
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
    "patience": 5,
    "cuda_device": cuda_device,
    "grad_norm": 5.0,
    "grad_clipping": null,
    "validation_metric": "-LevenshteinDistance",
    "optimizer": {
      "type": "adam",
      "lr": 1e-3,
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 2,
    },
    "tensorboard_writer": {
        "summary_interval": 100, # 100
        "histogram_interval": 10000, # null
        "should_log_learning_rate": true,
    },
    "trainer_callbacks": [
      {
        "type": "log_to_comet",
        "project_name": "jwtd",
        "upload_serialization_dir": true,
        "log_interval": 100,
        "log_batch_output": true,
        "send_notification": true
      },
    ],
    // "learning_rate_scheduler": {
    //   "type": "noam",
    //   "factor": 1,
    //   "model_size": hidden_dim,
    //   "warmup_steps": 4000
    // }
  },
  "data_loader": {
      "type": "pytorch_dataloader",
      // "batch_sampler": {
      //   "type": "bucket",
      //   "batch_size": batch_size,
      //   "sorting_keys": ["source_tokens", "target_tokens"],
      // },
      "num_workers": 2,
  },
  "vocabulary": {
    "type": "from_files",
    "directory": "experiments/jwtd/0002_lstm_cell/vocabulary"
    // "extend": true
  }
}
