local cuda_device = -1;

local hidden_dim = 8;
local max_len = 64;
local num_encoder_layers = 1;
local num_decoder_layers = 1;
local dropout = 0.0;
local bidirectional = true;
local batch_size = 2;


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
  "train_data_path": "tests/fixtures/jwtd_sample.tsv",
  "validation_data_path": "tests/fixtures/jwtd_sample.tsv",
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
      "num_layers": num_encoder_layers,
      "bias": true,
      "dropout": dropout,
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
        "type": "lstm",
        "decoding_dim": hidden_dim,
        "target_embedding_dim": hidden_dim,
        "num_layers": num_decoder_layers,
        "bias": true,
        "dropout": dropout,
        "bidirectional_input": bidirectional,
        "attention": {
          "type": "dot_product"
        },
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
      "eps": 1e-9,
      "betas": [
        0.9,
        0.98
      ],
      "weight_decay": 0.01,
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
        "project_name": "jwtd_test",
        "upload_serialization_dir": true
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
      "batch_sampler": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": ["source_tokens", "target_tokens"],
      },
      "num_workers": 2,
  },
//   "vocabulary": {
//     "directory_path": "data/vocabulary",
//     "extend": true
//   }
}
