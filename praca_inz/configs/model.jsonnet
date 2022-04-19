{   
    "sfnet":{
        "embedding_dim": 20,

        "num_item_emb" : 1378,
        "num_category_emb" : 7,
        "num_cup_size_emb" : 12,
        "num_user_emb" : 47958,

        "num_user_numeric": 5, // number of user features which are numeric
        "num_item_numeric": 2, // number of item features which are numeric

        "user_pathway": [256, 128, 64], //series of transformations for the use embeddings + features
        "item_pathway": [256, 128, 64], //series of transformations for the item embeddings + features
        "combined_pathway": [256, 128, 64, 16], //series of transformations for the item embeddings + features

        "activation": "relu", // relu or tanh
        "dropout": 0.03859220339148282, 

        "num_targets": 3, // small, fit or large
    },
    "trainer": {
        "num_epochs": 20,
        "batch_size": 128,
        "optimizer": {
          "lr": 0.0013848202147112318,
          "type": "adam",
          "weight_decay": 0.00013553953774461246,
        }
    },
    "logging": {
        "save_model_path": "runs/",
        "run_name": "trial_",
        "tensorboard": true,
        "print_every": 10
    }
}