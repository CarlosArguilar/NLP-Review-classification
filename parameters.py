# default parameters
params = {
    'preprocessing':{
        'max_length_truncate':800,
    },
    'model':{
        'max_tokens':500,
        'epochs':7,
        'validation_split':0.2,
        'emb_dim':16,
        'lstm_cells':128,
        'dense_neurons':300,
        'dropout_rate':0.5
    }
}