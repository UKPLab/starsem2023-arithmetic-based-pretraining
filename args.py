
def add_args(parser):

    #
    # Model specific parameters
    #

    parser.add_argument("--checkpoint_model", default=None, type=str,
        help="A model pretrained using the INP task that should now be finetuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
            required=True, help="The model identifier from huggingface.co/models")
    parser.add_argument(
        "--config_name", default="", type=str, 
        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
        help="Where do you want to store the pre-trained models downloaded from s3?")        
    parser.add_argument("--tokenizer_name", default="", type=str,
        help="Pretrained tokenizer name or path if not the same as model_name")            
    parser.add_argument("--max_source_length", default=384, type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=384, type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")        

    #
    # Run mode
    #

    parser.add_argument("--do_train", action="store_true", 
        help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", 
        help="Whether to run predictions on the test set.")

    #
    # Training specific parameters
    #

    parser.add_argument("--num_train_epochs", default=1, type=int, 
        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=8, type=int)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--val_check_interval", default=None,   
        help="Intermediate validation during an epoch. Pass an int for validation after n batches.")
    parser.add_argument("--limit_val_batches", default=None, help="Limit the batches for validation.")    
    parser.add_argument("--early_stopping_patience", type=int, default=-1,
        required=False, help="-1 means never early stop. early_stopping_patience is measured in validation checks, not epochs. So val_check_interval will effect it.")    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
        help="The output directory where the model predictions and checkpoints will be written.")       
    parser.add_argument("--data_dir", type=str, required=True, help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.")         

    #
    # Technical training parameters
    #

    parser.add_argument("--max_grad_norm", default=1.0, type=float, 
        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", type=int,
        default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")        
    parser.add_argument("--seed", type=int, default=42, 
        help="random seed for initialization")        
    parser.add_argument("--learning_rate", default=5e-5, type=float, 
        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, 
        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, 
        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, 
        help="Linear warmup over warmup_steps.")        

    #
    # Tasks
    #

    parser.add_argument("--masked_number_prediction", action="store_true", 
        help="Whether to run the inferable number prediction task.")
    parser.add_argument("--masked_number_prediction_contrastive",   
        action="store_true", 
        help="Whether to run the inferable number prediction task with contrastive loss.")
    parser.add_argument("--finetuning", action="store_true", 
        help="Whether to just run the finetuning task.")

    #
    # validation metric
    #
    # if you want to do [DROP, InfoTabs or masked number prediction task, / 
    # masked number]: em_score
    # else: mover_score
    # 

    parser.add_argument("--em_score", action="store_true", 
        help="Whether to use the em score as validation metric. This is usually necessary for the inferable number prediction task and for finetuning using DROP and InfoTabs.")
    parser.add_argument("--mover_score", action="store_true", 
        help="Whether to use the Moverscore as validation metric. This is usually necessary for finetuning using the table-to-text generation datasets")


    #
    # Number Representations
    #

    parser.add_argument("--scientific_notation", default=False, type=bool, 
        help="Whether to use the scientific notation for numbers.")
    parser.add_argument("--char_level_representation", default=False, help="Whether to use the char-level representation for numbers.", 
        type=bool)
        
    #
    # Contrastive Number Representation
    #

    parser.add_argument("--verbalized", default=False, type=bool, help="Use the verbalized representation for numbers with the contrastive loss.")
    parser.add_argument("--all", default=False, type=bool, help="Use the char-level representation, the scientific representation and the verbalized representation for numbers with the contrastive loss.")

    return parser
