import argparse

def init_parameters():
    parser = argparse.ArgumentParser()

    #数据加载等相关参数
    parser.add_argument("--data_dir",default="",type=str,
                        help="The inputs dir")
    parser.add_argument("--dataset",type=str)

    parser.add_argument("--pretrain_dir", default='', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    #Bert模型相关超参数
    parser.add_argument("--pre_bert_model",default="bert-base-uncased",type=str,
                        help="general pretrain model dir")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer, than this will be truncated, sequences shorter will be padded." )
    parser.add_argument("--feat_dim",default=768,type=int,
                        help="The input's dim")
    parser.add_argument("--freeze_bert_parameters",default=False,type=bool,
                        help="Don't update the bert's parameters")
    #模型训练过程需要提供的参数
    parser.add_argument("--warmup_proportion",default=0.1,type=float,
                    )
    parser.add_argument("--total_para",default=True,type=bool)
    parser.add_argument("--known_cls_ratio", type=float,
                        help="Split the data into known and unknown classes according to a certain ratio. This parameter specifies the proportion of known classes.")
    parser.add_argument("--labels_ratio", default=1.0, type=float,
                        help="Proportion of randomly selected training data.")
    parser.add_argument("--method", default="DASM_OSR", type=str,
                        help="Which margin-based method to use")
    parser.add_argument("--seed", default=40, type=int,
                        help="Set random generator seed to ensure experiment reproducibility")
    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")
    parser.add_argument("--lr", default=3e-5, type=float,
                        help="Parameter update step size")
    parser.add_argument("--epochs", default=35, type=int,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Training data batch size")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Validation/testing data batch size")
    parser.add_argument("--wait_patient", default=15, type=int,
                        help="If model performance does not improve for a specified number of epochs, training is terminated early")
    parser.add_argument("--lr_boundary", default=0.08, type=float,
                        help="Learning rate for training boundary values")

    # Save model parameters and output results
    parser.add_argument("--save_model_parameters_dir", default="", type=str,
                        help="Path to save the best model parameters")
    parser.add_argument("--save_model", default=False, type=bool,
                        help="Whether to save the model")
    parser.add_argument("--save_center_boundaries_dir", default="/", type=str,
                        help="Path to save the best class centers and boundary parameters")
    parser.add_argument("--output_file", default="t", type=str,
                        help="Evaluation results output by the best model")
    return parser



