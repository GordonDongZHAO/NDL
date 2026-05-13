import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="NDL: Noise-aware Dimension Learning")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="yelp",
                        help="Choose a dataset: [yelp, book, movielens]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== #
    parser.add_argument("--gnn", nargs="?", default="lightgcn",
                        help="Choose a recommender: [lightgcn]")
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=32,
                        help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False,
                        help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1,
                        help="message dropout ratio")
    parser.add_argument("--edge_dropout", type=bool, default=False,
                        help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1,
                        help="edge dropout ratio")
    parser.add_argument("--batch_test_flag", type=bool, default=True,
                        help="evaluate users in mini-batches")
    parser.add_argument("--pool", type=str, default='mean',
                        help="[concat, mean, sum, final]")
    parser.add_argument("--simi", type=str, default='ip', help="[ip, cos, ed]")
    parser.add_argument("--context_hops", type=int, default=2, help="number of LightGCN layers")
    parser.add_argument("--K", type=int, default=1,
                        help="number of negative samples")
    parser.add_argument("--n_negs", type=int, default=1,
                        help="number of candidate negatives")

    # ===== NDL: noise estimation parameters ===== #
    parser.add_argument("--tau_coh", type=float, default=0.5,
                        help="Temperature for coherence-based noise estimation")
    parser.add_argument("--tau_loss", type=float, default=10.0,
                        help="Temperature for loss-based noise estimation "
                             "(Yelp: 10, Book: 5, MovieLens: 5)")
    parser.add_argument("--ema_decay", type=float, default=0.9,
                        help="EMA decay for loss smoothing "
                             "(Yelp: 0.9, Book: 0.5, MovieLens: 0.9)")
    parser.add_argument("--min_activity", type=float, default=30.0,
                        help="Users with at least this many interactions fully trust "
                             "coherence; colder users rely more on the loss signal")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Standard training epochs before enabling denoising")

    # ===== contrastive learning ===== #
    parser.add_argument("--cl_lambda", type=float, default=0.1,
                        help="Contrastive loss weight; set to 0 to disable CL")
    parser.add_argument("--cl_temp", type=float, default=0.2,
                        help="InfoNCE temperature")
    parser.add_argument("--cl_drop", type=float, default=0.1,
                        help="Dropout rate for CL augmentation")
    parser.add_argument("--cl_start_epoch", type=int, default=-1,
                        help="Epoch to start CL; default -1 starts after warmup_epochs")

    # ===== general ===== #
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Disable CUDA even if available")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    parser.add_argument('--Ks', nargs='?', default='[5, 10, 20, 50, 100]',
                        help='Top-K values for evaluation')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}')

    # ===== save model ===== #
    parser.add_argument("--save", type=bool, default=False, help="save model or not")
    parser.add_argument(
        "--out_dir", type=str, default="./weights/", help="output directory for model"
    )

    args = parser.parse_args()

    import torch as _torch
    args.cuda = (not args.no_cuda) and _torch.cuda.is_available()
    return args
