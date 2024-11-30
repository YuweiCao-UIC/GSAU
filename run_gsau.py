import argparse
from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon-beauty', help='dataset')
    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--graph_encoder', type=str, default='LightGCN', help='graph encoder')
    parser.add_argument('--sequential_encoder', type=str, default='SASRec', help='sequential encoder')
    parser.add_argument('--enable_graph_encoder', default = False, action="store_true", help = 'enable graph_encoder')
    parser.add_argument('--enable_sequential_encoder', default = False, action="store_true", help = 'enable sequential_encoder')
    parser.add_argument('--predict_with_graph_encoder', default = False, action="store_true", help = 'predict with graph encoder instead of sequential encoder')
    parser.add_argument('--align_per_layer', default = False, action="store_true", help = 'enable per-layer alignment for the graph encoder')
    parser.add_argument('--rec', default = False, action="store_true", help = 'replace the alignment loss of the sequential encoder with rec (CE) loss')
    parser.add_argument('--enable_u_i_uniformity', default = False, action="store_true", help = 'enable (user, negative item) uniformity loss')
    parser.add_argument('--config_files', type=str, default='./gsau.yaml', help='config files that set other hyperparameters')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    # parameter dictionaries > config file
    config_dict = {'dataset': args.dataset, 'epochs': args.epochs, \
        'gamma': args.gamma, \
        'graph_encoder': args.graph_encoder, 'sequential_encoder': args.sequential_encoder, \
        'enable_graph_encoder': args.enable_graph_encoder, \
        'enable_sequential_encoder': args.enable_sequential_encoder, \
        'predict_with_graph_encoder': args.predict_with_graph_encoder, \
        'align_per_layer': args.align_per_layer, \
        'rec': args.rec, \
        'enable_u_i_uniformity': args.enable_u_i_uniformity}
    
    run_recbole(model='GSAU', dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict)