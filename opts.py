import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    # Location of data
    parser.add_argument('--ann-file', type=str, default='data/annotation.json',
                        help='annotation file of the dataset')
    parser.add_argument('--pair-file', type=str, default='data/select_data.json',
                        help='annotation file of the dataset')
    parser.add_argument('--image-dir', type=str, default='../celebA/img_align_celeba/',
                        help='directory of the images')
 

    parser.add_argument('--save', type=str, default='../results/',
                        help='path to folder where to save the final model and log files and corpus')
    parser.add_argument('--save-every', type=int, default=1,
                        help='Save the model every x epochs')
    parser.add_argument('--eval-step', type=int, default=5,
                        help='Computing evaludation loss every x epochs')
    parser.add_argument('--eval-batch', type=int, default=None,
                        help='Evaluating x batchs randomly each time')
    parser.add_argument('--clean', dest='clean', action='store_true',
                        help='Delete the models and the log files in the folder')
    
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout for CNN layers')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='SGD weight decay')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='starting epoch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper epoch limit')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--resume-path', type=str, default=None,
                        help='the path of model to resume')
    parser.add_argument('--nthreads', type=int, default=16,
                        help='number of threads to load data')
    
    parser.add_argument('--alpha', type=float, default=1,
                        help='coefficient of reconstruction loss')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='coefficient of classify loss')
    parser.add_argument('--att-num', type=int, default=6,
                        help='attribute number to classify')
    
    args = parser.parse_args()
    
    return args