import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--model', type=str, default='parsenet', choices=['parsenet'])
    parser.add_argument('--imsize', type=int, default=512)
    parser.add_argument('--version', type=str, default='parsenet')

    # Training setting
    parser.add_argument('--total_step', type=int, default=500, help='how many times to update the generator')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=0.0002)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Testing setting
    parser.add_argument('--test_size', type=int, default=2824) 
    parser.add_argument('--model_name', type=str, default='5000_OPTIMAL_HP.pth') 

    # using pretrainedALL_
    parser.add_argument('--pretrained_model', type=int, default=None)

    ### USING DEEP LAB V3
    parser.add_argument('--dlv3', action='store_true', help='train using deep lab v3')


    parser.add_argument('--gpu_id', type=int, help='which gpu to use')

    # Misc
    parser.add_argument('--train', action='store_true', help='Enable training mode')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel training/ testing')
    parser.add_argument('--pickle_in', action='store_true', help='load data from pickle file')

    parser.add_argument('--split_face', action='store_true', help='split face in half for training and testing')
    parser.add_argument('--hp_tune', action='store_true', help='hp tuning on off')


    # parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--get_sam_iris', action='store_true', help='Use Segment Anything Model to find iris masks')

    parser.add_argument('--finetune', action='store_true', help='Finetune model using cfd data')

    # Trainign data path (celeb data)
    parser.add_argument('--img_path', type=str, default='./Data_preprocessing/cfd_images')
    parser.add_argument('--label_path', type=str, default='./Data_preprocessing/cfd_annotations') 
    
    # Whether or not to train with a subset of the iamges
    parser.add_argument('--train_limit', type=int, default=None, help='Limit the number of images used for training')

    # Step size 
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=float, default=100.0)

    
    # Paths for saving
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models/parsenet')
    parser.add_argument('--sample_path', type=str, default='./samples')

    #which dataset to test on
    parser.add_argument('--dataset', type=str, default='celeb', choices=['celeb', 'ted_long', 'ted', 'md', 'cfd'], help='Choose dataset')

    args = parser.parse_args()
    
    if args.dataset == 'ted':
        args.test_image_path = './Data_preprocessing/ted_images'
        args.test_label_path_gt = './Data_preprocessing/ted_annotations'
        args.test_label_path = './test_results_TED'
        args.test_color_label_path = './test_color_visualize_TED'
        args.csv_path = './csvs/2024_ALL_TED_key_SIZED.csv'
    elif args.dataset == 'md':
        args.test_image_path = './Data_preprocessing/md_images'
        args.test_label_path_gt = './Data_preprocessing/md_annotations'
        args.test_label_path = './test_results_MD'
        args.test_color_label_path = './test_color_visualize_MD'
        args.csv_path = './csvs/2024_ALL_MD_key_SIZED.csv'
    elif args.dataset == 'cfd':
        args.test_image_path = './Data_preprocessing/cfd_images'
        args.test_label_path_gt = './Data_preprocessing/cfd_annotations'
        args.test_label_path = './test_results_CFD'
        args.test_color_label_path = './test_color_visualize_CFD'
        args.csv_path = './csvs/2024_CFD_key_SIZED.csv'
    
    elif args.dataset == 'ted_long':

        args.test_image_path = 'Data_preprocessing/ted_tepezza_long'
        # args.test_label_path_gt = './Data_preprocessing/cfd_annotations'
        # args.test_label_path = './test_results_CFD'
        # args.test_color_label_path = './test_color_visualize_CFD'
        args.csv_path = './csvs/2024_TEPEZZA_TED_LONGITUDINAL_key.csv'

    else:
        args.test_image_path = './Data_preprocessing/test_img'
        args.test_label_path_gt = './Data_preprocessing/test_label'
        args.test_label_path = './test_results'
        args.test_color_label_path = './test_color_visualize'    
        args.csv_path = None


    return args
