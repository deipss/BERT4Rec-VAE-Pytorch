def set_template(args):
    if args.template is None:
        return

    elif args.template.startswith('train_bert_cnn'):
        args.mode = 'train'
        args.stride = 2
        args.kernel_size = 6

        # args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.dataset_code = 'ml-1m'

        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 6
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 100 if args.dataset_code == 'ml-1m' else 500
        args.test_negative_sample_size = 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda'
        args.num_gpu = 2
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 10
        args.metric_ks = [1, 5, 10, 15, 20, 25, 30]
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert_cnn'

        args.model_init_seed = 0

        args.bert_dropout = 0.1
        args.bert_hidden_units = 256
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 4
        args.dim = args.bert_hidden_units

    elif args.template.startswith('train_bert'):
        args.mode = 'train'
        # args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.dataset_code = 'ml-1m'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 8 if not args.dataset_code.startswith('ml') else 128
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 500 if args.dataset_code == 'ml-20m' else 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda'
        args.num_gpu = 2
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 10 if args.dataset_code == 'ml-20m' else 100
        args.metric_ks = [1, 5, 10, 15, 20, 25, 30]
        args.best_metric = 'NDCG@10'

        args.model_code = 'bert'
        args.model_init_seed = 0

        args.bert_dropout = 0.1
        args.bert_hidden_units = 64
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.dim = args.bert_hidden_units

    elif args.template.startswith('train_pop'):
        args.mode = 'test50'
        # args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.dataset_code = 'ml-10m'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'bert'
        batch = 8 if not args.dataset_code.startswith('ml') else 128
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch if not args.mode == 'test50' else 7000

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 500 if args.dataset_code == 'ml-20m' else 100
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda'
        args.num_gpu = 2
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 10 if args.dataset_code == 'ml-20m' else 100
        args.metric_ks = [1, 5, 10, 15, 20, 25, 30]
        args.best_metric = 'NDCG@10'

        args.model_code = 'pop'
        args.model_init_seed = 0

        args.bert_dropout = 0.1
        args.bert_hidden_units = 64
        args.bert_mask_prob = 0.15
        args.bert_max_len = 100
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.dim = args.bert_hidden_units

    elif args.template.startswith('train_ncf'):
        args.mode = 'train'
        # args.dataset_code = 'ml-' + input('Input 1 for ml-1m, 20 for ml-20m: ') + 'm'
        args.dataset_code = 'ml-1m'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'leave_one_out'

        args.dataloader_code = 'ncf'
        batch = 8 if not args.dataset_code.startswith('ml') else 128
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.train_negative_sampler_code = 'random'
        args.train_negative_sample_size = 0
        args.train_negative_sampling_seed = 0
        args.test_negative_sampler_code = 'random'
        args.test_negative_sample_size = 500 if args.dataset_code == 'ml-20m' else 200
        args.test_negative_sampling_seed = 98765

        args.trainer_code = 'ncf'
        args.device = 'cuda'
        args.num_gpu = 2
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
        args.decay_step = 25
        args.gamma = 1.0
        args.num_epochs = 10 if args.dataset_code == 'ml-20m' else 100
        args.metric_ks = [1, 5, 10, 15, 20, 25, 30]
        args.best_metric = 'NDCG@10'

        args.model_code = 'ncf'
        args.model_init_seed = 0

        args.bert_dropout = 0.1
        args.bert_hidden_units = 64
        args.bert_mask_prob = 0.15
        args.bert_max_len = 200
        args.bert_num_blocks = 2
        args.bert_num_heads = 2
        args.dim = 8
        args.ncf_num_layers = 1
        args.ncf_dropout = 0.1
        args.ncf_model = 'MLP'


    elif args.template.startswith('train_dae'):
        args.mode = 'train'
        args.dataset_code = 'card'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-20m' else 100

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 256
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'dae'
        args.device = 'cuda'
        args.num_gpu = 2
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.00
        args.num_epochs = 10 if args.dataset_code == 'ml-20m' else 300
        args.metric_ks = [1, 5, 15, 10, 20, 25, 30]
        args.best_metric = 'NDCG@10'

        args.model_code = 'dae'
        args.model_init_seed = 0
        args.dae_num_hidden = 2
        args.dae_hidden_dim = 600
        args.dae_latent_dim = 200
        args.dae_dropout = 0.5
        args.dim = args.dae_latent_dim

    elif args.template.startswith('train_vae_search_beta'):
        args.mode = 'train'

        args.dataset_code = 'card'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 256 if args.dataset_code == 'ml-20m' else 100

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'vae'
        args.device = 'cuda'
        args.num_gpu = 2
        args.device_idx = '1'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.01
        args.num_epochs = 300 if not args.dataset_code == 'ml-20m' else 10
        args.metric_ks = [1, 5, 15, 10, 20, 25, 30]
        args.best_metric = 'NDCG@10'
        args.total_anneal_steps = 3000 if args.dataset_code == 'ml-1m' else 10000
        args.find_best_beta = True

        args.model_code = 'vae'
        args.model_init_seed = 0
        args.vae_num_hidden = 2
        args.vae_hidden_dim = 600
        args.vae_latent_dim = 200
        args.vae_dropout = 0.5
        args.dim = args.vae_latent_dim

    elif args.template.startswith('train_vae_give_beta'):
        args.mode = 'train'

        args.dataset_code = 'ml-20m'
        args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
        args.min_uc = 5
        args.min_sc = 0
        args.split = 'holdout'
        args.dataset_split_seed = 98765
        args.eval_set_size = 500 if args.dataset_code == 'ml-1m' else 10000

        args.dataloader_code = 'ae'
        batch = 128 if args.dataset_code == 'ml-1m' else 512
        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        args.trainer_code = 'vae'
        args.device = 'cuda'
        args.num_gpu = 1
        args.device_idx = '0'
        args.optimizer = 'Adam'
        args.lr = 1e-3
        args.enable_lr_schedule = False
        args.weight_decay = 0.01
        args.num_epochs = 100 if args.dataset_code == 'ml-1m' else 10
        args.metric_ks = [1, 5, 15, 10, 20, 25, 30]
        args.best_metric = 'NDCG@100'
        args.find_best_beta = False
        args.anneal_cap = 0.342
        args.total_anneal_steps = 3000 if args.dataset_code == 'ml-1m' else 20000

        args.model_code = 'vae'
        args.model_init_seed = 0
        args.vae_num_hidden = 2
        args.vae_hidden_dim = 600
        args.vae_latent_dim = 200
        args.vae_dropout = 0.5
        args.dim = args.vae_latent_dim
