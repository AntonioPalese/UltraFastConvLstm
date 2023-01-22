**Exaplination of the `configurations.yaml` file**

model: 

    `row_anchors` : list of row-anchors (in y coordinates). List.
    `griding_num` : number of griding cell for each row. Int.
    `num_lanes` : number of possible lane lines in each image. Int.
    `num_layers` : number of the Conv-Lstm layers in cascade. Int.
    `hidden_sizes` : hidden dimension of the Conv-Lstm layer. List.
    `kernel_sizes` : kernel size of the convolution in the Conv-Lstm. List
    `backbone` :  backbone to use. String.
    `size` : Shape (H,W) of the resized images. List
    `seq_len` : temporal sequence length. Int
    `cls_dim` : [griding_num + 1, row_anchors length, num_lanes]. List
    `bias` : use bias or not. Bool.
    `device` : GPU device number to use. Int

train : 

    `dataset` : dataset name. String.
    `batch_size` : batch size. Int.
    `path` : root path of the dataset. String.
    `optimizer` : optimizer name (SGD, Adam). String.
    `use_aux` : useless
    `epoch` : number of epochs to train. Int.
    `list_path` : path of the metadata file  `train_gt.txt` of CuLane. String.
    `sim_loss_w` : coefficient of use of the similitude loss function. Double [0-1]
    `shp_loss_w` : coefficient of use of the shape loss function. Double [0-1]
    `scheduler` : scheduler type (multi, cos). String.
    `steps` : range of epochs for modifing the learning_rate. List.
    `gamma` : gamma value used in the scheduler. Int.
    `warmup` : kind of warmup (linear). String.
    `warmup_iters` : number of warmup steps. Int.
    `weight_decay` : weight decay used in SGD. Double.
    `learning_rate` : learning rate. Double.
    `momentum` : momentum value used in Adam. Double.
    `weights` : the weight for the griding cells and the weight for the no-line cell. List or null. Default null.

test: 

    `list_path` : path of the metadata file  `val_gt.txt` of CuLane. String.
    `batch_size` : batch size for the test. Int

multipletest:

    `list_path`: path of the metadata folder  `test_split` of CuLane for the environmental test. String.
    `batch_size` : batch size for the test. Int

save:

    `weights_path`: weight path to load for testing. String.
    `overwrite` : overwrite or not the weights during the saving. Bool.
    `keep_weights` : path of the weight where restart the training, otherwise empty string. String
    `resume_epoch` : epoch number where restart the training, only if `keep_weights` is not empty string. Int.
    `note` : 'note to add to the weights folder name'
