
from CustomConfigs import RankModelConfig
import os
import pickle
from torchvision import models
from pre_process import Model_Obj_Feat
from pre_process import DataGenerator
from pre_process.PreProcNet import PreProcNet
from pre_process.Dataset import Dataset


DATASET_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/"   # Change to your location
PRE_PROC_DATA_ROOT = "/home/zaimaz/Desktop/research1/QAGNet/Dataset/IRSR_ASSR/asd_extra/"    # Change to your location


if __name__ == '__main__':
    # add pre-trained weight path - backbone pre-trained on salient objects (binary, no rank)

    # Run Script twice to generate pre-processed object features of GT objects for "train" and "val" data_splits
    # data_split = "train"
    data_split = "train"

    out_path = PRE_PROC_DATA_ROOT + "pre_process_feat/" + data_split + "/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    mode = "train"
    config = RankModelConfig()
    log_path = "logs/"

    keras_model = Model_Obj_Feat.build_obj_feat_model(config)
    model_name = "Obj_Feat_Net"

    # model = PreProcNet(mode=mode, config=config, model_dir=log_path, keras_model=keras_model, model_name=model_name)
    model = models.resnet152(weights='IMAGENET1K_V1')

    if mode == "train":
        # ********** Create Datasets
        # Train/Val Dataset
        dataset = Dataset(DATASET_ROOT, data_split)

        predictions = []

        num = len(dataset.img_ids)
        for i in range(num):

            image_id = dataset.img_ids[i]
            print(i + 1, " / ", num, " - ", image_id)

            input_data, gt_ranks, sel_not_sal_obj_idx_list, shuffled_indices, chosen_obj_idx_order_list = DataGenerator.load_inference_data_obj_feat_gt(dataset, image_id, config)

            result = model.detect(input_data, verbose=1)
            result["gt_ranks"] = gt_ranks
            result["sel_not_sal_obj_idx_list"] = sel_not_sal_obj_idx_list
            result["shuffled_indices"] = shuffled_indices
            result["chosen_obj_idx_order_list"] = chosen_obj_idx_order_list

            o_p = out_path + image_id
            with open(o_p, "wb") as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


