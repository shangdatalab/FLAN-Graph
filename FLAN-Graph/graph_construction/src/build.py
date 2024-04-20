from tree_builder import buildGraphs
from utils import *
from dgl.data.utils import save_graphs


if __name__ == '__main__':
    # args
    args = parse_args()
    config_file_path = args.config
    config = load_config(config_file_path)
    print(args)
    # build train graphs
    claims_list_transformed_train = data_prep(
                config["train_csv_data"]
            )
    train_graph, train_claim_level_infos = buildGraphs(claims_list_transformed_train, root2child=args.root2child)
    validation_claim_info_dicts = {
            "app_num": torch.stack([a[0] for a in train_claim_level_infos]),\
            "feat_encoding": torch.stack([a[1] for a in train_claim_level_infos]),\
            "original_claim_idx": torch.stack([a[2] for a in train_claim_level_infos]),\
            "label": torch.stack([a[3] for a in train_claim_level_infos])
            } # has to be in the format of (str:Tensor)
    save_graphs(config["train_data_graph_path"], train_graph, labels=validation_claim_info_dicts)

    # build validation graphs
    claims_list_transformed_validation = data_prep(
                config["validation_csv_data"]
            )
    validation_graph, validation_claim_level_infos = buildGraphs(claims_list_transformed_validation, root2child=args.root2child)
    validation_claim_info_dicts = {
            "app_num": torch.stack([a[0] for a in validation_claim_level_infos]),\
            "feat_encoding": torch.stack([a[1] for a in validation_claim_level_infos]),\
            "original_claim_idx": torch.stack([a[2] for a in validation_claim_level_infos]),\
            "label": torch.stack([a[3] for a in validation_claim_level_infos])
            } # has to be in the format of (str:Tensor)
    save_graphs(config["validation_data_graph_path"], validation_graph, labels=validation_claim_info_dicts)

    # build test graphs
    claims_list_transformed_test = data_prep(
                config["test_csv_data"]
            )
    test_graph, test_claim_level_infos = buildGraphs(claims_list_transformed_test, root2child=args.root2child)
    test_claim_info_dicts = {
            "app_num": torch.stack([a[0] for a in test_claim_level_infos]),\
            "feat_encoding": torch.stack([a[1] for a in test_claim_level_infos]),\
            "original_claim_idx": torch.stack([a[2] for a in test_claim_level_infos]),\
            "label": torch.stack([a[3] for a in test_claim_level_infos])
            } # has to be in the format of (str:Tensor)
    save_graphs(config["test_data_graph_path"], test_graph, labels=test_claim_info_dicts)