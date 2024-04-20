from tree_builder import buildGraphs
from utils import *
from dgl.data.utils import save_graphs

if __name__ == '__main__':
    # args
    args = parse_args()

    # bulid the graphs
    claims_list_transformed = data_prep(args.csv_path)
    graphs, claim_level_infos = buildGraphs(claims_list_transformed, root2child=args.root2child)
    claim_info_dicts = {
            "app_num": torch.stack([a[0] for a in claim_level_infos]),\
            "feat_encoding": torch.stack([a[1] for a in claim_level_infos]),\
            "original_claim_idx": torch.stack([a[2] for a in claim_level_infos]),\
            "label": torch.stack([a[3] for a in claim_level_infos])
            } # has to be in the format of (str:Tensor)
    save_graphs(args.graph_path, graphs, labels=claim_info_dicts)