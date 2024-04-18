
def group_edge_type(edge_type):
    d = {
        1:0,
        2:1,
        3:2,

        # Beilei: Missing default for nl->l and l -> l
        4:8,
        5:9,


        10:3,
        11:3,
        12:3,
        13:3,
        14:3,
        15:3,
        16:3,
        17:3,
        18:3,

        20:4,
        21:4,
        22:4,

        30:5,

        40:6,
        41:6,

        50:7
    }
    return d[edge_type]
    # return 0