design_conf = {
    'model_list': ['social', 'rating', 'twin'],

    'distill_dict': {
        ('social', 'rating'): 1.0,
        ('rating', 'social'): 1.0,
        ('social', 'twin'): 1.0,
        ('twin', 'social'): 1.0,
        ('rating', 'twin'): 1.0,
        ('twin', 'rating'): 1.0,
    },
    'l_user': [1, 2],
    'l_item': [0],
}