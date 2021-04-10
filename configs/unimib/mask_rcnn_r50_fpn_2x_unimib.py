# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=73),
        mask_head=dict(num_classes=73)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('arancia', 'arrosto', 'arrosto_di_vitello', 'banane', 'bruscitt', 'budino', 'carote', 'cavolfiore', 'cibo_bianco_non_identificato', 'cotoletta', 'crema_zucca_e_fagioli', 'fagiolini', 'finocchi_gratinati', 'finocchi_in_umido', 'focaccia_bianca', 'guazzetto_di_calamari', 'insalata_2_(uova mais)', 'insalata_mista', 'lasagna_alla_bolognese', 'mandarini', 'medaglioni_di_carne', 'mele', 'merluzzo_alle_olive', 'minestra', 'minestra_lombarda', 'orecchiette_(ragu)', 'pane', 'passato_alla_piemontese', 'pasta_bianco', 'pasta_cozze_e_vongole', 'pasta_e_ceci', 'pasta_e_fagioli', 'pasta_mare_e_monti', 'pasta_pancetta_e_zucchine', 'pasta_pesto_besciamella_e_cornetti', 'pasta_ricotta_e_salsiccia',
           'pasta_sugo', 'pasta_sugo_pesce', 'pasta_sugo_vegetariano', 'pasta_tonno', 'pasta_tonno_e_piselli', 'pasta_zafferano_e_piselli', 'patate_pure', 'patate_pure_prosciutto', 'patatine_fritte', 'pere', 'pesce_(filetto)', 'pesce_2_(filetto)', 'piselli', 'pizza', 'pizzoccheri', 'polpette_di_carne', 'riso_bianco', 'riso_sugo', 'roastbeef', 'rosbeef', 'rucola', 'salmone_(da_menu_sembra_spada_in_realta)', 'scaloppine', 'spinaci', 'stinco_di_maiale', 'strudel', 'torta_ananas', 'torta_cioccolato_e_pere', 'torta_crema', 'torta_crema_2', 'torta_salata_(alla_valdostana)', 'torta_salata_3', 'torta_salata_rustica_(zucchine)', 'torta_salata_spinaci_e_ricotta', 'yogurt', 'zucchine_impanate', 'zucchine_umido')

data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/UNIMIB2016/train/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/UNIMIB2016/unimib_train_coco_format.json'),
    val=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/UNIMIB2016/test/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/UNIMIB2016/unimib_test_coco_format.json'),
    test=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/UNIMIB2016/test/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/UNIMIB2016/unimib_test_coco_format.json'))

checkpoint_config = dict(interval=4)
workflow = [('train', 1), ('val', 1)]
evaluation = dict(metric=['bbox', 'segm'], proposal_nums=(1, 10, 100))
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
