# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=201),
        mask_head=dict(num_classes=201)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('apple pie', 'arugula', 'asparagus', 'au jus', 'avocado', 'baby back ribs', 'bacon', 'baked potato', 'baklava', 'basil', 'bean sprouts', 'beans', 'beef', 'beef carpaccio', 'beef tartare', 'beer', 'beet salad', 'beignets', 'berries', 'bibimbap', 'black beans', 'blackberries', 'blueberries', 'bread', 'bread pudding', 'breakfast burrito', 'broccoli', 'bruschetta', 'bun', 'butter', 'cabbage', 'caesar salad', 'cake', 'cannoli', 'caprese salad', 'carrot cake', 'carrots', 'celery', 'ceviche', 'cheese', 'cheese plate', 'cheeseburger', 'cheesecake', 'chicken', 'chicken curry', 'chicken quesadilla', 'chicken wings', 'chives', 'chocolate cake', 'chocolate mousse', 'chocolate sauce', 'churros', 'cilantro', 'clam chowder', 'club sandwich', 'cocktail sauce', 'coffee', 'coleslaw', 'corn', 'crab cakes', 'crackers', 'cream', 'creme brulee', 'croque madame', 'croutons', 'cucumber slice', 'cup cakes', 'deviled eggs', 'dipping sauce', 'donuts', 'dumplings', 'edamame', 'eggs', 'eggs benedict', 'escargots', 'falafel', 'filet mignon', 'fish', 'fish and chips', 'foie gras', 'french fries', 'french onion soup', 'french toast', 'fried calamari', 'fried egg', 'fried rice', 'frosting', 'frozen yogurt', 'garlic bread', 'garnish', 'ginger', 'gnocchi', 'grapes', 'gravy', 'greek salad', 'green beans', 'green onions', 'greens', 'grilled cheese sandwich',
           'grilled salmon', 'guacamole', 'gyoza', 'ham', 'hamburger', 'hash browns', 'home fries', 'hot and sour soup', 'hot dog', 'huevos rancheros', 'hummus', 'ice cream', 'ketchup', 'lasagna', 'lemon slice', 'lettuce', 'lime slice', 'lobster bisque', 'lobster roll sandwich', 'macaroni and cheese', 'macarons', 'mashed potatoes', 'mint', 'miso soup', 'mixed fruit', 'mixed greens', 'mixed vegetables', 'mushrooms', 'mussels', 'mustard', 'nachos', 'noodles', 'nuts', 'olives', 'omelette', 'onion rings', 'onions', 'orange slice', 'oysters', 'pad thai', 'paella', 'pancakes', 'panna cotta', 'parmesan cheese', 'parsley', 'peas', 'peking duck', 'peppers', 'pho', 'pickles', 'pita bread', 'pizza', 'pork chop', 'potato chips', 'potatoes', 'poutine', 'powdered sugar', 'prime rib', 'pulled pork sandwich', 'ramen', 'raspberries', 'ravioli', 'red velvet cake', 'rice', 'risotto', 'salad', 'salsa', 'samosa', 'sashimi', 'sauce', 'sausage', 'scallops', 'seaweed salad', 'shrimp', 'shrimp and grits', 'soup', 'sour cream', 'soy sauce', 'spaghetti bolognese', 'spaghetti carbonara', 'spinach', 'spring rolls', 'steak', 'strawberries', 'strawberry shortcake', 'sushi', 'syrup', 'tacos', 'takoyaki', 'tartar sauce', 'tiramisu', 'toast', 'tomato sauce', 'tomato slice', 'tortilla', 'tortilla chips', 'tuna tartare', 'waffles', 'wasabi', 'whipped cream', 'wine', 'chocolate')
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=4,  # Worker to pre-fetch data for each single GPU
    train=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/food201/segmented_train/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/food201/food201_train_coco_format.json'),
    val=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/food201/segmented_test/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/food201/food201_test_coco_format.json'),
    test=dict(
        img_prefix='/home/hatsunemiku/dev/food_dete/data/food201/segmented_test/',
        classes=classes,
        ann_file='/home/hatsunemiku/dev/food_dete/data/food201/food201_test_coco_format.json'))

checkpoint_config = dict(interval=4)
workflow = [('train', 1), ('val', 1)]
evaluation = dict(metric=['bbox', 'segm'], proposal_nums=(1, 10, 100))
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
