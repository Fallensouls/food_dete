import argparse
from mmdet.apis import init_detector, inference_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img', help='image file')
    args = parser.parse_args()
    return args


def visualize(model, img):
    result = inference_detector(model, img)
    # visualize the results in a new window
    model.show_result(img, result, show=True)


def main():
    args = parse_args()

    if args.config and args.checkpoint:
        model = init_detector(args.config, args.checkpoint, device='cuda:0')
    else:
        raise ValueError()

    if args.img:
        visualize(model, args.img)
    else:
        raise ValueError()


if __name__ == '__main__':
    main()
