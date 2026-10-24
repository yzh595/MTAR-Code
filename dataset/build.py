from dataset.imagenet import build_imagenet, build_imagenet_code


def build_dataset(args, **kwargs):
    # images
    if args.dataset == 'imagenet':
        return build_imagenet(args, **kwargs)
    if args.dataset == 'imagenet_code':
        return build_imagenet_code(args, **kwargs)
    if args.dataset == 'coco':
        from dataset.coco import build_coco
        return build_coco(args, **kwargs)
    if args.dataset == 'openimage':
        from dataset.openimage import build_openimage
        return build_openimage(args, **kwargs)
    if args.dataset == 'pexels':
        from dataset.pexels import build_pexels
        return build_pexels(args, **kwargs)
    if args.dataset == 't2i_image':
        from dataset.t2i import build_t2i_image
        return build_t2i_image(args, **kwargs)
    if args.dataset == 't2i':
        from dataset.t2i import build_t2i
        return build_t2i(args, **kwargs)
    if args.dataset == 't2i_code':
        from dataset.t2i import build_t2i_code
        return build_t2i_code(args, **kwargs)
    
    raise ValueError(f'dataset {args.dataset} is not supported')
