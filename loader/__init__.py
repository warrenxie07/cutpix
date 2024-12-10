from .dataloaders import DataLoader


def get_loader(loader_type, root, image_size, batch_size, num_cores, dim, mix, seed):
    loader_dict = {
        "none": DataLoader,
    }
    loader = loader_dict[loader_type]
    arg_dict = dict(
        root=root,
        image_size=image_size,
        batch_size=batch_size,
        num_cores=num_cores,
        dim=dim,
        mix=mix,
        seed=seed,
    )

    return loader(**arg_dict)
