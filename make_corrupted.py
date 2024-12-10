import h5py
from pathlib import Path
from einops import rearrange

from utils.corruption import get_noise_dict, make_noise, load_mat, norm
from multiprocessing import Pool
from functools import partial

LABLE_DICT = {'Acinetobacter_baumannii': 0,
              'Bacillus_subtilis': 1,
              'Enterobacter_cloacae': 2,
              'Enterococcus_faecalis': 3,
              'Escherichia_coli': 4,
              'Haemophilus_influenzae': 5,
              'Klebsiella_pneumoniae': 6,
              'Listeria_monocytogenes': 7,
              'Micrococcus_luteus': 8,
              'Proteus_mirabilis': 9,
              'Pseudomonas_aeruginosa': 10,
              'Serratia_marcescens': 11,
              'Staphylococcus_aureus': 12,
              'Staphylococcus_epidermidis': 13,
              'Stenotrophomonas_maltophilia': 14,
              'Streptococcus_agalactiae': 15,
              'Streptococcus_anginosus': 16,
              'Streptococcus_pneumoniae': 17,
              'Streptococcus_pyogenes': 18}


def create_a_dataset(path, noise_type):
    root = 'corrupted_dataset'
    label = path.parts[-2]
    label_index = LABLE_DICT[label]
    Path(f'{root}/{label_index}').mkdir(parents=True, exist_ok=True)
    volume = rearrange(norm(load_mat(path)), 'h w d -> d h w')
    for severity in range(5):
        noised_volume = make_noise(volume, noise_type, severity)
        with h5py.File(f'{root}/{label_index}/{path.stem}.h5', 'a') as h:
            if noise_type not in h:
                h.create_group(noise_type)
            g = h[noise_type]
            g.create_dataset(str(severity), data=noised_volume)


def create_all_dataset(path):
    root = 'corrupted_dataset'
    noise_dict = get_noise_dict()
    label = path.parts[-2]
    label_index = LABLE_DICT[label]
    Path(f'{root}/{label_index}').mkdir(parents=True, exist_ok=True)
    volume = rearrange(norm(load_mat(path)), 'h w d -> d h w')
    for noise_type in noise_dict:
        for severity in range(5):
            noised_volume = make_noise(volume, noise_type, severity)
            with h5py.File(f'{root}/{label_index}/{path.stem}.h5', 'a') as h:
                if noise_type not in h:
                    h.create_group(noise_type)
                g = h[noise_type]
                g.create_dataset(str(severity), data=noised_volume)


def main_all():
    path_list = list(Path('dataset/test').rglob("*.mat"))
    with Pool(50) as pool:
        pool.map(create_all_dataset, path_list)
    # for path in path_list:
    #     create_all_dataset(path)


def main_partial(noise_type):
    path_list = list(Path('dataset/test').rglob("*.mat"))
    create_noise_fn = partial(create_a_dataset, noise_type=noise_type)
    with Pool(50) as pool:
        pool.map(create_noise_fn, path_list)


if __name__ == "__main__":
    main_all()
    # main_partial('jpeg_compression')
