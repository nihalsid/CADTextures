import traceback

import hydra
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from pyflann import *
import torch.utils.data

from dataset.texture_map_dataset import TextureMapDataset
from model.retrieval import Patch16
from util.misc import to_underscore, load_net_for_eval, normalize_tensor_color
from util.timer import Timer
import multiprocessing


def create_dictionary(feature_extractor, dictionary_config, latent_dim, dataset, tree_path):
    tree_path.mkdir(exist_ok=True, parents=True)
    num_patch_x = dataset.texture_map_size // dictionary_config.patch_size
    number_of_patches = len(dataset) * num_patch_x ** 2

    database = np.zeros((number_of_patches, 1 + 2 * 2 + latent_dim), dtype=np.float32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dictionary_config.batch_size, shuffle=False, num_workers=dictionary_config.num_workers, drop_last=False)
    with torch.no_grad():
        db_idx = 0
        for i, item in enumerate(tqdm(dataloader, desc='dict_feats')):
            dataset.apply_batch_transforms(item)
            target_patches = item['texture'].cuda()
            prediction = feature_extractor(target_patches)
            prediction = torch.nn.functional.normalize(prediction, dim=1).cpu().numpy()
            for batch_idx in range(item['texture'].shape[0]):
                for p_y in range(num_patch_x):
                    for p_x in range(num_patch_x):
                        embedding = prediction[batch_idx * num_patch_x ** 2 + p_y * num_patch_x + p_x]
                        texture_idx = i * dictionary_config.batch_size + batch_idx
                        x_start, x_end = p_x * dictionary_config.patch_size, (p_x + 1) * dictionary_config.patch_size
                        y_start, y_end = p_y * dictionary_config.patch_size, (p_y + 1) * dictionary_config.patch_size
                        database[db_idx, :] = np.hstack([np.array([texture_idx, x_start, x_end, y_start, y_end]), embedding])
                        db_idx += 1

    with Timer("Dictionary Indexing"):
        np.save(tree_path / "database", database)
        Path(tree_path / "index.json").write_text(json.dumps(dataset.items))
        flann_obj = FLANN()
        params = flann_obj.build_index(database[:, 5:], algorithm="kdtree", trees=64, log_level="info")
        # nihalsid: some past algo's we tried. autotune is too slow to be usable, linear gets unusable when too many patches - not sure if its really bruteforce
        # params = flann_obj.build_index(database[:, 7:], algorithm="autotuned", target_precision=1, log_level="info")
        # params = flann_obj.build_index(database[:, 7:], algorithm="linear", target_precision=1.0, log_level="info")
        Path(tree_path / "params.json").write_text(json.dumps(params))
        flann_obj.save_index((str(tree_path / "index_010_64_tree.idx")).encode('utf-8'))


def extract_features(feature_extractor, dictionary_config, latent_dim, dataset, key):
    num_patch_x = dataset.texture_map_size // dictionary_config.patch_size
    features = np.zeros((len(dataset) * num_patch_x ** 2, latent_dim), dtype=np.float32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dictionary_config.batch_size, shuffle=False, num_workers=dictionary_config.num_workers, drop_last=False)
    patch_names = []
    with torch.no_grad():
        f_idx = 0
        for i, item in enumerate(tqdm(dataloader, desc='query_features')):
            dataset.apply_batch_transforms(item)
            item[key] = item[key].cuda()
            prediction = torch.nn.functional.normalize(feature_extractor(item[key]), dim=1).cpu().numpy()
            for batch_idx in range(item[key].shape[0]):
                for p_y in range(num_patch_x):
                    for p_x in range(num_patch_x):
                        embedding = prediction[batch_idx * num_patch_x ** 2 + p_y * num_patch_x + p_x]
                        x_start, x_end = p_x * dictionary_config.patch_size, (p_x + 1) * dictionary_config.patch_size
                        y_start, y_end = p_y * dictionary_config.patch_size, (p_y + 1) * dictionary_config.patch_size
                        patch_names.append(f"{item['name'][batch_idx]}__{item['view_index'][batch_idx]:02d}__{x_start:03d}__{x_end:03d}__{y_start:03d}__{y_end:03d}")
                        features[f_idx, :] = embedding
                        f_idx += 1
    return patch_names, features


def extract_input_features(feature_extractor, dictionary_config, latent_dim, dataset):
    return extract_features(feature_extractor, dictionary_config, latent_dim, dataset, 'partial_texture')


def extract_target_features(feature_extractor, dictionary_config, latent_dim, dataset):
    return extract_features(feature_extractor, dictionary_config, latent_dim, dataset, 'texture')


def flann_knn_worker(results_dict, K, tree_path, worker_texture_names, worker_patch_names, worker_features, ignore_patches_from_source):
    try:
        flann_obj = FLANN()
        database = np.load(tree_path / "database.npy")
        flann_obj.load_index(str(tree_path / "index_010_64_tree.idx").encode('utf-8'), database[:, 5:])
        dataset_index = json.loads((tree_path / "index.json").read_text())
        params = json.loads((tree_path / "params.json").read_text())

        query_batch_size = 1024
        for batch_idx in tqdm(range(worker_features.shape[0] // query_batch_size + 1), 'flann_knn_worker'):
            feature_subset = worker_features[batch_idx * query_batch_size: (batch_idx + 1) * query_batch_size, :]
            worker_texture_name_subset = worker_texture_names[batch_idx * query_batch_size: (batch_idx + 1) * query_batch_size]
            worker_patch_name_subset = worker_patch_names[batch_idx * query_batch_size: (batch_idx + 1) * query_batch_size]
            results, dists = flann_obj.nn_index(feature_subset, 2 * K, checks=params['checks'])
            all_extents = np.stack([np.hstack((database[np.array(results[:, i]), 0:5], np.array(dists[:, i])[:, None])) for i in range(2 * K)]).transpose((1, 0, 2))
            for i in range(all_extents.shape[0]):
                if ignore_patches_from_source and worker_texture_name_subset[i] in dataset_index:
                    M = all_extents[i, :, 0] == dataset_index.index(worker_texture_name_subset[i])
                    all_extents[i, :, :] = np.concatenate((all_extents[i, ~M, :], all_extents[i, M, :]))
            all_extents = all_extents.transpose((1, 0, 2))
            for i, cn in enumerate(worker_patch_name_subset):
                results_dict[cn] = all_extents[:K, i, :]
                # if all_extents[:K, i, 0].max() >= len(dataset_index):
                #     print("ERROR: ", cn, all_extents[:K, i, 0])
    except Exception as err:
        print("FLANN Worker failed with exception", err)
        traceback.print_exc()


def query_dictionary_using_features_parallel(dictionary_config, tree_path, input_texture_names, input_patch_names, input_features, ignore_patches_from_source):
    manager = multiprocessing.Manager()
    # noinspection PyTypeChecker
    shared_dict = manager.dict([(input_patch_names[i], None) for i in range(len(input_patch_names))])
    items_per_worker = input_features.shape[0] // dictionary_config.flann_num_workers + 1
    process = []
    for pid in range(dictionary_config.flann_num_workers):
        worker_texture_names = input_texture_names[pid * items_per_worker: (pid + 1) * items_per_worker]
        worker_patch_names = input_patch_names[pid * items_per_worker: (pid + 1) * items_per_worker]
        worker_features = input_features[pid * items_per_worker: (pid + 1) * items_per_worker, :]
        process.append(multiprocessing.Process(target=flann_knn_worker, args=(shared_dict, dictionary_config.K, tree_path, worker_texture_names, worker_patch_names, worker_features, ignore_patches_from_source)))
    for p in process:
        p.start()
    for p in process:
        p.join()
    total_items = len(input_patch_names)
    total_mapped = 0
    retrieval_mapping = {}
    for item in shared_dict:
        retrieval_mapping[item] = shared_dict[item]
        if shared_dict[item] is not None:
            total_mapped += 1
    print(f"{total_mapped}/{total_items} mapped")
    return retrieval_mapping


def query_dictionary_using_features(dictionary_config, patch_names, input_features, dataset, tree_path, ignore_patches_from_source):
    texture_names = [p.split('__')[0] for p in patch_names]
    with Timer("FLANN"):
        if dictionary_config.flann_num_workers == 0:
            retrieval_mapping = dict.fromkeys(patch_names)
            flann_knn_worker(retrieval_mapping, dictionary_config.K, tree_path, texture_names, patch_names, input_features, ignore_patches_from_source)
        else:
            retrieval_mapping = query_dictionary_using_features_parallel(dictionary_config, tree_path, texture_names, patch_names, input_features, ignore_patches_from_source)
    return retrieval_mapping


def create_retrieval_from_mapping(texture_view_id, retrieval_mappings, K, dataset_train, dataset, dictionary_config, tree_path):
    num_patch_x = dataset.texture_map_size // dictionary_config.patch_size
    dataset_index = json.loads((tree_path / "index.json").read_text())
    texture_retrieval = torch.from_numpy(np.zeros((K, 3, dataset.texture_map_size, dataset.texture_map_size), dtype=np.float32))
    distances = torch.ones([K, dataset.texture_map_size, dataset.texture_map_size]) * 100
    all_patches_for_texture = []
    for p_y in range(num_patch_x):
        for p_x in range(num_patch_x):
            all_patches_for_texture.append(
                f"{texture_view_id}__{p_x * dictionary_config.patch_size:03d}__{(p_x + 1) * dictionary_config.patch_size:03d}__{p_y * dictionary_config.patch_size:03d}__{(p_y + 1) * dictionary_config.patch_size:03d}"
            )
    for k in range(K):
        for p in all_patches_for_texture:
            X0, X1, Y0, Y1 = retrieval_mappings[p][k, 1:5].astype(np.int32).tolist()
            current_distance = retrieval_mappings[p][k, 5]
            xx0, xx1, yy0, yy1 = map(int, p.split('__')[2:])
            if distances[k, yy0: yy1, xx0: xx1].mean() > current_distance:
                index_ptr = int(retrieval_mappings[p][k, 0])
                if index_ptr >= 0:
                    shape = torch.from_numpy(dataset_train.get_texture(dataset_index[index_ptr]))
                else:
                    shape = torch.from_numpy(np.zeros((3, dataset.texture_map_size, dataset.texture_map_size)))
                shape = normalize_tensor_color(shape.unsqueeze(0), dataset.color_space).squeeze(0)
                texture_retrieval[k, :, yy0: yy1, xx0: xx1] = shape[:, Y0:Y1, X0:X1]
                distances[k, yy0: yy1, xx0: xx1] = float(current_distance)
    return texture_retrieval


def get_error_retrieval(retrievals, dataset):
    total_error = 0
    for t_idx, texture in enumerate(tqdm(dataset.items, 'metric')):
        for v_idx in range(dataset.views_per_shape):
            retrieved_texture = retrievals[t_idx * dataset.views_per_shape + v_idx][0].numpy()
            error = np.abs(retrieved_texture - normalize_tensor_color(dataset.get_texture(texture)[np.newaxis, :, :, :], dataset.color_space).squeeze(0)).mean()
            total_error += error
    return total_error / (len(dataset.items) * dataset.views_per_shape)


class RetrievalInterface:

    def __init__(self, dictionary_config, latent_dim):
        self.config = dictionary_config
        self.latent_dim = latent_dim

    def get_retrieval_mapping(self, fenc, extraction_func, tree_path, dataset, ignore_patches_from_source):
        patch_names, feats_input = extraction_func(fenc, self.config, self.latent_dim, dataset)
        retrieval_mapping = query_dictionary_using_features(self.config, patch_names, feats_input, dataset, tree_path, ignore_patches_from_source)
        return retrieval_mapping

    def get_features(self, fenc_input, fenc_target, dataset):
        patch_names_0, feats_input = extract_input_features(fenc_input, self.config, self.latent_dim, dataset)
        patch_names_1, feats_target = extract_target_features(fenc_target, self.config, self.latent_dim, dataset)
        assert len(patch_names_0) == len(patch_names_1) and sorted(patch_names_0) == sorted(patch_names_1)
        return patch_names_0, feats_input, feats_target

    def retrieve_nearest_textures(self, retrieval_mapping, texture_view, K, tree_path, dataset_train, dataset):
        return create_retrieval_from_mapping(texture_view, retrieval_mapping, K, dataset_train, dataset, self.config, tree_path)

    def retrieve_nearest_textures_for_all(self, retrieval_mapping, texture_views, K, tree_path, dataset_train, dataset):
        retrieved_textures = []
        for texture_view in tqdm(texture_views, desc='recompose_textures'):
            retrieved_textures.append(self.retrieve_nearest_textures(retrieval_mapping, texture_view, K, tree_path, dataset_train, dataset).unsqueeze(0))
        return torch.cat(retrieved_textures, dim=0)

    def create_mapping_and_retrieve_nearest_textures_for_all(self, fenc_input, tree_path, dataset_train, dataset, K, ignore_patches_from_source):
        texture_views = []
        for view_idx in range(dataset.views_per_shape):
            texture_views.extend([f'{t}__{view_idx:02d}' for t in dataset.items])
        return self.retrieve_nearest_textures_for_all(self.get_retrieval_mapping(fenc_input, extract_input_features, tree_path, dataset, ignore_patches_from_source), texture_views, K, tree_path, dataset_train, dataset)


def get_retrievals_dir(config):
    ckpt_experiment = Path(config.retrieval_ckpt).parents[0].name
    ckpt_epoch = Path(config.retrieval_ckpt).name.split('.')[0]
    return Path(config.retrieval_dir, 'retrieval', to_underscore(config.dataset.name), config.dataset.splits_dir, ckpt_experiment, ckpt_epoch, str(config.dictionary.K))


def retrievals_to_disk(mode, config, use_target_for_feats, num_proc=1, proc=0):
    ckpt_experiment = Path(config.retrieval_ckpt).parents[0].name
    ckpt_epoch = Path(config.retrieval_ckpt).name.split('.')[0]
    retrievals_dir = get_retrievals_dir(config)
    preload_dict = {}
    tree_path = Path("runs", 'retrieval_scratch', to_underscore(config.dataset.name), config.dataset.splits_dir, ckpt_experiment, ckpt_epoch, str(config.dictionary.K))
    dataset_train = TextureMapDataset(config, 'train', preload_dict)
    dataset_val = TextureMapDataset(config, 'val', preload_dict)

    if mode == 'map':
        fenc_input, fenc_target = Patch16(config.fenc_nf, config.fenc_zdim), Patch16(config.fenc_nf, config.fenc_zdim)
        fenc_input = load_net_for_eval(fenc_input, Path(config.retrieval_ckpt), "fenc_input")
        fenc_target = load_net_for_eval(fenc_target, Path(config.retrieval_ckpt), "fenc_target")
        (retrievals_dir / "map").mkdir(exist_ok=True, parents=True)
        # create_dictionary(fenc_target, config.dictionary, config.fenc_zdim, dataset_train, tree_path)
        retrieval_handler = RetrievalInterface(config.dictionary, config.fenc_zdim)

        fenc = fenc_input if not use_target_for_feats else fenc_target
        extract_feats = extract_input_features if not use_target_for_feats else extract_target_features

        dataset_train.set_all_view_indexing(True)
        dataset_val.set_all_view_indexing(True)

        retrieval_mappings = retrieval_handler.get_retrieval_mapping(fenc, extract_feats, tree_path, dataset_train, True)
        with Timer('np_save_train'):
            np.save(retrievals_dir / "map_train.npy", retrieval_mappings)

        retrieval_mappings = retrieval_handler.get_retrieval_mapping(fenc, extract_feats, tree_path, dataset_val, False)
        with Timer('np_save_val'):
            np.save(retrievals_dir / "map_val.npy", retrieval_mappings)

    elif mode == "compose":
        retrieval_handler = RetrievalInterface(config.dictionary, config.fenc_zdim)
        (retrievals_dir / "compose").mkdir(exist_ok=True, parents=True)
        map_name = ['map_train.npy', 'map_val.npy']
        datasets = [dataset_train, dataset_val]
        for d_idx, dataset in enumerate(datasets):
            texture_views = []
            for view_idx in range(dataset.views_per_shape):
                texture_views.extend([f'{t}__{view_idx:02d}' for t in dataset.items])
            split_textures = [x for i, x in enumerate(texture_views) if i % num_proc == proc]
            retrieval_mapping = np.load(retrievals_dir / map_name[d_idx], allow_pickle=True)[()]
            for texture_view in tqdm(split_textures, desc=f'recompose_textures_{["train", "val"][d_idx]}'):
                retrieval = retrieval_handler.retrieve_nearest_textures(retrieval_mapping, texture_view, config.dictionary.K, tree_path, dataset_train, dataset)
                np.savez_compressed(retrievals_dir / "compose" / f"{texture_view}.npz", retrieval.numpy())
    elif mode == "evaluate":
        retrievals = []
        for texture in tqdm(dataset_val.textures, desc='evaluate'):
            retrieval = np.load(retrievals_dir / 'compose' / f'{texture}.npz')["arr_0"]
            retrievals.append(retrieval[:1, :, :, :])


@hydra.main(config_path='../config', config_name='retrieve_util')
def main(config):

    # --retrieval_ckpt /cluster_HDD/gondor/ysiddiqui/repatch/runs/03022058_3DFront_base_1000/_ckpt_epoch=79.ckpt --config config/surface_reconstruction/3DFront/retrieval_dummy_128_064.yaml
    # --retrieval_ckpt /cluster_HDD/gondor/ysiddiqui/repatch/runs/03020122_3DFront_iou_scale_bs196/_ckpt_epoch=29.ckpt --config config/superresolution/3DFront/retrieval_dummy_008_064.yaml
    # --retrieval_ckpt /cluster_HDD/gondor/ysiddiqui/repatch/runs/02021816_Matterport3D16_iou_scale_bs192/_ckpt_epoch=145.ckpt --config config/superresolution/Matterport3D/retrieval_official_016_064_iou_scale.yaml

    for _mode in config.mode:
        retrievals_to_disk(_mode, config, False, config.num_proc, config.proc)


if __name__ == '__main__':
    main()
