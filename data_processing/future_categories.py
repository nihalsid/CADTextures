import json
from pathlib import Path
from collections import defaultdict
from shutil import copytree, rmtree
from tqdm import tqdm

from util.misc import read_list


def get_category_ids(categories, future_cat_json_path):
    list_of_relevant_ids = []
    cat_json = json.loads(Path(future_cat_json_path).read_text())
    cat_models = defaultdict(list)
    for item in cat_json:
        cat_models[item['super-category']].append(item['model_id'])
    for item in categories:
        list_of_relevant_ids.extend(cat_models[item])
    return list_of_relevant_ids


def write_sofa_categories_to_file():
    cat_json_path = "/cluster/gondor/mdahnert/datasets/future3d/model_info.json"
    cats = ["Sofa"]
    relevant_ids = sorted(list(set(get_category_ids(cats, cat_json_path))))
    print(f"Number of models in {cats}: {len(relevant_ids)}")
    Path("relevant_ids.txt").write_text("\n".join(relevant_ids))


def copy_future_categories():
    source_path = Path("/cluster/gondor/mdahnert/datasets/future3d/3D-FUTURE-model")
    destination_path = Path("/cluster/gimli/ysiddiqui/future3d/3D-FUTURE-model/Sofa")
    destination_path.mkdir(exist_ok=True, parents=True)
    ids = read_list('data/splits/3D-FUTURE/Sofa/all.txt')
    for _id in tqdm(ids, desc='copy'):
        if (destination_path / _id).exists():
            rmtree(destination_path / _id)
        copytree(source_path / _id, destination_path / _id)


if __name__ == '__main__':
    copy_future_categories()
