from pathlib import Path

import torch
import hydra

from dataset.texture_end2end_dataset import TextureEnd2EndDataset
from dataset.texture_map_dataset import TextureMapDataset
from model.attention import Fold2D
from model.retrieval import get_input_feature_extractor, get_target_feature_extractor
from util.misc import load_net_for_eval


class InferenceEnd2End:

    def __init__(self, config):
        self.preload_dict = {}
        self.hparams = config
        self.fenc_input, self.fenc_target = get_input_feature_extractor(config), get_target_feature_extractor(config)
        self.dataset = lambda split: TextureEnd2EndDataset(config, split, self.preload_dict)
        self.dataset('train')
        self.vis_dataset = self.dataset('val_vis')
        self.fold = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, config.dictionary.patch_size, 3)
        self.device = torch.device("cuda:0")

    def run_inference(self, K):
        output_dir = Path("runs") / self.hparams.experiment / "visualization"
        output_dir.mkdir(exist_ok=True, parents=True)

        fenc_input = load_net_for_eval(self.fenc_input, self.hparams.resume, "fenc_input")
        fenc_target = load_net_for_eval(self.fenc_target, self.hparams.resume, "fenc_target")
        loader = torch.utils.data.DataLoader(self.vis_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)

        with torch.no_grad():
            candidate_codes = self.vis_dataset.get_all_texture_patch_codes(fenc_target, self.device, self.hparams.batch_size)
            for batch_idx, batch in enumerate(loader):
                self.vis_dataset.move_batch_to_gpu(batch, self.device)
                self.vis_dataset.apply_batch_transforms(batch)
                features_in = torch.nn.functional.normalize(fenc_input(batch['partial_texture']), dim=1).cpu()
                features_candidates = candidate_codes.unsqueeze(0).expand(features_in.shape[0], -1, -1)
                argsort_indices = torch.argsort(torch.einsum('ik,ijk->ij', features_in, features_candidates), dim=1, descending=True)
                knn_retrievals = [batch['texture'].cpu()]
                for k in range(K):
                    retrieved_texture = self.vis_dataset.get_patches_with_indices(argsort_indices[:, k])
                    retrieved_texture = TextureMapDataset.apply_mask_texture(self.fold(retrieved_texture), batch['mask_texture'].cpu())
                    knn_retrievals.append(retrieved_texture.clone())
                self.vis_dataset.visualize_texture_knn_batch(torch.cat(knn_retrievals).numpy(), K, output_dir / f"{batch_idx:04d}.jpg")


@hydra.main(config_path='../config', config_name='texture_end2end')
def main(config):
    inference_handler = InferenceEnd2End(config)
    inference_handler.run_inference(8)


if __name__ == "__main__":
    main()
