import os, sys
import torch
import torch.nn.functional as torch_F
import matplotlib.pyplot as plt
import utils.options as options
import tqdm
import clip
import csv
import importlib
from utils.util import log
from PIL import Image

class NN_annotator():
    def __init__(self, opt):
        super().__init__()
        self.clip_encoder, self.preprocess = clip.load("ViT-L/14", device=opt.device)
        self.clip_dim = 512

    @torch.no_grad()
    def compute_NN(self, opt):
        # iterate on the loader
        for self.split, loader in self.loaders.items():
            self.compute_NN_split(opt, loader)

    def compute_NN_split(self, opt, loader):
        raise NotImplementedError

    @torch.no_grad()
    def calc_matches(self, opt, features, k_nearest=6):
        indices = []
        values = []
        N = features.shape[0]
        for i in tqdm.tqdm(range(N), desc="Calculating k-NN"):
            # perform KNN query for ith sample
            query = features[i].unsqueeze(0)
            # calcualte the cosine similarity
            cos_sim = (query * features).sum(dim=1)
            if opt.thres is None:
                top_k_val, top_k_ind = cos_sim.topk(k_nearest, largest=True)
                indices.append(top_k_ind)
                values.append(top_k_val)
            else:
                index = ((cos_sim >= opt.thres) & (cos_sim < 1.)).nonzero()
                n_valid = len(index)
                if n_valid < k_nearest - 1:
                    top_k_val, top_k_ind = cos_sim.topk(k_nearest, largest=True)
                    indices.append(top_k_ind)
                    values.append(top_k_val)
                else:
                    index_sampled = index[torch.randperm(n_valid)[:k_nearest-1]].squeeze(1)
                    index_selfinclusive = torch.cat([torch.tensor([i]).to(opt.device), index_sampled], dim=0)
                    value_sampled = cos_sim[index_selfinclusive]
                    indices.append(index_selfinclusive)
                    values.append(value_sampled)
        # log the mean similarity by K
        values = torch.stack(values, dim=0)
        return indices, values

    # visualize samples by showing the nearest neighbors
    def save_vis(self, opt, label2path, root, labels, ind, values, k_nearest=6, n_vis=15):
        N = len(labels)
        count = 1
        sample_id = [N//n_vis*i for i in range(n_vis)]
        plt.figure(figsize=(5*k_nearest, 5*n_vis))

        for i in sample_id:
            # get the source image
            label = labels[i]
            image_path = label2path(root, label)[0]
            image = Image.open(image_path).convert('RGB')

            # visualize the source image
            plt.subplot(n_vis, k_nearest, count)
            plt.imshow(image)
            plt.xticks([])
            plt.yticks([])
            count += 1

            # visualize the nearest neighbors
            top_k_ind = ind[i][1:]
            for j, index in enumerate(top_k_ind):
                label = labels[index]
                image_path = label2path(root, label)[0]
                image = Image.open(image_path).convert('RGB')

                # visualize the source image
                plt.subplot(n_vis, k_nearest, count)
                plt.imshow(image)
                similarity = values[i, j+1].item()
                plt.title('{:.3f}'.format(similarity), fontweight='bold')
                plt.xticks([])
                plt.yticks([])
                count += 1

        plt.tight_layout()
        plt.savefig(os.path.join(opt.output_path, 'CLIP_NN_{}.png'.format(self.split)))

    def save_anno(self, opt, label2path, labels, index_topk, value_topk, k_nearest=6, category_set='all'):
        # save the NN results as csv
        category_name = opt.data[opt.data.dataset].cat.replace(', ', '_') if category_set == 'custom' else category_set
        csv_path = os.path.join(opt.anno_root, '{}_{}.csv'.format(category_name, self.split))
        os.makedirs(opt.anno_root, exist_ok=True)
        with open(csv_path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            header = ['Query']
            for i in range(1, k_nearest):
                header.append('Top_{}'.format(i))
            for i in range(1, k_nearest):
                header.append('Top_{}_score'.format(i))
            csvwriter.writerow(header)
            for i, label in enumerate(labels):
                rel_path = label2path('', label)[0]
                current_row = [rel_path]
                for index in index_topk[i][1:]:
                    rel_path = label2path('', labels[index])[0]
                    current_row.append(rel_path)
                for value in value_topk[i][1:]:
                    current_row.append('{:.4f}'.format(value))
                csvwriter.writerow(current_row)
        # sort the csv for better readability
        with open(csv_path, 'r') as csvfile:
            csvreader = list(csv.reader(csvfile))
            sortedlist = sorted(csvreader[1:], key=lambda row: row[0])
        with open(csv_path, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csvreader[0])
            csvwriter.writerows(sortedlist)

class Pix3D_annotator(NN_annotator):

    def __init__(self, opt):
        super().__init__(opt)
    
    def load_dataset(self, opt):
        data = importlib.import_module("data.{}".format(opt.data.dataset))
        log.info("loading training data...")
        self.train_data = data.Dataset(opt, split="train", transform=self.preprocess)
        self.train_loader = self.train_data.setup_loader(opt, shuffle=False, drop_last=False)
        log.info("loading validation data...")
        self.val_data = data.Dataset(opt, split="val", transform=self.preprocess)
        self.val_loader = self.val_data.setup_loader(opt, shuffle=False, drop_last=False)
        log.info("loading testing data...")
        self.test_data = data.Dataset(opt, split="test", transform=self.preprocess)
        self.test_loader = self.test_data.setup_loader(opt, shuffle=False, drop_last=False)
        self.loaders = {
            'val': self.val_loader,
            'train': self.train_loader,
            'test': self.test_loader
        }

    def compute_NN_split(self, opt, loader):
        # iterate on the data and save the embedding with labels
        # set the root paths
        root_image = os.path.join(loader.dataset.path, 'img_processed')
        label_list_all = loader.dataset.rel_path_list

        # create lists
        feature_list = []
        label_list = []

        for i, sample in enumerate(tqdm.tqdm(loader, desc="CLIP Inference on {} [{}]".format(opt.data.dataset, self.split))):
            image = sample['rgb_input'].to(opt.device)
            current_labels = label_list_all[i*opt.batch_size:(i+1)*opt.batch_size]
            for label in current_labels:
                label_list.append(label)
            CLIP_embedding = self.clip_encoder.encode_image(image).float()
            CLIP_embedding = torch_F.normalize(CLIP_embedding, dim=-1)
            feature_list.append(CLIP_embedding)

        # calculate the nearest neighbor
        CLIP_features = torch.cat(feature_list, dim=0)

        index_topk, value_topk = self.calc_matches(opt, CLIP_features, k_nearest=opt.k_nearest)
        self.save_vis(opt, self.label2path, root_image, label_list, index_topk, value_topk, k_nearest=opt.k_nearest, n_vis=15)
        self.save_anno(opt, self.label2path, label_list, index_topk, value_topk, k_nearest=opt.k_nearest, category_set='custom')

    def label2path(self, root, label):
        path = os.path.join(
            root,
            label
        )
        return path, None

def main():
    log.process(os.getpid())
    log.title("[{}] (compute CLIP-NN)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):
        if opt.data.dataset.startswith('pix3d'):
            annotator = Pix3D_annotator(opt)
        else:
            raise NotImplementedError
        annotator.load_dataset(opt)
        annotator.compute_NN(opt)

if __name__ == "__main__":
    main()