import torch
import time
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler
# TESTE-----------------
import torchvision
import cv2
# ----------------------


class HMERDataset(Dataset):
    def __init__(self, params, image_path, label_path, words, is_train=True, transform=None):
        super(HMERDataset, self).__init__()
        if image_path.endswith('.pkl'):
            with open(image_path, 'rb') as f:
                self.images = pkl.load(f)
        elif image_path.endswith('.list'):
            with open(image_path, 'r') as f:
                lines = f.readlines()
            self.images = {}
            print(f'data files: {lines}')
            for line in lines:
                name = line.strip()
                print(f'loading data file: {name}')
                start = time.time()
                with open(name, 'rb') as f:
                    images = pkl.load(f)
                self.images.update(images)
                print(f'loading {name} cost: {time.time() - start:.2f} seconds!')

        with open(label_path, 'r') as f:
            self.labels = f.readlines()

        self.words = words
        self.is_train = is_train
        self.params = params
        
        # TESTE---------------------------
        self.transform = transform
        # --------------------------------

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)

    def __getitem__(self, idx):
        name, *labels = self.labels[idx].strip().split()
        #name = name.split('.')[0] if name.endswith('jpg') else name
        image = self.images[name]
        #image = torch.Tensor(255-image) / 255
        image = torch.Tensor(image) / 255
        image = image.unsqueeze(0)
        labels.append('eos')
        words = self.words.encode(labels)
        words = torch.LongTensor(words)
        
        
        # TESTE -----------------------------------------
        if self.transform:
            image = self.transform(image)
        # -----------------------------------------------
        
        return image, words


def get_crohme_dataset(params):
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    print(f"train images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"test images: {params['eval_image_path']} labels: {params['eval_label_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True)
    
    
    # TESTE-------------------------------------------------------

    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToPILImage(),
    #     torchvision.transforms.RandomResizedCrop(size=(150, 150),
    #                                              #scale=(0.8, 1.2), #original de sergio
    #                                              scale=(0.7, 1.25),
    #                                              #ratio=(3.0 / 4.0, 4.0 / 3.0)), #original de sergio
    #                                              ratio=(0.65, 1.5)),
    #     torchvision.transforms.RandomRotation(degrees=45), #original de sergio: degrees=30
    #     torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.8), #original de sergio: sharpness_factor=2
    #     torchvision.transforms.ElasticTransform(alpha=40.0, sigma=3.0), #original de sergio: alpha=40.0, sigma=3.0
    #     # torchvision.transforms.RandomEqualize(),
    #     torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=0.6), #nova
    #
    #     torchvision.transforms.ToTensor()
    # ])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomResizedCrop(size=(150, 150),
                                                 scale=(0.8, 1.2),
                                                 ratio=(3.0 / 4.0, 4.0 / 3.0)),
        torchvision.transforms.RandomRotation(degrees=30),
        torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
        torchvision.transforms.ElasticTransform(alpha=40.0, sigma=3.0),
        # torchvision.transforms.RandomEqualize(),

        torchvision.transforms.ToTensor()
    ])

    datasets_list = [train_dataset]

    for i in range(params['data_augmentation']):
        train_dataset_transformed = HMERDataset(params, params['train_image_path'], params['train_label_path'], words, is_train=True,
                                                  transform=transform)
        datasets_list.append(train_dataset_transformed)

    train_dataset_final = torch.utils.data.ConcatDataset(datasets_list)

    # ------------------------------------------------------------
    
    
    
    
    eval_dataset = HMERDataset(params, params['eval_image_path'], params['eval_label_path'], words, is_train=False)

    #train_sampler = RandomSampler(train_dataset)
    train_sampler = RandomSampler(train_dataset_final)
    
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset_final, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict[params['collate_fn']], pin_memory=True)

    print(f'train dataset: {len(train_dataset_final)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


def collate_fn(batch_images):
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    
    #print("\nIMAGES COLLATE: ")
    #print(str(images.size()))
    #print("\nimages_mask: ")
    #print(str(image_masks.size()))
    
    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'共 {len(words)} 类符号。')
        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label


collate_fn_dict = {
    'collate_fn': collate_fn
}
