import os
from PIL import Image
import torch.utils.data as data


class MSCOCO(data.Dataset):

    def __init__(self, root, train=True, image_transforms=None):
        super().__init__()

        self.train = train
        folder_name = 'train2017' if train else 'val2017'

        from pycocotools.coco import COCO
        self.root = os.path.join(root, folder_name)
        caption_path = os.path.join(root, 'annotations/instances_{}.json'.format(folder_name))
        self.coco = COCO(caption_path)
        self.imgIds = []
        self.catIds = []
        cats = ['person','car','traffic light', 'stop sign']
        
        num_images = 250 # Shrink number of images per class
        for cat in cats:
            catId = self.coco.getCatIds(catNms=cat)
            self.imgIds.extend(self.coco.getImgIds(catIds=catId)[0:num_images])
            self.catIds.extend([catId]*num_images)
        # self.imgIds = list(self.coco.imgs.keys())

        self.image_transforms = image_transforms

    def __getitem__(self, index):
        img_id = self.imgIds[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        label = self.catIds[index]

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        return index, img, img, label

    def __len__(self):
        return len(self.imgIds)