

import contextlib
import datetime
import io
import json
import logging
import numpy as np
import os
import logging
import shutil
import pycocotools.mask as mask_util
from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
import cv2
from detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.utils.file_io import PathManager
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

# data: List[Dict] = DatasetCatalog.get("my_dataset")
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x] # 这里分割mask用多边形点集表示

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
# for d in ["train", "val"]:
#     # 注册数据集
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     # 数据集添加元信息，主要是类别名，用于可视化
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
# balloon_metadata = MetadataCatalog.get("balloon_train")



































DATA_ROOT = "/home/zyc/projects_dir/project_temp/course_image_segmentation/A_my_seg/dataset"
# 获取元数据
def _get_builtin_metadata(dataset_name):
    if dataset_name == "ElectronicComponent":
        DATA_ALL_CATEGORIES = [ {'id': 1, 'name': 'component'}]

        thing_ids = [k["id"] for k in DATA_ALL_CATEGORIES]
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        thing_classes = [k["name"] for k in DATA_ALL_CATEGORIES]
        return {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes
        }
    elif dataset_name == "cityscapes":
        # fmt: off
        CITYSCAPES_THING_CLASSES = [
            "person", "rider", "car", "truck",
            "bus", "train", "motorcycle", "bicycle",
        ]
        CITYSCAPES_STUFF_CLASSES = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle",
        ]
        # fmt: on
        return {
            "thing_classes": CITYSCAPES_THING_CLASSES,
            "stuff_classes": CITYSCAPES_STUFF_CLASSES,
        }
    elif dataset_name == "crack":
        STUFF_CLASSES = ["crack", "background" ]  # 裂缝更像是stuff类别
        return {
            "stuff_classes": STUFF_CLASSES
        }
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))




def load_electronic_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).

        # It works by looking at the "categories" field in the json, therefore
        # if users' own json also have incontiguous ids, we'll
        # apply this mapping as well but print a warning.
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if "coco" not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = coco_api.loadImgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'iscrowd': 0,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))




    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0

    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id

            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs
        record['sem_seg_file_name']=  os.path.join(image_root , '../../mask' ,os.path.basename(image_root),  img_dict["file_name"])
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process.  Please "
            "check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully"
        )
    return dataset_dicts

# 注册特定数据集 ElectronicComponent
def register_elec_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    if name not in DatasetCatalog:
        DatasetCatalog.register(name, lambda: load_electronic_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,     # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
        # json_file=json_file, image_root=image_root, evaluator_type="sem_seg", **metadata
    )

COCO_ELEC_SPLITS = {
	'elec_train': (
		os.path.join('ElectronicComponent', 'images', 'train'),
		os.path.join('ElectronicComponent', 'annotations', 'instances_train.json')),
    'elec_val': (
		os.path.join('ElectronicComponent', 'images', 'val'),
		os.path.join('ElectronicComponent', 'annotations', 'instances_val.json')),
	'elec_test': (
		os.path.join('ElectronicComponent', 'images', 'test'),
		os.path.join('ElectronicComponent', 'annotations', 'instances_test.json'))
}

# 注册 DATA_SPLITS 中所有数据集
def register_elec(root):
    for key, (image_root, json_file) in COCO_ELEC_SPLITS.items():
        assert os.path.exists(os.path.join(root, json_file))
        register_elec_instances(
            key,  # elec_train
            _get_builtin_metadata('ElectronicComponent'),
            os.path.join(root, json_file),
            os.path.join(root, image_root)
        )
register_elec(DATA_ROOT)




def _get_crack_files(image_dir, gt_dir):
    files = []
    # scan through the directory
    cities = PathManager.ls(image_dir)
    logger.info(f"{len(cities)} cities found in '{image_dir}'.")
 
    for basename in PathManager.ls(image_dir):
        image_file = os.path.join(image_dir, basename)
        filename = basename[:-len('.jpg')]
        label_file = os.path.join(gt_dir, filename + ".png")
        # json_file = os.path.join(image_dir, basename + "gtFine_polygons.json")
        # files.append((image_file,  label_file, json_file))
        files.append((image_file,  label_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files

def load_crack_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    # for image_file, _, label_file, json_file in _get_crack_files(image_dir, gt_dir):
    for image_file, label_file in _get_crack_files(image_dir, gt_dir):
        # label_file = label_file.replace("labelIds", "labelTrainIds")

        # with PathManager.open(json_file, "r") as f:
        #     jsonobj = json.load(f)
        image = Image.open(image_file)
        
        label = Image.open(label_file)
        width, height = image.size
        width2, height2 = label.size
        image.close()
        label.close()
        assert width == width2 and height == height2, f"Image and label widths do not match: {image_file}, {label_file}"
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": height,
                "width": width,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(
        ret[0]["sem_seg_file_name"]
    ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
    return ret

_RAW_CITYSCAPES_SPLITS = {
    "carck_train":(os.path.join( 'DeepCrack', 'train_img'), os.path.join( 'DeepCrack', 'train_lab')),
    "carck_test": (os.path.join( 'DeepCrack', 'test_img') , os.path.join( 'DeepCrack', 'test_lab'))
}
#   "cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
def register_crack(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("crack")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        if key not in DatasetCatalog:
            DatasetCatalog.register(
                key, lambda x=image_dir, y=gt_dir: load_crack_semantic(x, y)
            )
        MetadataCatalog.get(key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            # evaluator_type="cityscapes_sem_seg",
            evaluator_type="crack_sem_seg",
            ignore_label=255,
            **meta,
        )
register_crack(DATA_ROOT)

