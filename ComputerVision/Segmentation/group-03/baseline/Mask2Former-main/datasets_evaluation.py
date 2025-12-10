import glob
import logging
import numpy as np
import os
import sys
import tempfile
from collections import OrderedDict
import torch
from PIL import Image
from collections import namedtuple
import fnmatch

from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager

from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator


def printError(message):
    """Print an error message and quit"""
    print('ERROR: ' + str(message))
    sys.exit(-1)

# elec components

# crack

# <city>_123456_123456_gtFine_labelIds.png
def getPrediction( args, groundTruthFile ):
    # determine the prediction path, if the method is first called
    if not args.predictionPath:
        rootPath = None
        if 'CITYSCAPES_RESULTS' in os.environ:
            rootPath = os.environ['CITYSCAPES_RESULTS']
        elif 'CITYSCAPES_DATASET' in os.environ:
            rootPath = os.path.join( os.environ['CITYSCAPES_DATASET'] , "results" )
        else:
            rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..','results')

        if not os.path.isdir(rootPath):
            printError("Could not find a result root folder. Please read the instructions of this method.")

        args.predictionPath = rootPath

    # walk the prediction path, if not happened yet
    if not args.predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(args.predictionPath):
            walk.append( (root,filenames) )
        args.predictionWalk = walk

    baseName = os.path.basename(groundTruthFile)
    baseName = baseName[:-4]
    filePattern = "{}_pred.png".format(baseName)

    predictionFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError("Found multiple predictions for ground truth {}".format(groundTruthFile))
    if not predictionFile:
        printError("Found no prediction for ground truth {}".format(groundTruthFile))
    return predictionFile



class CrackSemSegEvaluator(CityscapesEvaluator):
    """
    Evaluate semantic segmentation results on cityscapes dataset using cityscapes API.

    Note:
        * It does not work in multi-machine distributed training.
        * It contains a synchronization, therefore has to be used on all ranks.
        * Only the main process runs evaluation.
    """
    def reset(self):
        self._working_dir = tempfile.TemporaryDirectory(prefix="cityscapes_eval_" , dir= './output')
        self._temp_dir = self._working_dir.name
        
        # All workers will write to the same results directory
        # TODO this does not work in distributed training
        assert (
            comm.get_local_size() == comm.get_world_size()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        self._temp_dir = comm.all_gather(self._temp_dir)[0]
        if self._temp_dir != self._working_dir.name:
            self._working_dir.cleanup()
        self._logger.info(
            "Writing cityscapes results to temporary directory {} ...".format(self._temp_dir)
        )
        # del self._working_dir
        #self._working_dir.cleanup = lambda: None  # 覆盖 cleanup 方法为空操作
    
    def process(self, inputs, outputs):
        # from deeplearning.projects.cityscapesApi.cityscapesscripts.helpers.labels import (
        # from cityscapesscripts.helpers.labels import (
        #     trainId2label,
        # )
        from crack_eval import trainId2label
        
        for input, output in zip(inputs, outputs):
            file_name = input["file_name"]
            basename = os.path.splitext(os.path.basename(file_name))[0]
            pred_filename = os.path.join(self._temp_dir, basename + "_pred.png")

            output = output["sem_seg"].argmax(dim=0).to(self._cpu_device).numpy()
            pred = 255 * np.ones(output.shape, dtype=np.uint8)
            for train_id, label in trainId2label.items():
                if label.ignoreInEval:
                    continue
                pred[output == train_id] = label.id
            Image.fromarray(pred).save(pred_filename)
            #保存结果
            # save_path = "./output/crack_eval"
            # os.makedirs(save_path, exist_ok=True)
            # Image.fromarray(pred).save(os.path.join(save_path, basename + "_pred.png"))

    def evaluate(self):
        comm.synchronize()
        if comm.get_rank() > 0:
            return
        # Load the Cityscapes eval script *after* setting the required env var,
        # since the script reads CITYSCAPES_DATASET into global variables at load time.
        # import deeplearning.projects.cityscapesApi.cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval  # noqa: E501
        # import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as cityscapes_eval  # noqa: E501
        import crack_eval as  cityscapes_eval  # noqa: E501

        self._logger.info("Evaluating results under {} ...".format(self._temp_dir))

        # set some global states in cityscapes evaluation API, before evaluating
        cityscapes_eval.args.predictionPath = os.path.abspath(self._temp_dir)
        cityscapes_eval.args.predictionWalk = None
        cityscapes_eval.args.JSONOutput = False
        cityscapes_eval.args.colorized = False

        # These lines are adopted from
        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py # noqa
        gt_dir = PathManager.get_local_path(self._metadata.gt_dir)
        groundTruthImgList = glob.glob(os.path.join(gt_dir, "*.png"))
        assert len(
            groundTruthImgList
        ), "Cannot find any ground truth images to use for evaluation. Searched for: {}".format(
            cityscapes_eval.args.groundTruthSearch
        )
        predictionImgList = []
        for gt in groundTruthImgList:
            predictionImgList.append(getPrediction(cityscapes_eval.args, gt))
 
        results = cityscapes_eval.evaluateImgLists(
            predictionImgList, groundTruthImgList, cityscapes_eval.args
        )
        ret = OrderedDict()
        # ret["sem_seg"] = {
        #     "IoU": 100.0 * results["averageScoreClasses"],
        #     "iIoU": 100.0 * results["averageScoreInstClasses"],
        #     "IoU_sup": 100.0 * results["averageScoreCategories"],
        #     "iIoU_sup": 100.0 * results["averageScoreInstCategories"],
        # }

        ret["sem_seg"] = {
            "IoU": 100.0 * results["averageScoreClasses"],
        }
        self._working_dir.cleanup()
        return ret
