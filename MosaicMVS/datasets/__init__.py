import importlib


# find the dataset definition by name, for example dtu_yao (dtu_yao.py)
def find_dataset_def(dataset_name):
    module_name = 'datasets.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    if dataset_name == "dataset_mosaic" :
        return getattr(module, "MosaicMVSDataset")
