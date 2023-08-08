import yaml
from yaml.loader import SafeLoader
from munch import DefaultMunch


class Config:
    def __init__(self, pathfile):
        self.pathfile = pathfile
        assert pathfile[-5:] == '.yaml'

    def read(self, doc: int, munch=True):
        with open(self.pathfile, 'r', encoding='utf8') as f:
            data = list(yaml.load_all(f, Loader=SafeLoader))
        if munch:
            return DefaultMunch.fromDict(data[doc])
        else:
            return data[doc]

    def dump(self, data):
        with open(self.pathfile, 'w') as f:
            yaml.dump(data, f, sort_keys=True)
