import os
from collections import OrderedDict
from datetime import datetime

def get_typed_value(s, key_type):
    if key_type == "num":
        try:
            return int(s)
        except:
            try:
                return float(s)
            except:
                raise ValueError("Cannot read value {} as num".format(s))
    elif key_type == "int":
        try:
            return int(s)
        except:
            raise ValueError("Cannot read value {} as int".format(s))
    elif key_type == "float":
        try:
            return float(s)
        except:
            raise ValueError("Cannot read value {} as float".format(s))
    elif key_type == "bool":
        try:
            return bool(s)
        except:
            raise ValueError("Cannot read value {} as bool".format(s))
    elif key_type == "path":
        try:
            s = os.path.expandvars(s)
            d = os.path.dirname(s)
            if not os.path.exists(d):
                print("Created directory: {}".format(d))
                os.makedirs(d)
            return s
        except:
            raise ValueError("Cannot read value {} as path".format(s))
    elif key_type == "date":
        try:
            return datetime.strptime(s, "%Y%m%d")
        except:
            raise ValueError("Cannot read value {} as date".format(s))
    elif key_type == "str" or key_type is None:
        return s
    else:
        raise ValueError("Unknown type {}".format(key_type))

class Config(object):
    def __init__(self, filename=None, vsep="=", tsep=":"):
        self.default = OrderedDict()
        self.sections = OrderedDict()
        self.filename = None
        if filename is not None:
            self.from_file(filename, vsep=vsep, tsep=tsep)

    def __add_value(self, key, value, section):
        if section is None:
            if key in self.default:
                raise ValueError("default key '{}' appears twice".format(key))
            self.default[key] = value
        else:
            if section not in self.sections:
                self.sections[section] = OrderedDict()
            if key in self.sections[section]:
                raise ValueError("key '{}' appears twice in section '{}'".format(key, section))
            self.sections[section][key]= value
