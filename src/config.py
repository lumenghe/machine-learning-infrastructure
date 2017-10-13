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

    def __getitem__(self, item):
        if isinstance(item, tuple):
            section = item[0]
            key = item[1]
            return self.sections.get(section, {}).get(key, None)
        else:
            return self.default.get(item, None)

    def __contains__(self, item):
        if isinstance(item, tuple):
            section = item[0]
            key = item[1]
            return key in self.sections.get(section, {})
        else:
            return item in self.default

    def from_file(self, filename, vsep="=", tsep=":"):
        section = None
        with open(filename) as f:
            self.filename = filename
            for i, line in enumerate(f):
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1]
                    continue
                l = line.split(vsep, 1)
                if len(l) < 2:
                    raise ValueError("wrong format at line {}: {}", i+1, line)
                key = l[0].strip()
                value = l[1].strip()
                # typed key
                l = key.split(tsep)
                key_type = None
                is_list = False
                if len(l) == 2:
                    key_type = l[1].strip()
                    key = l[0].strip()
                    if key_type == "list" or (key_type.startswith("list(") and key_type.endswith(")")):
                        if key_type == "list":
                            key_type = None
                        else:
                            key_type = key_type.rstrip(")").split("list(")[1].strip()
                        is_list = True
                if value == "None":
                    value = None
                elif value == "False":
                    value = False
                elif value == "True":
                    value = True
                elif "," in value and not (value.startswith('"') and value.endswith('"')) or is_list:
                    value = [get_typed_value(x.strip(), key_type) for x in value.split(",")]
                else:
                    value = get_typed_value(value, key_type)
                self.__add_value(key, value, section)

    def __str__(self):
        l = []
        s = "~~~~~~~~~~~~~~~~~~~~ {} ~~~~~~~~~~~~~~~~~~~~\n\n".format(self.filename)
        for k,v in self.default.items():
             s += "{} = {}\n".format(k, v)
        l.append(s)
        if len(self.sections):
            for section, d in self.sections.items():
                s = "[{}]\n".format(section)
                for k,v in d.items():
                    s += "{} = {}\n".format(k, v)
                l.append(s)
        s = "~~~~~~~~~~~~~~~~~~~~ {} ~~~~~~~~~~~~~~~~~~~~\n".format(self.filename)
        l.append(s)
        return "\n".join(l)
