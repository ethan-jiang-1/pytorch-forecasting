import inspect
import collections

class HyperParameters(collections.OrderedDict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def get_exist_val(self, name, default_val):
        if name in self:
            return self[name]
        return default_val

class CeInspect(object):
    @classmethod
    def inspect_instance_mro(cls, instance, mark):
        print("instance {}".format(mark), type(instance))
        print(instance.__class__.__name__)
        for kls in inspect.getmro(instance.__class__):
            print(" ", kls)

    @classmethod
    def inspect_instance_key_memebers(cls, obj, mark=None):
        if mark is not None:
            print(mark)
        try:
            for key in dict(obj):
                if not key.startswith("_"):
                    print(key)
            return
        except:
            if hasattr(obj, "__dict__"):
                for key in obj.__dict__:
                    if not key.startswith("_"):
                        print(key)        

    @classmethod
    def inspect_predefined(cls, lcs):
        keys = list(lcs.keys())
        keys = sorted(keys)
        for key in keys:
            key_cp = "{}".format(key)
            try:
                lc0, lc1 = key_cp[0], key_cp[1]
                if (lc0 >= 'A' and lc0 <= 'Z') and (lc1 >= 'A' and lc1 <= 'Z'):
                    obj = lcs[key]
                    if isinstance(obj, bool) or isinstance(obj, int) or isinstance(obj, float):
                        print(key, "\t", obj)
            except Exception as ex:
                print("exception occured", ex)
