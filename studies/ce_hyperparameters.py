import inspect
import collections

class DefaultHyperParameters(object):
    MAX_ENCODER_LENGTH = 24
    MAX_PREDICTION_LENGTH = 6
    BATCH_SIZE = 64

    gradient_clip_val = 0.02
    hidden_size = 43
    dropout = 0.15
    hidden_continuous_size = 29
    attention_head_size = 1
    learning_rate = 0.009 


class HyperParameters(collections.OrderedDict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            if hasattr(DefaultHyperParameters, name):
                print("Warning: load default parameter in DefaultHyperParameters for {} as {}".format(name, getattr(DefaultHyperParameters, name)))
                return getattr(DefaultHyperParameters, name)
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
    
    def __str__(self) -> str:
        output = ""
        for name in self:
            output += "{}={}\n".format(name, self[name])
        for name in dir(DefaultHyperParameters):
            if not name.startswith("_"):
                if name not in self:
                    output += "  {}={}\n".format(name, getattr(DefaultHyperParameters, name))
        return output
            

def _exam():
    hp = HyperParameters()

    val = hp.gradient_clip_val
    print(val)

    try:
        val = hp.xxx
    except Exception as ex:
        print("exception occured", ex)

    hp.xxx = "xxx"
    print(hp)


if __name__ == '__main__':
    _exam()
