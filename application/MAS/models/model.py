from coala.tracking.evaluation import model_size
from application.MAS import models


def get_model(arch, tasks, pretrained=False):
    model = models.__dict__[arch](pretrained=pretrained, tasks=tasks)
    print(f"Model has {model_size(model)} MB parameters")
    try:
        print(f"Encoder has {model_size(model.encoder)} MB parameters")
    except:
        print(f"Each encoder has {model_size(model.encoders[0])} MB parameters")
    for decoder in model.task_to_decoder.values():
        print(f"Decoder has {model_size(decoder)} MB parameters")
    return model
