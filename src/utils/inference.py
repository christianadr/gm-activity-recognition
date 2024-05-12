from constants import globals as g
from mmaction.apis import inference_recognizer, init_recognizer


def run_inference(config_path: str, checkpoint_path: str, input_path: str):
    model = init_recognizer(config_path, checkpoint_path, device="cuda:0")
    result = inference_recognizer(model, input_path)
    idx = result.pred_label[0].item()

    return g.CLASS_NAMES[idx]
