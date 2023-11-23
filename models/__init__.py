from tpu_perf.infer import SGInfer
import numpy as np 
import time 
import torch
import os

SGTypeTuple = (
   (np.float32, 0),
   (np.int32, 6),
   (np.uint32, 7),
   (np.int8, 2),
   (np.uint8, 3),
)

typemap = {
    0: np.float32,
    6: np.int32,
    7: np.uint32,
    2: np.int8,
    3: np.uint8,
}


def generate_func(shapes, dtype, mode=1):
    # 0: random 
    # 1: zero
    return np.random.random(shapes).astype(dtype) if mode == 0 else np.zeros(shapes).astype(dtype)


class EngineOV:
    
    def __init__(self, model_path="", batch=1, device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            # print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])
        
    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
    def generate_randome_data(self):
        info = self.model.get_input_info()
        # {'latent.1': {'scale': 1.0, 'dtype': 0, 'shape': [2, 4, 128, 128]}, 't.1': {'scale': 1.0, 'dtype': 0, 'shape': [1]}, 'prompt_embeds.1': {'scale': 1.0, 'dtype': 0, 'shape': [2, 77, 2048]}, 'add_text_embeds.1': {'scale': 1.0, 'dtype': 0, 'shape': [2, 1280]}, 'add_time_ids.1': {'scale': 1.0, 'dtype': 0, 'shape': [2, 6]}}
        res = {}
        for k,v in info.items():
            res[k] = generate_func(v["shape"], typemap[v['dtype']], 1)
        return list(res.values())
    
        
    def __call__(self, args):
        start = time.time()
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
            # print(values)
        # print(time.time() - start)
        # start = time.time()
        task_id = self.model.put(*values)
        # print("put time : ",time.time() - start)
        task_id, results, valid = self.model.get()
        return results

