import sys, os, argparse, json
import mmap
import numpy as np
from concurrent.futures import ThreadPoolExecutor
# https://huggingface.co/docs/safetensors/index

HEADER_SIZE = 8
byteorder='little'
newbyteorder = '<'
sttype_to_nptype = {
    'F64' : np.dtype(np.float64).newbyteorder(newbyteorder),
    'F32' : np.dtype(np.float32).newbyteorder(newbyteorder),
    'F16' : np.dtype(np.float16).newbyteorder(newbyteorder),
    #'BF16' : np.dtype(np.float16).newbyteorder(newbyteorder),
    'I64' : np.dtype(np.int64).newbyteorder(newbyteorder),
    'I32' : np.dtype(np.int32).newbyteorder(newbyteorder),
    'I16' : np.dtype(np.int16).newbyteorder(newbyteorder),
    'I8' : np.dtype(np.int8).newbyteorder(newbyteorder),
    'U8' : np.dtype(np.uint8).newbyteorder(newbyteorder),
    'BOOL' : np.dtype(np.bool_).newbyteorder(newbyteorder),
}
to_np_type = {
    'F64' : 'float64',
    'F32' : 'float32', 
    'F16' : 'float16',
    #'BF16' 
    'I64' : 'int64',
    'I32' : 'int32',
    'I16' : 'int16',
    'I8' : 'int8',
    'U8' : 'uint8',
    'BOOL' : 'bool'
}
size_dict_st = {
    'F64' : 8,
    'F32' : 4,
    'F16' : 2,
    #'BF16' : 2,
    'I64' : 8,
    'I32' : 4,
    'I16' : 2,
    'I8' : 1,
    'U8' : 1,
    'BOOL' : 1,
}

class safetensor_loader:
    def __init__(self, path):
        f = open(path, 'r+b')
        self.mm = mmap.mmap(f.fileno(), 0)

        self.header_size = int.from_bytes(self.mm[0:HEADER_SIZE], byteorder=byteorder)
        self.json_dict = json.loads(self.mm[HEADER_SIZE: HEADER_SIZE + self.header_size].decode('utf-8'))
        if '__metadata__' in self.json_dict:
            self.metadata = self.json_dict['__metadata__']
            del self.json_dict['__metadata__']

        self.data_offset = self.header_size + HEADER_SIZE
        self.filename = os.path.basename(path)

    def get_json(self):
        return self.json_dict

    def get_resized_json(self, sttype):
        ret = {}
        new_unit = size_dict_st[sttype]
        old_begin = 0
        for k, v in self.json_dict.items():
            new_size = new_unit * int(np.prod(v["shape"]))
            ret[k] = {
                "dtype" : sttype,
                "shape" : v["shape"],
                "data_offsets": [old_begin, old_begin + new_size]
            }
            old_begin = old_begin + new_size

        return ret

    def get_params(self, key, nptype=None):
        if not key in self.json_dict:
            raise Exception(f'key {key} is not found.')

        val = self.json_dict[key]
        begin, end = val['data_offsets']
        buffer = np.frombuffer(self.mm[self.data_offset + begin : self.data_offset + end], dtype=sttype_to_nptype[ val['dtype'] ] )
        if nptype != to_np_type[val['dtype']]:
            buffer = buffer.astype(nptype)
        return buffer

def write_multi_thread(out_path, safetensor1, safetensor2, base_ratio, sttype='F16', num_thread=0, metadata="", write_metadata=True, weights=[]):
    if num_thread == 0:
        num_thread = os.cpu_count()

    with open(out_path, "wb") as wf:
        # avoid "cannot mmap an empty file" error
        wf.write(1024*1024*b'\0')
        sys.stdout.flush()

    with open(out_path, 'r+b') as wf:
        wmm = mmap.mmap(wf.fileno(), 0, access=mmap.ACCESS_WRITE)

        # write header size
        json_file = safetensor1.get_resized_json(sttype)
        if write_metadata:
            json_file['__metadata__'] = {
                "Model_A" : safetensor1.filename,
                "Model_B" : safetensor2.filename,
                "ratio" : str(base_ratio),
                "text" : metadata,
                "Model_A metadata" : json.dumps(safetensor1.metadata),
                "Model_B metadata" : json.dumps(safetensor2.metadata),
            }

        json_binary =bytes(json.dumps(json_file).replace(' ', ''), "utf-8")
        json_size = len(json_binary)

        if write_metadata:
            del json_file['__metadata__']
        
        total_size = HEADER_SIZE
        wmm.resize(total_size)
        wmm.write((json_size).to_bytes(8, byteorder))

        # write json
        total_size += json_size
        wmm.resize(total_size)
        wmm.write(json_binary)

        # fix position_id https://note.com/bbcmc/n/n12c05bf109cc
        position_id_keys = [ k for k, v in json_file.items() if k.endswith('position_ids')]

        # merge
        if num_thread == 1:
            np_type = to_np_type[sttype]
            for k, v in json_file.items():
                begin, end = v['data_offsets']
                total_size += end - begin
                ratio = get_ratio(weights, k, base_ratio)

                try:
                    w1 = safetensor1.get_params(k, nptype=np_type)
                    w2 = safetensor2.get_params(k, nptype=np_type)
                    weighted = ratio * w1 + (1-ratio) * w2
                except:
                    weighted = w1
                
                if k in position_id_keys:
                    # fix position_id https://note.com/bbcmc/n/n12c05bf109cc
                    weighted.round()

                wmm.resize(total_size)
                wmm.write(weighted.tobytes())
        else:
            # merge on memory
            def merge_mt(args):
                k, json_file, safetensor1, safetensor2, base_ratio, np_type, weights = args
                try:
                    w1 = safetensor1.get_params(k, nptype=np_type)
                    w2 = safetensor2.get_params(k, nptype=np_type)
                    ratio = get_ratio(weights, k, base_ratio)
                    weighted = (ratio * w1 + (1-ratio) * w2)

                    if k in position_id_keys:
                        # fix position_id https://note.com/bbcmc/n/n12c05bf109cc
                        weighted.round()
                    json_file[k]['weights'] = weighted.tobytes()
                except:
                    json_file[k]['weights'] = w1.tobytes()

            arg_zip = [(k, json_file, safetensor1, safetensor2, base_ratio, to_np_type[sttype], weights) for k, v in json_file.items()]
            with ThreadPoolExecutor(max_workers=num_thread) as thread_pool:
                thread_pool.map(merge_mt, arg_zip)

            # write
            for k, v in json_file.items():
                begin, end = v['data_offsets']
                total_size += end - begin
                wmm.resize(total_size)
                wmm.write(v['weights'])

def get_ratio(weights, k, base_ratio):
    for key, weight in weights:
        if key in k:
            return weight
    return base_ratio

def weight_loader(path):
    with open(path, 'r', encoding='UTF-8') as f:
        data = [float(i) for i in f.read().split(',')]
        l = len(data)
        if l == 25:
            print("Apply SD 1.x or 2.x weights.")
        elif l == 17:
            print("Apply SDXL weights.")
        else:
            print(f"Invalid form. Input file {path} contains {l} weights. SD 1.x or 2.x just contains 25 weights. SDXL just contains 17 weights.")
            return []
        
        out = []
        num_block = int((l-1)/2)
        for i in range(0, num_block):
            out.append([f"diffusion_model.input_blocks.{i}", data[i]])
        out.append(["diffusion_model.middle_block", data[num_block]])
        for i in range(0, num_block):
            out.append([f"diffusion_model.output_blocks.{i}", data[num_block + 1 + i]])
    return out

parser = argparse.ArgumentParser()
parser.add_argument("pathA", type=str)
parser.add_argument("pathB", type=str)
parser.add_argument("out", type=str)
parser.add_argument("ratio", type=float, default=0.5, help="ratio * A + (1-ratio) * B. --weight_file_path takes precedence")
parser.add_argument("dtype", type=str, default="F16", help="F64, F32, F16, I64, I32, I16, I8, U8, BOOL")
parser.add_argument("--weight_file_path", type=str, help="25 weigts if SD 1.x or 2.x model. 17 weights if SDXL model.")
parser.add_argument("--num_thread", type=int, default=0, help="Use os.cpu_count() if default. Use lesser RAM if num_thread is 1")
parser.add_argument("--meta_data", type=str, default="", help="meta data")
parser.add_argument("--allow_overwrite", action='store_true', help="Overwrite out file if ture")
parser.add_argument("--discard_metadata", action='store_true')

args = parser.parse_args()
if os.path.isfile(args.out) and not args.allow_overwrite:
    print("Out file exists. Using --allow_overwrite overwrite an out file.")
    exit(1)

st = safetensor_loader(args.pathA)
st2 = safetensor_loader(args.pathB)

weights = []
if args.weight_file_path:
    if os.path.isfile(args.weight_file_path):
        weights = weight_loader(args.weight_file_path)
    else:
        print(f"Weight file {args.weight_file_path} is not found.")
        exit(1)

write_multi_thread(args.out, st, st2, args.ratio, sttype=args.dtype, num_thread=args.num_thread, metadata=args.meta_data, write_metadata=not args.discard_metadata, weights=weights)