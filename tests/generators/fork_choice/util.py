from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML(pure=True)
yaml.default_flow_style = None

from eth2spec.phase0 import spec

configs_path = '../../../configs/'
from eth2spec.config import config_util
from importlib import reload
config_util.prepare_config(configs_path, 'minimal')
# reload spec to make loaded config effective
reload(spec)

from typing import Any

from eth2spec.utils.ssz.ssz_typing import (
    uint, Container, ByteList, List, boolean,
    Vector, ByteVector,
    Bytes32, uint64, Bitlist, Bitvector
)

from eth2spec.debug.encode import encode

def decode(data: Any, typ):
    if issubclass(typ, (uint, boolean)):
        if type(data) is str:
            return typ(int(data))
        else:
            if data < 0 and issubclass(typ, uint64):
                data = 2**64+data
            return typ(data)
    elif issubclass(typ, (List, Vector)):
        return typ(decode(element, typ.element_cls()) for element in data)
    elif issubclass(typ, (ByteList, ByteVector)):
        return typ(bytes.fromhex(data[2:]))
    elif issubclass(typ, (Bitlist,Bitvector)):
        v = bytes.fromhex(data[2:])
        if issubclass(typ, Bitvector):
            ln = typ.vector_length()
        else:
            ln = None
            for l in range(7,-1,-1):
                if v[-1] & (1 << l) != 0:
                    ln = l
                    break
            if ln is None:
                raise Exception("Wrong Bitlist format")
        res = [0] * ((len(v)-1)*8 + ln)
        for i in range(len(res)):
            if v[i // 8] & (1 << (i % 8)) != 0:
                res[i] = 1
        return typ(res)
    elif issubclass(typ, Container):
        temp = {}
        for field_name, field_type in typ.fields().items():
            temp[field_name] = decode(data[field_name], field_type)
            if field_name + "_hash_tree_root" in data:
                assert (data[field_name + "_hash_tree_root"][2:] ==
                        hash_tree_root(temp[field_name]).hex())
        ret = typ(**temp)
        if "hash_tree_root" in data:
            assert (data["hash_tree_root"][2:] ==
                    hash_tree_root(ret).hex())
        return ret
    else:
        raise Exception(f"Type not recognized: data={data}, typ={typ}")


def dump_bls_caches(fn = 'bls_caches.yaml'):
    yaml.dump({'verify_cache': bls.verify_cache, 'fa_verify_cache': bls.fa_verify_cache, 'sign_cache': bls.sign_cache},
    Path(fn))

def load_bls_caches(fn = 'bls_caches.yaml'):
    caches = yaml.load(Path(fn))
    bls.verify_cache = caches['verify_cache']
    bls.fa_verify_cache = caches['fa_verify_cache']
    bls.sign_cache = caches['sign_cache']

def dump_data(data, path):
    yaml.dump(encode(data), path)

def load_data(path, cls):
    return decode(yaml.load(path), cls)

def store_copy(store):
    return spec.Store(
        time=store.time,
        genesis_time=store.genesis_time,
        justified_checkpoint=store.justified_checkpoint.copy(),
        finalized_checkpoint=store.finalized_checkpoint.copy(),
        best_justified_checkpoint=store.best_justified_checkpoint.copy(),
        blocks=store.blocks.copy(),
        block_states=store.block_states.copy(),
        checkpoint_states=store.checkpoint_states.copy(),
        latest_messages=store.latest_messages.copy()
    )

