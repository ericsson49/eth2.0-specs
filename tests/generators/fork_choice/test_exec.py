from pathlib import Path

import sys

from util import spec, store_copy, decode, encode, yaml

from eth2spec.utils import bls

default_bls_active = False

from eth2spec.utils.ssz.ssz_typing import Bytes32, uint64

def get_frames(f, acc=None):
    if acc is None:
        acc = []
    acc.append(f)
    if f.tb_next is not None:
        return get_frames(f.tb_next, acc)
    else:
        return acc

def last_frame_line(ex):
    fr = get_frames(ex.__traceback__)[-1]
    with open(fr.tb_frame.f_code.co_filename) as f:
        return f.readlines()[fr.tb_frame.f_lineno-1]

def get_step_key(step):
    return [k for k in step.keys() if k != 'root'][0]

part_cache = {}
def load_cached_part(part_id):
    global part_cache
    if part_id not in part_cache:
        part_cache[part_id] = yaml.load(Path(test_dir, 'cache', part_id))
    return part_cache[part_id]

def do_step(store_holder, step, msg_buffer, fixSignatures=False):
    key = get_step_key(step)
    if type(step[key]) is str:
        step_data = load_cached_part(step[key])
    else:
        step_data = step[key]
    if key == 'slot':
        store = store_copy(store_holder[0])
        slot = decode(step[key], spec.Slot)
        while spec.get_current_slot(store) < slot:
            spec.on_tick(store, store.time+1)
        store_holder[0] = store
    elif key == 'block':
        block = decode(step_data, spec.SignedBeaconBlock)
        if fixSignatures:
            pre = store.block_states[block.message.parent_root].copy()
            spec.process_slots(pre,block.message.slot)
            proposer = pre.validators[spec.get_beacon_proposer_index(pre)]
            signing_root = spec.compute_signing_root(block.message, spec.get_domain(pre, spec.DOMAIN_BEACON_PROPOSER))
            block.signature = bls.Sign(pubkey_to_privkey[proposer.pubkey], signing_root)
        store = store_copy(store_holder[0])
        try:
            spec.on_block(store, block)
            store_holder[0] = store
        except AssertionError as e:
            msg_buffer.append(block)
            #raise
        msg_buffer.extend(block.message.body.attestations)
    elif key == "attestation":
        attestation = decode(step_data, spec.Attestation)
        store = store_copy(store_holder[0])
        try:
            spec.on_attestation(store, attestation)
            store_holder[0] = store
        except AssertionError as e:
            msg_buffer.append(attestation)
            #raise
    elif key == "checks":
        store = store_copy(store_holder[0])
        checks = step[key]
        for check, data in checks.items():
            value = data['value']
            optional = data['optional']
            if check == 'block_in_store':
                decode(value, Bytes32) in store.blocks
            elif check == 'block_not_in_store':
                res = decode(value, Bytes32) not in store.blocks
            elif check == 'head':
                res = decode(value, Bytes32) == spec.get_head(store)
            elif check == 'justified_checkpoint_epoch':
                res = decode(value, uint64) == store.justified_checkpoint.epoch
            else:
                raise Exception('unknown check:', check)
            if optional:
                if not res:
                    print(check + " failed, but it's optional")
            else:
                assert res, check + ' failed'
    else:
        print('unknown step kind:', key)

def redo_steps(store_holder, buff):
    buff_copy = buff
    buff = []
    for msg in buff_copy:
        if isinstance(msg, spec.SignedBeaconBlock):
            do_step(store_holder, {'block': encode(msg)}, buff)
        elif isinstance(msg, spec.Attestation):
            do_step(store_holder, {'attestation': encode(msg)}, buff)
    return buff

def do_steps(genesis_state, steps):
    buff = []
    store_holder = [spec.get_forkchoice_store(genesis_state)]
    for step in steps:
        key = get_step_key(step)
        if key == "slot":
            data = (key, step['slot'])
        else:
            data = (key,)
        #print('step ', *data)
        buff = redo_steps(store_holder, buff)
        do_step(store_holder, step, buff)

def run_test(n, test_case):
    print(n)
    if test_case.get('meta', {}).get('bls_setting', 0) == 1:
        bls.bls_active = True
    else:
        bls.bls_active = default_bls_active
    genesis_state = decode(load_cached_part(test_case['genesis']), spec.BeaconState)
    try:
        do_steps(genesis_state, test_case['steps'])
    except Exception as e:
        #if (len(e.args) > 0 and e.args[0] not in []):
        print('exception', e)
        print(last_frame_line(e))
        raise
    else:
        print('OK')

def run_tests(test_cases):
    for n,tc in test_cases:
        run_test(n, tc)

if len(sys.argv) >= 2:
    test_dir = Path(sys.argv[1])
else:
    test_dir = Path('integration_tests')

test_cases = [(p.parts[-2:],d) for p,d in [(p, yaml.load(p)) for p in test_dir.glob('*/*.yaml')] if 'steps' in d]

if len(test_cases) == 0:
    print('no tests found')
else:
    run_tests(test_cases)