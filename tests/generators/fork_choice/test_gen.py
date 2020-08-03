from pathlib import Path
from ruamel.yaml import YAML

from eth2spec.utils.ssz.ssz_typing import (
    Container, List, Bytes32, uint64
)
from util import spec, load_data, store_copy, decode

from eth2spec.debug.encode import encode

from eth2spec.test.helpers.deposits import prepare_genesis_deposits
from eth2spec.test.helpers.keys import pubkey_to_privkey

from eth2spec.utils import bls

from itertools import groupby
from collections import OrderedDict


class ChainStart(Container):
    eth1_block_hash: Bytes32
    eth1_timestamp: uint64
    deposit_root: Bytes32
    deposits: List[spec.Deposit, spec.VALIDATOR_REGISTRY_LIMIT]

def make_chain_start(eth1_block_hash=b'\x42' * 32, eth1_timestamp=spec.MIN_GENESIS_TIME, deposit_count=32):
    deposits, deposit_root, _ = prepare_genesis_deposits(spec, deposit_count, spec.MAX_EFFECTIVE_BALANCE, signed=True)
    return ChainStart(
        eth1_block_hash = eth1_block_hash,
        eth1_timestamp = eth1_timestamp,
        deposit_root = deposit_root,
        deposits = deposits
    )

def load_chain_start(path):
    return load_data(path, ChainStart)

def get_genesis_state(genesis_file=None, chain_start=None, chain_start_file=None):
    if genesis_file is not None:
        return load_data(Path(genesis_file), spec.BeaconState)
    else:
        if chain_start is None:
            chain_start_path = Path(chain_start_file)
            chain_start = load_chain_start(chain_start_path)
        return spec.initialize_beacon_state_from_eth1(chain_start.eth1_block_hash, chain_start.eth1_timestamp, chain_start.deposits)

def block_root(b):
    return spec.hash_tree_root(b.message)

def advance_state(state, slot):
    assert state.slot <= slot
    r = state.copy()
    if state.slot < slot:
        spec.process_slots(r, slot)
    return r

def is_finalized_checkpoint_ancestor(store, signed_block):
    block = signed_block.message
    tmp_store = store_copy(store)
    root = spec.hash_tree_root(block)
    tmp_store.blocks[root] = block
    finalized_slot = spec.compute_start_slot_at_epoch(tmp_store.finalized_checkpoint.epoch)
    return block.slot > finalized_slot and spec.get_ancestor(tmp_store, root, finalized_slot) == tmp_store.finalized_checkpoint.root

class TestEvent(object):
    pass

class SlotEvent(TestEvent):
    def __init__(self, slot):
        self.slot = slot

class BlockEvent(TestEvent):
    def __init__(self, block):
        self.block = block

class AttestationEvent(TestEvent):
    def __init__(self, attestation):
        self.attestation = attestation

class CheckEvent(TestEvent):
    def __init__(self, checks):
        self.checks = checks

class State(object):
    def __init__(self, state):
        self.store = spec.get_forkchoice_store(state)
        self.att_cache = []
        self.buff = []
        self.events = []
        self.head_check()
        self.set_bls_setting(0)
    
    def set_bls_setting(self, bls_setting):
        self.bls_setting = bls_setting
        #if bls_setting == 1:
        #    bls.bls_active = True
        #else:
        #    bls.bls_active = False

    def _add_check(self, kind, param):
        if len(self.events) == 0 or not isinstance(self.events[-1], CheckEvent):
            self.events.append(CheckEvent({}))
        checks = self.events[-1].checks
        if kind in checks:
            raise Exception()
        else:
            checks[kind] = param
    
    def head_check(self):
        self._add_check('head', encode(self.get_head()))
    
    def block_in_store_check(self, block_root):
        assert block_root in self.store.blocks
        self._add_check('block_in_store', encode(block_root))

    def block_not_in_store_check(self, block_root):
        assert block_root not in self.store.blocks
        self._add_check('block_not_in_store', encode(block_root))
    
    def justified_checkpoint_epoch_check(self, checkpoint):
        assert self.store.justified_checkpoint.epoch == checkpoint.epoch
        self._add_check('justified_checkpoint_epoch', encode(checkpoint.epoch))

    def get_head(self):
        return spec.get_head(self.store)

    def set_slot(self, slot):
        while spec.get_current_slot(self.store) < slot:
            store = self.store
            spec.on_tick(store, store.time + spec.SECONDS_PER_SLOT)
            self.events.append(SlotEvent(encode(spec.get_current_slot(self.store))))
            self.redo_steps()
            self.head_check()

    def send_block(self, b, redo=False):
        if not redo:
            self.events.append(BlockEvent(b))
        store = store_copy(self.store)
        try:
            spec.on_block(store, b)
        except:
            self.buff.append(b)
        else:
            self.store = store
            self.buff.extend(b.message.body.attestations)
            if not redo:
                self.redo_steps()
        if not redo:
            self.head_check()

    def send_attestation(self, a, redo=False):
        if not redo:
            self.events.append(AttestationEvent(a))
        store = store_copy(self.store)
        try:
            spec.on_attestation(store, a)
        except:
            self.buff.append(a)
        else:
            self.store = store
            if not redo:
                self.redo_steps()
        if not redo:
            self.head_check()

    def redo_steps(self):
        buff_copy = self.buff[:]
        self.buff = []
        for msg in buff_copy:
            if isinstance(msg, spec.SignedBeaconBlock):
                self.send_block(msg, redo=True)
            elif isinstance(msg, spec.Attestation):
                self.send_attestation(msg, redo=True)

def bitlist_to_indices(bl):
    return set(i for i, e in enumerate(bl) if e)

def merge_agg_bits(bls):
    acc = set()
    for bl in bls:
        acc = acc.union(bitlist_to_indices(bl))
    return acc

def group_atts_by_data(atts):
    return dict((g, merge_agg_bits([pa.aggregation_bits for pa in pas])) for g, pas in groupby(atts, lambda a: a.data))

def find_new_atts(prev_atts, atts):
    grouping = group_atts_by_data(prev_atts)
    new_atts = []
    for a in atts:
        indices = bitlist_to_indices(a.aggregation_bits)
        if len(indices) != 1:
            raise Exception('only single attester attestations are supproted')
        if a.data not in grouping or list(indices)[0] not in grouping[a.data]:
            new_atts.append(a)
    res = []
    for data, _ats in groupby(new_atts, lambda a: a.data):
        ats = list(_ats)
        bl = spec.Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE]([False] * (ats[0].aggregation_bits.length()))
        for idx in merge_agg_bits([a.aggregation_bits for a in ats]):
            bl[idx] = True
        agg_sig = spec.get_aggregate_signature(ats)
        res.append(spec.Attestation(aggregation_bits = bl, data = data, signature = agg_sig))
    return res

def mk_block(st, slot, parent_ref, atts=[], graffiti=0,bad_parent=False, bad_state=False, bad_signature=False):
    if isinstance(parent_ref, spec.SignedBeaconBlock):
        parent_root = block_root(parent_ref)
    elif isinstance(parent_ref, spec.Bytes32):
        parent_root = parent_ref
    else:
        raise Exception("parent_root should be either a block or a root")
    store = st.store
    parent_state = store.block_states[parent_root].copy()
    state = advance_state(parent_state, slot)
    proposer = spec.get_beacon_proposer_index(state)
    SK = pubkey_to_privkey[state.validators[proposer].pubkey]
    randao_reveal = spec.get_epoch_signature(state, spec.BeaconBlock(slot=slot), SK)
    eth1vote = spec.get_eth1_vote(state, [])
    new_atts = find_new_atts(list(state.previous_epoch_attestations) + list(state.current_epoch_attestations), atts)
    block = spec.BeaconBlock(slot=slot, proposer_index=proposer, parent_root=parent_root,
        body=spec.BeaconBlockBody(
            randao_reveal=randao_reveal,attestations=new_atts,eth1_data=eth1vote,graffiti=graffiti.to_bytes(32,spec.ENDIANNESS)))
    if bad_state:
        block.state_root = spec.Root()
    else:
        block.state_root = spec.compute_new_state_root(parent_state, block)
    if bad_parent:
        block.parent_root = spec.Root()
    if bad_signature:
        block_signature = spec.BLSSignature()
    else:
        block_signature = spec.get_block_signature(state, block, SK)
    return spec.SignedBeaconBlock(message=block, signature=block_signature)

def get_attesters(st, slot, head):
    store = st.store
    state = advance_state(store.block_states[head], slot)
    attesters = []
    for index in range(spec.get_committee_count_per_slot(state, slot)):
        committee = spec.get_beacon_committee(state, slot, index)
        attesters.extend(committee)
    return attesters

def mk_atts(st, slot, head_ref, attesters=None, bad_signature=False, bad_index=False, bad_target_epoch=False, bad_target_root=False, allow_offchain_blocks=False):
    store = st.store
    if isinstance(head_ref, spec.SignedBeaconBlock):
        head = block_root(head_ref)
    elif isinstance(head_ref, spec.Bytes32):
        head = head_ref
    else:
        raise Exception("head_ref should be either a block or a root")
    if head not in store.block_states and isinstance(head_ref, spec.SignedBeaconBlock) and allow_offchain_blocks:
        parent_state = store.block_states[head_ref.message.parent_root].copy()
        state = spec.state_transition(parent_state, head_ref, False)
    else:
        state = advance_state(store.block_states[head], slot)
    src = state.current_justified_checkpoint
    start_slot = spec.compute_start_slot_at_epoch(spec.get_current_epoch(state))
    epoch_boundary_block_root = head if start_slot == state.slot else spec.get_block_root_at_slot(state, start_slot)
    target_chkpt = spec.Checkpoint(epoch=spec.get_current_epoch(state), root=epoch_boundary_block_root)
    atts = []
    committee_count = spec.get_committee_count_per_slot(state, slot)
    for index in range(committee_count):
        data = spec.AttestationData(slot=slot, index=index, beacon_block_root=head,
            source=state.current_justified_checkpoint,
            target=target_chkpt)
        if bad_index:
            data.index += committee_count*8
        if bad_target_epoch:
            data.target.epoch = spec.FAR_FUTURE_EPOCH
        if bad_target_root:
            data.target.root = spec.hash_tree_root(spec.Bytes96((123456789).to_bytes(96,spec.ENDIANNESS)))
        committee = spec.get_beacon_committee(state, slot, index)
        for i in range(len(committee)):
            if attesters == None or committee[i] in attesters: 
                bits = spec.Bitlist[spec.MAX_VALIDATORS_PER_COMMITTEE]([0] * len(committee))
                bits[i] = 1
                if bad_signature:
                    att_sig = spec.BLSSignature()
                else:
                    att_sig = spec.get_attestation_signature(state, data,
                        pubkey_to_privkey[state.validators[committee[i]].pubkey])
                atts.append(spec.Attestation(aggregation_bits=bits, data=data, signature=att_sig))
    return atts

def convert_test_event(e):
    if isinstance(e, SlotEvent):
        d = ('slot', e.slot)
    elif isinstance(e, BlockEvent):
        d = ('block', e.block)
    elif isinstance(e, AttestationEvent):
        d = ('attestation', e.attestation)
    elif isinstance(e, CheckEvent):
        d = ('checks', e.checks)
    else:
        raise Exception("Unsupported event type " + type(e))
    return d

def convert_test_case(bls, genesis, steps, path):
    cache_path = path.joinpath('cache')
    data = []
    for s in steps:
        (kind, o) = convert_test_event(s)
        if kind == "block" or kind == "attestation":
            v = cache_object(cache_path, kind, o).name
        else:
            v = o
        data.append({kind: v})
    return {
        'meta': {'bls_setting': bls},
        'genesis': cache_object(cache_path, "state", genesis).name,
        'steps': data
    }

def dump_test_case(bls, genesis, steps, path, name):
    tc = convert_test_case(bls, genesis, steps, path)
    d,n = name.split('__')
    dp = path.joinpath(d)
    if not dp.exists():
        dp.mkdir()
    fp = dp.joinpath(n + '.yaml')
    yaml.dump(tc, fp)


yaml = YAML(pure=True)

caches = {}

def load_obj(p):
    return yaml.load(p)

def store_obj(o, path):
    yaml.dump(o, path)

def get_cache(path):
    if path not in caches:
        if not path.exists():
            path.mkdir()
        res = {}
        for p in path.glob('*.yaml'):
            kind = p.name.split('_')[0]
            if kind == 'state':
                o = decode(load_obj(p), spec.BeaconState)
            elif kind == 'block':
                o = decode(load_obj(p), spec.SignedBeaconBlock)
            elif kind == 'attestation':
                o = decode(load_obj(p), spec.Attestation)
            res[spec.hash_tree_root(o)] = p.relative_to(path)
        caches[path] = res
    return caches[path]

def cache_object(path, prefix, o):
    cache = get_cache(path)
    root = spec.hash_tree_root(o)
    if root not in cache:
        cnt = len(list(path.glob(prefix + "_*.yaml")))
        p = path.joinpath(prefix + "_" + str(cnt) + ".yaml")
        yaml.dump(encode(o), p)
        pssz = path.joinpath(prefix + "_" + str(cnt) + ".ssz")
        with open(pssz, 'wb') as f:
            f.write(o.encode_bytes())
        cache[root] = pssz.relative_to(path)
    return cache[root]
