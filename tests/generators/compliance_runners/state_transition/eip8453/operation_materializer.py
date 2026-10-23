"""Shared signed operation fixtures and serialization for EIP-8453."""

from functools import lru_cache

from eth_consensus_specs.utils import bls
from tests.generators.compliance_runners.state_transition.materializer import Materializer
from tests.generators.compliance_runners.state_transition.validation_helpers import (
    bls_enabled as real_bls,
)


@lru_cache(maxsize=128)
def signature(keys, root):
    with real_bls():
        return bls.Aggregate([bls.Sign(key, root) for key in keys])


def sign(s, pre, data, domain_type, epoch, indices):
    domain = s.get_domain(pre, domain_type, epoch)
    return signature(
        tuple(int(i) + 1 for i in indices), bytes(s.compute_signing_root(data, domain))
    )


def verify(s, pre, data, domain_type, epoch, indices, sig):
    if not indices or any(int(i) >= len(pre.validators) for i in indices):
        return False
    with real_bls():
        return bls.FastAggregateVerify(
            [pre.validators[i].pubkey for i in indices],
            s.compute_signing_root(data, s.get_domain(pre, domain_type, epoch)),
            sig,
        )


@lru_cache(maxsize=8)
def _base(s):
    with real_bls():
        validators = [
            s.Validator(
                pubkey=bls.SkToPk(i + 1),
                activation_epoch=0,
                exit_epoch=s.FAR_FUTURE_EPOCH,
                withdrawable_epoch=s.FAR_FUTURE_EPOCH,
                effective_balance=s.MAX_EFFECTIVE_BALANCE_ELECTRA,
            )
            for i in range(8)
        ]
    return s.BeaconState(
        slot=3 * s.SLOTS_PER_EPOCH + 2,
        validators=s.Validators(data=validators),
        balances=s.Balances(data=[s.MAX_EFFECTIVE_BALANCE_ELECTRA] * 8),
        inactivity_scores=s.InactivityScores(data=[0] * 8),
        round_participation=s.RoundParticipation(data=[0] * 8),
        target_participation=s.HeightParticipation(data=[s.Boolean(0)] * 8),
        progress_participation=s.HeightParticipation(data=[s.Boolean(0)] * 8),
    )


def base_state(s):
    return _base(s).copy()


def checkpoint(s, height, marker=None):
    return s.HeightCheckpoint(height=height, root=(marker or height).to_bytes(32, "little"))


class OperationMaterializer(Materializer):
    runner_name = "operations"

    def __init__(self, spec, fork_name="eip8453", preset_name="minimal", seed=0):
        if spec.fork != "eip8453" or fork_name != spec.fork:
            raise ValueError("operation materializer requires EIP-8453")
        super().__init__(spec, fork_name, preset_name, seed)

    def materialize_solution(self, solution):
        claimed = self.claimed_record(self.target, solution)
        with real_bls():
            pre, operation = self.build_inputs(solution)
            post = pre.copy()
            try:
                getattr(self.spec, f"process_{self.handler_name}")(post, operation)
            except AssertionError:
                post = None
        if (post is not None) != claimed["accepted"]:
            raise ValueError(f"Unexpected outcome for {self.handler_name}: {claimed}")
        parts = [
            ("pre", "ssz", pre.encode_bytes()),
            (self.input_name, "ssz", operation.encode_bytes()),
        ]
        if post is not None:
            parts.append(("post", "ssz", post.encode_bytes()))
        return {
            "description": f"process_{self.handler_name}",
            "bls_setting": 1,
            "claimed": claimed,
        }, parts
