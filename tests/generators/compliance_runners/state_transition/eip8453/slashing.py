"""Shared fixtures and independent input observation for both slashing domains."""

from .operation_materializer import base_state, checkpoint, OperationMaterializer, sign, verify
from .operation_target import operation_target

FAULTS = (
    "NONE",
    "FIRST_SIGNATURE",
    "SECOND_SIGNATURE",
    "EMPTY_FIRST",
    "UNSORTED_SECOND",
    "DISJOINT",
    "ALREADY_SLASHED",
)


def target(legacy, observer):
    kinds = (
        ("DOUBLE", "SURROUND", "REVERSE_SURROUND", "IDENTICAL", "NONCONFLICTING")
        if legacy
        else (
            "DOUBLE",
            "CONFLICT_FORWARD",
            "CONFLICT_REVERSE",
            "TIMEOUT_CONFLICT",
            "IDENTICAL",
            "TIMEOUTS",
            "NONCONFLICTING",
        )
    )
    return operation_target(
        "legacy_attester_slashing" if legacy else "attester_slashing",
        {
            "kind": kinds,
            "fault": FAULTS,
            "multiple": bool,
        },
        lambda x: (
            (x["fault"] == "NONE")
            & (
                (x["kind"] == "DOUBLE") | (x["kind"] == "SURROUND")
                if legacy
                else (x["kind"] != "IDENTICAL")
                & (x["kind"] != "TIMEOUTS")
                & (x["kind"] != "NONCONFLICTING")
            )
        ),
        observer,
    )


class SlashingMaterializer(OperationMaterializer):
    legacy = False

    def build_inputs(self, solution):
        s = self.spec
        pre = base_state(s)
        t = s.heze if self.legacy else s
        kind, fault = solution.kind, solution.fault
        if self.legacy:
            a = t.AttestationData(source=t.Checkpoint(epoch=1), target=t.Checkpoint(epoch=2))
            b = a.copy()
            b.beacon_block_root = b"\x02" * 32
            if kind in ("SURROUND", "REVERSE_SURROUND"):
                a.source.epoch, a.target.epoch = t.Epoch(0), t.Epoch(3)
                if kind == "REVERSE_SURROUND":
                    a, b = b, a
            if kind == "NONCONFLICTING":
                b.target.epoch = t.Epoch(3)
            if kind == "IDENTICAL":
                b = a.copy()
        else:
            a = s.AttestationData(target=checkpoint(s, 5))
            b = s.AttestationData(target=checkpoint(s, 5, 99))
            if kind in ("CONFLICT_FORWARD", "CONFLICT_REVERSE", "TIMEOUT_CONFLICT"):
                b = s.AttestationData(target=checkpoint(s, 10), finality=checkpoint(s, 5, 99))
                if kind == "CONFLICT_REVERSE":
                    a, b = b, a
                if kind == "TIMEOUT_CONFLICT":
                    a.timeout = True
            elif kind == "TIMEOUTS":
                a.timeout = b.timeout = True
            elif kind == "NONCONFLICTING":
                b.target = checkpoint(s, 7)
            elif kind == "IDENTICAL":
                b = a.copy()
        indices = [1, 2] if solution.multiple else [1]
        first = t.IndexedAttestation(data=a, attesting_indices=t.AttestingIndices(data=indices))
        second = t.IndexedAttestation(data=b, attesting_indices=t.AttestingIndices(data=indices))
        if not self.legacy:
            first.epoch = second.epoch = s.Epoch(3)
        if fault == "DISJOINT":
            second.attesting_indices = t.AttestingIndices(data=[i + 3 for i in indices])
        if fault == "EMPTY_FIRST":
            first.attesting_indices = t.AttestingIndices()
        if fault == "UNSORTED_SECOND":
            second.attesting_indices = t.AttestingIndices(data=[2, 1])
        if fault == "ALREADY_SLASHED":
            for i in indices:
                pre.validators[i].slashed = True
        for j, att in enumerate((first, second)):
            domain = s.DOMAIN_BEACON_ATTESTER if self.legacy else s.DOMAIN_FINALITY_ATTESTER
            if fault == ("FIRST_SIGNATURE" if j == 0 else "SECOND_SIGNATURE"):
                domain = s.DOMAIN_FINALITY_ATTESTER if self.legacy else s.DOMAIN_BEACON_ATTESTER
            if att.attesting_indices:
                att.signature = sign(
                    s,
                    pre,
                    att.data,
                    domain,
                    att.data.target.epoch if self.legacy else att.epoch,
                    att.attesting_indices,
                )
        return pre, t.AttesterSlashing(attestation_1=first, attestation_2=second)


def observe(ctx, legacy):
    s, pre, op = ctx.spec, ctx.pre, ctx.operation
    a, b = op.attestation_1, op.attestation_2
    x, y = a.data, b.data
    if legacy:
        if x == y:
            kind = "IDENTICAL"
        elif x.target.epoch == y.target.epoch:
            kind = "DOUBLE"
        elif x.source.epoch < y.source.epoch and y.target.epoch < x.target.epoch:
            kind = "SURROUND"
        elif y.source.epoch < x.source.epoch and x.target.epoch < y.target.epoch:
            kind = "REVERSE_SURROUND"
        else:
            kind = "NONCONFLICTING"
    elif x == y:
        kind = "IDENTICAL"
    elif (
        not x.timeout
        and not y.timeout
        and x.target.height == y.target.height
        and x.target.root != y.target.root
    ):
        kind = "DOUBLE"
    elif (
        y.finality != s.HeightCheckpoint.empty()
        and x.target.height == y.finality.height
        and (x.timeout or x.target.root != y.finality.root)
    ):
        kind = "TIMEOUT_CONFLICT" if x.timeout else "CONFLICT_FORWARD"
    elif (
        x.finality != s.HeightCheckpoint.empty()
        and y.target.height == x.finality.height
        and (y.timeout or y.target.root != x.finality.root)
    ):
        kind = "CONFLICT_REVERSE"
    elif x.timeout and y.timeout:
        kind = "TIMEOUTS"
    else:
        kind = "NONCONFLICTING"
    fault = "NONE"
    if not a.attesting_indices:
        fault = "EMPTY_FIRST"
    elif list(b.attesting_indices) != sorted(set(b.attesting_indices)):
        fault = "UNSORTED_SECOND"
    else:
        domain = s.DOMAIN_BEACON_ATTESTER if legacy else s.DOMAIN_FINALITY_ATTESTER
        for label, att in (("FIRST_SIGNATURE", a), ("SECOND_SIGNATURE", b)):
            if not verify(
                s,
                pre,
                att.data,
                domain,
                att.data.target.epoch if legacy else att.epoch,
                att.attesting_indices,
                att.signature,
            ):
                fault = label
                break
        if fault == "NONE":
            shared = set(a.attesting_indices) & set(b.attesting_indices)
            if not shared:
                fault = "DISJOINT"
            elif not any(
                not pre.validators[i].slashed
                and pre.validators[i].activation_epoch
                <= int(pre.slot) // int(s.SLOTS_PER_EPOCH)
                < pre.validators[i].withdrawable_epoch
                for i in shared
            ):
                fault = "ALREADY_SLASHED"
    # Empty first / malformed second retain the unmodified side as the count witness.
    count = len(b.attesting_indices) if not a.attesting_indices else len(a.attesting_indices)
    return {"observed_kind": kind, "observed_fault": fault, "observed_multiple": count > 1}
