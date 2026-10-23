"""Observe body guards and operation-list composition."""


def observe_attributes(ctx):
    s, pre, operation = ctx.spec, ctx.pre, ctx.operation
    body = operation.message.body if hasattr(operation, "message") else operation
    slot = operation.message.slot if hasattr(operation, "message") else pre.slot
    votes, bits = body.head_votes, body.head_vote_support_bits
    if len(body.legacy_attester_slashings) > s.MAX_ATTESTER_SLASHINGS_ELECTRA:
        case = "OVER_LEGACY"
    elif len(votes) > s.MAX_HEAD_VOTES:
        case = "OVER_HEAD"
    elif len(bits) < len(votes):
        case = "SHORT_SUPPORT"
    elif len(bits) > len(votes):
        case = "LONG_SUPPORT"
    elif any(
        bit and vote.data.slot != slot - 1 for bit, vote in zip(bits, votes, strict=False)
    ):
        case = "OLD_SUPPORTED"
    elif votes and body.attestations and body.legacy_attester_slashings:
        case = "MIXED"
    elif votes:
        case = (
            "MAX_HEAD"
            if len(votes) == s.MAX_HEAD_VOTES
            else ("OLD_UNSUPPORTED" if votes[0].data.slot != slot - 1 else "HEAD")
        )
    elif body.legacy_attester_slashings:
        case = "LEGACY"
    elif body.attester_slashings:
        case = "FINALITY_SLASHING"
    elif body.attestations:
        case = "ATTESTATION"
    else:
        case = "EMPTY"
    return {"observed_case": case}
