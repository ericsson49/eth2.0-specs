from dataclasses import dataclass
from enum import auto, Enum


class WithdrawalCredentialType(Enum):
    BLS = auto()
    ETH1 = auto()
    COMP = auto()
    UNKNOWN = auto()


class ComparisonOp(Enum):
    LT = auto()
    EQ = auto()
    GT = auto()


class ValidatorStateBranchTarget(Enum):
    DEPOSIT_REQUEST_START_INDEX_UNSET = auto()
    DEPOSIT_REQUEST_START_INDEX_SET = auto()
    REGISTRY_NO_CHANGE = auto()
    REGISTRY_ACTIVATION_QUEUE = auto()
    REGISTRY_EJECTION = auto()
    REGISTRY_ACTIVATION = auto()
    SLASHINGS_NO_SLASHED_VALIDATORS = auto()
    SLASHINGS_PENALTY_APPLIED = auto()
    SLASHINGS_WRONG_WITHDRAWABLE_EPOCH = auto()
    SLASHINGS_ZERO_SLASHING_BALANCE = auto()
    EFFECTIVE_BALANCE_NO_CHANGE_AT_THRESHOLD = auto()
    EFFECTIVE_BALANCE_STEP_DOWN = auto()
    EFFECTIVE_BALANCE_STEP_UP = auto()
    EFFECTIVE_BALANCE_CAP_AT_MAX = auto()


@dataclass
class ValidatorStateProfile:
    branch_target: ValidatorStateBranchTarget

    # Lifecycle
    activation_eligibility_epoch_set: bool
    activation_eligibility_epoch_finalized: bool
    activation_epoch_set: bool
    exit_epoch_set: bool
    withdrawable_epoch_set: bool
    slashed: bool
    activation_epoch_gt_activation_eligibility_epoch: bool
    withdrawable_epoch_gt_exit_epoch: bool
    shard_committee_period_lte_current_epoch: bool
    activation_epoch_to_current_epoch: ComparisonOp
    exit_epoch_to_current_epoch: ComparisonOp
    withdrawable_epoch_to_current_epoch: ComparisonOp

    # Balance
    balance_is_zero: bool
    balance_to_effective_balance: ComparisonOp
    effective_balance_lte_ejection_balance: bool
    effective_balance_to_min_activation_balance: ComparisonOp
    effective_balance_to_max_effective_balance: ComparisonOp

    # Credential type
    withdrawal_credential_type: WithdrawalCredentialType

    # Operations
    has_pending_withdrawal_request: bool
    excess_balance_gt_pending_withdrawal_balance: bool
    has_pending_consolidation_request: bool


p: ValidatorStateProfile = ...


# Lifecycle
if p.slashed:
    p.exit_epoch_set

if p.exit_epoch_set:
    p.slashed or p.shard_committee_period_lte_current_epoch

if p.shard_committee_period_lte_current_epoch:
    p.activation_epoch_to_current_epoch == ComparisonOp.LT

if p.activation_epoch_set:
    p.activation_eligibility_epoch_finalized

if p.activation_eligibility_epoch_finalized:
    p.activation_eligibility_epoch_set

p.activation_eligibility_epoch_set == p.activation_epoch_gt_activation_eligibility_epoch
p.exit_epoch_set == p.withdrawable_epoch_set
p.withdrawable_epoch_set == p.withdrawable_epoch_gt_exit_epoch
(p.withdrawable_epoch_to_current_epoch in {ComparisonOp.LT, ComparisonOp.EQ}) == (
    p.withdrawable_epoch_set
)

if p.withdrawable_epoch_to_current_epoch in {ComparisonOp.LT, ComparisonOp.EQ}:
    p.exit_epoch_to_current_epoch == ComparisonOp.LT

if p.withdrawable_epoch_to_current_epoch in {ComparisonOp.LT, ComparisonOp.EQ}:
    p.shard_committee_period_lte_current_epoch

if p.exit_epoch_to_current_epoch in {ComparisonOp.LT, ComparisonOp.EQ}:
    p.exit_epoch_set

if p.exit_epoch_to_current_epoch in {ComparisonOp.LT, ComparisonOp.EQ}:
    p.activation_epoch_to_current_epoch == ComparisonOp.LT

if p.activation_epoch_to_current_epoch in {ComparisonOp.LT, ComparisonOp.EQ}:
    p.activation_epoch_set


# Balance
if p.balance_is_zero:
    p.balance_to_effective_balance in {ComparisonOp.LT, ComparisonOp.EQ}

if p.balance_is_zero and p.balance_to_effective_balance == ComparisonOp.EQ:
    p.effective_balance_lte_ejection_balance

if p.effective_balance_lte_ejection_balance:
    p.effective_balance_to_min_activation_balance == ComparisonOp.LT

p.effective_balance_to_max_effective_balance in {ComparisonOp.LT, ComparisonOp.EQ}

if p.withdrawal_credential_type != WithdrawalCredentialType.COMP:
    p.effective_balance_to_min_activation_balance == p.effective_balance_to_max_effective_balance

if (
    p.withdrawal_credential_type == WithdrawalCredentialType.COMP
    and p.effective_balance_to_min_activation_balance in {ComparisonOp.LT, ComparisonOp.EQ}
):
    p.effective_balance_to_max_effective_balance == ComparisonOp.LT

if p.balance_is_zero:
    p.slashed or (
        p.withdrawable_epoch_to_current_epoch in {ComparisonOp.LT, ComparisonOp.EQ}
        and (
            p.withdrawal_credential_type == WithdrawalCredentialType.ETH1
            or p.withdrawal_credential_type == WithdrawalCredentialType.COMP
        )
    )


# Operations
if p.has_pending_withdrawal_request:
    p.withdrawal_credential_type == WithdrawalCredentialType.COMP

if p.has_pending_withdrawal_request and not p.slashed:
    not p.exit_epoch_set

if p.has_pending_consolidation_request:
    p.withdrawal_credential_type in {
        WithdrawalCredentialType.ETH1,
        WithdrawalCredentialType.COMP,
    } and p.exit_epoch_set

if p.has_pending_consolidation_request:
    not p.has_pending_withdrawal_request

if p.has_pending_withdrawal_request:
    not p.has_pending_consolidation_request

if p.excess_balance_gt_pending_withdrawal_balance:
    p.has_pending_withdrawal_request
    p.effective_balance_to_min_activation_balance in {ComparisonOp.GT, ComparisonOp.EQ}
    p.balance_to_effective_balance == ComparisonOp.GT
