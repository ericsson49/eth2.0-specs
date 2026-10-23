"""State-only EIP-8453 epoch vector serialization."""

from tests.generators.compliance_runners.state_transition.materializer import Materializer


class EpochMaterializer(Materializer):
    runner_name = "epoch_processing"

    def __init__(self, spec, fork_name="eip8453", preset_name="minimal", seed=0):
        if spec.fork != "eip8453" or fork_name != spec.fork:
            raise ValueError("epoch materializer requires the EIP-8453 fork")
        super().__init__(spec, fork_name, preset_name, seed)

    def materialize_solution(self, solution):
        pre = self.build_pre_state(solution)
        post = pre.copy()
        getattr(self.spec, f"process_{self.handler_name}")(post)
        return {
            "description": f"process_{self.handler_name}",
            "claimed": self.claimed_record(self.target, solution),
        }, [("pre", "ssz", pre.encode_bytes()), ("post", "ssz", post.encode_bytes())]
