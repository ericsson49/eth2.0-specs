This is a fork choice tests generator and runner.
It's not (yet) compatible with beacon chain test generators requirements/guidelines.

## Setup

Regular setup for running pyspec tests is assumed, i.e. `make install_test` (see [here](https://github.com/ericsson49/eth2.0-specs/blob/v012x_fork_choice_tests/tests/core/pyspec/README.md) for details).


## Test generation

Test generation is painfully slow currently, because of BLS.
Replacing [../../core/pyspec/eth2spec/utils/bls.py](../../core/pyspec/eth2spec/utils/bls.py) with [../../core/pyspec/eth2spec/utils/caching_bls.py](../../core/pyspec/eth2spec/utils/caching_bls.py) is recommended, if one wants to generate tests.
One should re-install the python lib, afterwards.
But it should work with the normal uncached BLS too.

Command to generate tests:
```
(venv)$ python fc_test_suite.py <test_output_dir>
```

or

```
(venv)$ python fc_test_suite.py
```

Which will output tests to `integration_tests` directory.

## Using existing tests

Fork choice tests for `v012x` are already generated and can be obtained from here https://github.com/harmony-dev/eth2.0-spec-tests/tree/fork_choice_integration_tests_v012x/tests/minimal/phase0/fork_choice/integration_tests .

## Tests running

To run tests against the `pyspec` (phase0), run

```
(venv)$ python test_exec.py <fc_test_dir>
```

or just

```
(venv)$ python test_exec.py
```

which is equivalent to `python test_exec.py integration_tests`.

For example, if `fork_choice_integration_tests_v012x` tests are put into `../../../../eth2.0-spec-tests/` directory, then one can run:
```
(venv)$ python test_exec.py ../../../../eth2.0-spec-tests/tests/minimal/phase0/fork_choice/integration_tests/
```

