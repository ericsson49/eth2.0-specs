from py_ecc.bls import G2ProofOfPossession as bls
from py_ecc.bls.g2_primatives import signature_to_G2 as _signature_to_G2

# Flag to make BLS active or not. Used for testing, do not ignore BLS in production unless you know what you are doing.
bls_active = True

STUB_SIGNATURE = b'\x11' * 96
STUB_PUBKEY = b'\x22' * 48
STUB_COORDINATES = _signature_to_G2(bls.Sign(0, b""))


def only_with_bls(alt_return=None,check=False):
    """
    Decorator factory to make a function only run when BLS is active. Otherwise return the default.
    """
    def runner(fn):
        def entry(*args, **kw):
            if bls_active:
                return fn(*args, **kw)
            else:
                if check:
                    return args[-1] != (b'\x00' * 96)
                else:
                    return alt_return
        return entry
    return runner

pub2priv = {}
def init():
    from eth2spec.test.helpers.keys import pubkey_to_privkey
    global pub2priv
    pub2priv = pubkey_to_privkey

verify_cache = {}
fa_verify_cache = {}
#sign_cache = {}

@only_with_bls(alt_return=True,check=True)
def Verify(PK, message, signature):
    global verify_cache
    k = PK.hex() + '_' + message.hex()
    if k not in verify_cache:
        verify_cache[k] = Sign(pub2priv[PK], message).hex()
    return verify_cache[k] == signature.hex()

@only_with_bls(alt_return=True,check=True)
def AggregateVerify(pairs, signature):
    return bls.AggregateVerify(pairs, signature)


@only_with_bls(alt_return=True,check=True)
def FastAggregateVerify(PKs, message, signature):
    global fa_verify_cache
    k = ('_'.join(sorted(PK.hex() for PK in PKs))) + '_' + message.hex()
    if k not in fa_verify_cache:
        fa_verify_cache[k] = bls.Aggregate([Sign(pub2priv[PK], message) for PK in PKs]).hex()
    return fa_verify_cache[k] == signature.hex()


@only_with_bls(alt_return=STUB_SIGNATURE)
def Aggregate(signatures):
    return bls.Aggregate(signatures)


@only_with_bls(alt_return=STUB_SIGNATURE)
def Sign(SK, message):
    #if not SK in sign_cache:
    #    sign_cache[SK] = bls.Sign(SK, message)
    #return sign_cache[SK]
    return bls.Sign(SK, message)


@only_with_bls(alt_return=STUB_COORDINATES)
def signature_to_G2(signature):
    return _signature_to_G2(signature)


@only_with_bls(alt_return=STUB_PUBKEY)
def AggregatePKs(pubkeys):
    return bls._AggregatePKs(pubkeys)
