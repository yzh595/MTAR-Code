"""Load the standard NTP decoder from an MTAR++ training checkpoint."""


def load_ntp_weights(model, state):
    target = model.state_dict()
    missing = sorted(set(target) - set(state))
    wrong_shapes = sorted(
        key for key in target if key in state and state[key].shape != target[key].shape
    )
    auxiliary = set(state) - set(target)
    allowed = ("projectors.", "future_feature_heads.", "contrast_head.")
    unexpected = sorted(key for key in auxiliary if not key.startswith(allowed))
    if missing or wrong_shapes or unexpected:
        raise RuntimeError(
            f"Incompatible MTAR++ checkpoint: missing={missing}, "
            f"wrong_shapes={wrong_shapes}, unexpected={unexpected}"
        )
    model.load_state_dict({key: state[key] for key in target}, strict=True)
    print(f"Loaded all {len(target)} NTP tensors; ignored {len(auxiliary)} auxiliary tensors.")
