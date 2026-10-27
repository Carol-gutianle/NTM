"""Narrow compatibility fix for the project's pinned PEFT/Transformers pair."""
from importlib.metadata import version


def patch_ddp_adapter_reload():
    if (version("peft"), version("transformers")) != ("0.19.1", "4.51.3"):
        return
    # PEFT 0.19.1 calls its TP helper whenever distributed is initialized,
    # including plain DDP. That helper imports TP symbols absent in 4.51.3
    # BEFORE it checks whether any layer actually uses tensor parallelism.
    # Avoid only that inapplicable TP branch; never change adapter state loading.
    from peft.utils import save_and_load
    original = save_and_load._maybe_shard_state_dict_for_tp
    if getattr(original, "_ddp_compat", False):
        return

    def only_for_actual_tp(model, state_dict, adapter_name):
        for module in model.modules():
            if getattr(module, "_hf_device_mesh", None) is not None:
                raise RuntimeError(
                    "This pinned PEFT/Transformers compatibility fix supports DDP, "
                    "not tensor parallelism. Use a tested TP-compatible environment."
                )
        # No device mesh: every rank has full adapters; no sharding is needed.

    only_for_actual_tp._ddp_compat = True
    save_and_load._maybe_shard_state_dict_for_tp = only_for_actual_tp
