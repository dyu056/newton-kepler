"""Fix hook registration in kepler_cv_blocksize_mlp.py for MLP-only model."""
import sys

with open(sys.argv[1], "r") as f:
    content = f.read()

old = """    # Register hooks for each transformer block
    for block_idx in range(n_layer):
        block = model.transformer.h[block_idx]

        # 1. Attention output before merge
        hook = block.attn.register_forward_hook(make_attn_output_hook(block_idx))
        hooks.append(hook)

        # 2. Residual after attention merge (input to MLP)
        hook = block.mlp.register_forward_pre_hook(make_after_attn_merge_hook(block_idx))
        hooks.append(hook)

        # 3. MLP output before merge
        hook = block.mlp.register_forward_hook(make_mlp_output_hook(block_idx))
        hooks.append(hook)

        # 4. Residual after MLP merge (block output)
        hook = block.register_forward_hook(make_after_mlp_merge_hook(block_idx))
        hooks.append(hook)

        # 5. MLP hidden activation after silu
        hook = block.mlp.silu.register_forward_hook(make_mlp_hidden_hook(block_idx))
        hooks.append(hook)"""

new = """    # Register hooks for each transformer block (MLP-only: no attention)
    for block_idx in range(n_layer):
        block = model.transformer.h[block_idx]
        has_attn = hasattr(block, "attn")

        # 1. Pre-MLP representation (LN output since no attention)
        if has_attn:
            hook = block.attn.register_forward_hook(make_attn_output_hook(block_idx))
        else:
            hook = block.ln.register_forward_hook(make_attn_output_hook(block_idx))
        hooks.append(hook)

        # 2. Residual after merge (input to MLP)
        hook = block.mlp.register_forward_pre_hook(make_after_attn_merge_hook(block_idx))
        hooks.append(hook)

        # 3. MLP output before merge
        hook = block.mlp.register_forward_hook(make_mlp_output_hook(block_idx))
        hooks.append(hook)

        # 4. Residual after MLP merge (block output)
        hook = block.register_forward_hook(make_after_mlp_merge_hook(block_idx))
        hooks.append(hook)

        # 5. MLP hidden activation after activation function
        if hasattr(block.mlp, "silu"):
            hook = block.mlp.silu.register_forward_hook(make_mlp_hidden_hook(block_idx))
            hooks.append(hook)"""

if old in content:
    content = content.replace(old, new)
    with open(sys.argv[1], "w") as f:
        f.write(content)
    print("Fixed!")
else:
    print("ERROR: could not find hook block")
    sys.exit(1)
