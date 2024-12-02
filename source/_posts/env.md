```python
#(/root/miniconda3/envs/xDiT/lib/python3.10/site-packages/yunchang/kernels/attention.py)
# ORGINAL CODE
# try:
#     from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
#     HAS_FLASH_ATTN = True
# except ImportError:
#     HAS_FLASH_ATTN = False

# try:
#     from flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
#     from flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
#     from flash_attn_interface import flash_attn_func as flash3_attn_func
try:
    from flash_attn.flash_attn_interface import _flash_attn_forward as flash_attn_forward_hopper
    from flash_attn.flash_attn_interface import _flash_attn_backward as flash_attn_func_hopper_backward
    from flash_attn.flash_attn_interface import flash_attn_func as flash3_attn_func

```

```python
# torch_attn
def torch_attn(
    q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs
):
    batch_size, seq_len, hs, hd = q.size()
    query = q.view(batch_size, -1, hs, hd).transpose(1, 2)
    key = k.view(batch_size, -1, hs, hd).transpose(1, 2)
    value = v.view(batch_size, -1, hs, hd).transpose(1, 2)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=causal
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, hs, hd)
    hidden_states = hidden_states.to(query.dtype)
    return hidden_states
```