async def comfy_entrypoint():
    if __package__:
        from .comfyui_seqattn.nodes import comfy_entrypoint as load_extension
    else:
        from comfyui_seqattn.nodes import comfy_entrypoint as load_extension

    return await load_extension()

__all__ = ["comfy_entrypoint"]
