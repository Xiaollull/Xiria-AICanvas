# PixAI v1.0 static inference adapter — upstream attribution

`backend/pixai_tagger.py` is an inference-only adaptation of PixAI Labs'
`tagger_pipeline.py` at revision `9fe10addf9326e292da8a85a98ea74cd91b41771`:
https://huggingface.co/pixai-labs/pixai-tagger-v1.0

The upstream model card declares Apache-2.0. The architecture embeds Meta's SAM3
ViTDet implementation (https://github.com/facebookresearch/sam3/blob/main/sam3/model/vitdet.py),
whose SAM License, last updated November 19, 2025, is included as `SAM-LICENSE.txt`.
The derived architecture is **not described as solely Apache-2.0**. Redistribution
and use of the SAM-derived material are subject to that agreement.

The active MLP topology follows the `timm.layers.Mlp` interface (Ross Wightman and
contributors, Apache-2.0; https://github.com/huggingface/pytorch-image-models),
implemented locally with PyTorch layers; no timm source/package is bundled.
No remote Python, transformers `trust_remote_code`, or pickle is used by this adapter.
Only the fixed official safetensors/config/processor artifacts are downloaded.

This notice identifies upstream terms; it is not legal advice or a claim that
licensing obligations automatically do or do not propagate to other software.
