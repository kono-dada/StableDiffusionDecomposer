from dataclasses import dataclass
from diffusers import UNet2DConditionModel
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from utils import concat_img
from typing import Optional, List, Tuple, Dict


ATTN_BLOCKS = [
    'down_blocks.0.attentions.0.transformer_blocks.0.attn1',
    'down_blocks.0.attentions.0.transformer_blocks.0.attn2',
    'down_blocks.0.attentions.1.transformer_blocks.0.attn1',
    'down_blocks.0.attentions.1.transformer_blocks.0.attn2',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn2',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn2',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn2',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn2',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn2',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn2',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn2',
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.2.attentions.0.transformer_blocks.0.attn2',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn2',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn2',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn2',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn2',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn2',
    'mid_block.attentions.0.transformer_blocks.0.attn1',
    'mid_block.attentions.0.transformer_blocks.0.attn2',
]


def pca(x: torch.Tensor, target_dim: int):  # x: (n, d)
    _x = x.float()
    mean = _x .mean(dim=0)
    _x = _x - _x.mean(dim=0)
    U, S, V = torch.svd(_x)
    V_reduced = V[:, :target_dim]
    projected_x = torch.mm(_x, V_reduced)
    return projected_x


@dataclass
class QKV:
    t: int
    name: str
    q: Optional[torch.Tensor]
    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]
    a: Optional[torch.Tensor]
    guidance: bool

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in ['q', 'k', 'v', 'a']:
            return getattr(self, key)


class QKVRecordCallback:
    def __init__(
        self,
        p: StableDiffusionPipeline,
        attn_index: List[int],
        record_per_step: int,
        start: int,
        end: int,
        name: str,
        inverse: bool = False
    ):
        r"""
        Add this callback to the pipeline.__call__() to record the QKV of the attentions.
        
        Args:
            p (StableDiffusionPipeline): The pipeline.
            attn_index (List[int]): The indices of the attentions to record. The indices are defined in `ATTN_BLOCKS`.
            record_per_step (int): Record the QKV every `record_per_step` steps.
            start (int): Start recording at step `start`.
            end (int): Stop recording at step `end`.
            name (str): The name of the callback.
            inverse (bool): If it is a DDIM-inversion, set `inverse` to `True`.
        """
        self.attentions: Dict[Tuple[QKV]] = {}
        for attn_id in attn_index:
            self.attentions[ATTN_BLOCKS[attn_id]] = []
        self.last_rec = 10000
        self.record_per_step = record_per_step
        self.start = start
        self.end = end
        self.pipe = p
        self.name = name
        self.inverse = inverse
        self.timesteps = []

    def __call__(self, p: StableDiffusionPipeline, i: int, t, kwargs):
        if i % self.record_per_step == 0 and self.start <= t < self.end:
            unet: UNet2DConditionModel = p.unet
            modules = list(unet.named_modules())
            for name, module in modules:
                if name in self.attentions:
                    q, k, v, a = module.processor.record_QKV()
                    if 'attn1' in name:  # only record the attn_map for cross-attention, since self-attention is too large.
                        a = None
                    if self.inverse:
                        self.attentions[name].insert(
                            0, QKV(t, name, q, k, v, a, module.processor.guidance))
                    else:
                        self.attentions[name].append(
                            QKV(t, name, q, k, v, a, module.processor.guidance))
            if self.inverse:
                self.timesteps.insert(0, t)
            else:
                self.timesteps.append(t)
            self.last_rec = t
        return kwargs

    def qkv_visualization(self, attn_id: int, qkv: str, dim=3, chunk=1):
        r"""
        Visualize the QKV of the attentions by projecting them into a 3D (RGB) space.
        
        Args:
            attn_id (int): The index of the attention.
            qkv (str): The QKV to visualize. It can be 'q', 'k', 'v'.
            dim (int): The dimension of the projected space.
            chunk (int): The chunk of the QKV. If guidance_scale > 1, there will be two latents.
                The first chunk is the unconditional (negative prompt) latent, and the second chunk is the conditional latent.
        """
        img_list = []
        for attn in tqdm(self.attentions[ATTN_BLOCKS[attn_id]]):
            guidance = attn.guidance
            tgt = attn[qkv]  # (heads, h*w, c)
            h, n, c = tgt.shape
            if guidance:
                tgt = torch.chunk(tgt, 2, dim=0)[chunk]
            tgt = tgt.permute(1, 2, 0).reshape(n, -1)  # (h*w, c*heads)
            hxw = tgt.shape[-2]
            size = int(hxw ** 0.5)
            tgt = pca(tgt, dim)
            image = tgt.transpose(-1, -2).reshape(dim, size, size)  # (dim, h, w)
            image = (image - image.min()) / (image.max() - image.min())
            image = image.unsqueeze(0)
            img_list.append(image)
        plt.figure(figsize=(20, 10))
        plt.title(f'{self.name}: {qkv} of {ATTN_BLOCKS[attn_id]}')
        plt.imshow(concat_img(img_list))

    def attn_map_visualization(self, attn_id: int, pos: int, temperature, prompt: str = '', chunk=1):
        r"""
        Visualize the attention maps of the attentions.
        
        Args:
            attn_id (int): The index of the attention.
            pos (int): The position of the token in the prompt.
            temperature (float): The temperature of the attention map.
            prompt (str): The prompt.
            chunk (int): The chunk of the attention map. If guidance_scale > 1, there will be two latents.
                The first chunk is the unconditional (negative prompt) latent, and the second chunk is the conditional latent.
        """
        img_list = []
        tokens = self.pipe.tokenizer.tokenize(prompt)
        for attn in tqdm(self.attentions[ATTN_BLOCKS[attn_id]]):
            guidance = attn.guidance
            tgt = attn.a  # (heads, h*w, 77)
            if guidance:
                tgt = torch.chunk(tgt, 2, dim=0)[chunk]
            attention_weights = tgt[:, :, pos + 1]  # (heads, h*w)
            tgt = attention_weights.mean(dim=0, keepdim=True)  # (1, h*w)
            hxw = tgt.shape[-1]
            size = int(hxw ** 0.5)
            image = tgt.transpose(-1, -2).reshape(1, size, size)
            image = (image - image.min()) / (image.max() - image.min())
            image = image.unsqueeze(0)
            img_list.append(image)
        img_list.append(sum(img_list) / len(img_list))
        print(tokens)
        plt.figure(figsize=(20, 10))
        plt.title(
            f'{self.name}: Attention maps of {ATTN_BLOCKS[attn_id]} to token "{tokens[pos]}"')
        plt.imshow(concat_img(img_list))
