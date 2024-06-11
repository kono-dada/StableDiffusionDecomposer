from dataclasses import dataclass
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTokenizer
import torch
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


def pca(x: torch.Tensor, target_dim: int):  # x: (n, d)
    _x = x.float()
    mean = _x .mean(dim=0)
    _x = _x - _x.mean(dim=0)
    U, S, V = torch.svd(_x)
    V_reduced = V[:, :target_dim]
    projected_x = torch.mm(_x, V_reduced)
    return projected_x


def pca_visualization(x: torch.Tensor, dim=3, chunk=1, name=''):
    x = x[chunk] if x.shape[0] > 1 else x[0]
    c, h, w = x.shape
    x = x.reshape(x.shape[0], -1).permute(1, 0)  # (h*w, c)
    x = pca(x, dim)
    image = x.transpose(1, 0).reshape(dim, h, w)  # (dim, h, w)
    image = (image - image.min()) / (image.max() - image.min())
    plt.figure(figsize=(5, 5))
    plt.title(f'{name}: PCA of {x.shape}')
    plt.imshow(concat_img([image]))
            
    
def qkv_visualization(attentions: Dict[str, Tuple[QKV]], attn_id: int, qkv: str, dim=3, chunk=1, name=''):
    img_list = []
    for attn in tqdm(attentions[ATTN_BLOCKS[attn_id]]):
        guidance = attn.guidance
        tgt = attn[qkv]  # (heads, h*w, c)
        h, n, c = tgt.shape
        if guidance:
            tgt = torch.chunk(tgt, 2, dim=0)[chunk]
        tgt = tgt.permute(1, 2, 0).reshape(n, -1)  # (h*w, c*heads)
        hxw = tgt.shape[-2]
        size = int(hxw ** 0.5)  # for now, only support square image
        tgt = pca(tgt, dim)
        image = tgt.transpose(-1, -2).reshape(dim, size, size)  # (dim, h, w)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.unsqueeze(0)
        img_list.append(image)
    plt.figure(figsize=(20, 10))
    plt.title(f'{name}: {qkv} of {ATTN_BLOCKS[attn_id]}')
    plt.imshow(concat_img(img_list))


def attn_map_visualization(
    attentions: Dict[str, Tuple[QKV]],
    tokenizer: CLIPTokenizer,
    attn_id: int,
    pos: int,
    temperature,
    prompt: str = '',
    chunk=1,
    name=''
):
    img_list = []
    tokens = tokenizer.tokenize(prompt)
    for attn in tqdm(attentions[ATTN_BLOCKS[attn_id]]):
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
        f'{name}: Attention maps of {ATTN_BLOCKS[attn_id]} to token "{tokens[pos]}"')
    plt.imshow(concat_img(img_list))


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
                    # only record the attn_map for cross-attention, since self-attention is too large.
                    if 'attn1' in name:
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
        qkv_visualization(self.attentions, attn_id, qkv, dim, chunk, self.name)

    def attn_map_visualization(self, pos: int, temperature, prompt: str = '', chunk=1):
        r"""
        Visualize the attention maps of the attentions.

        Args:
            pos (int): The position of the token in the prompt.
            temperature (float): The temperature of the attention map.
            prompt (str): The prompt.
            chunk (int): The chunk of the attention map. If guidance_scale > 1, there will be two latents.
                The first chunk is the unconditional (negative prompt) latent, and the second chunk is the conditional latent.
        """
        attn_map_visualization(
            self.attentions, self.pipe.tokenizer, pos, temperature, prompt, chunk, self.name)


class ControlNetRecordCallback:
    def __init__(self, p: StableDiffusionPipeline, record_per_step: int=1, start: int=0, end: int=1000, name: str=''):
        r"""
        Add this callback to the pipeline.__call__() to record the QKV of the control nets.

        Args:
            p (StableDiffusionPipeline): The pipeline.
            record_per_step (int): Record the QKV every `record_per_step` steps.
            start (int): Start recording at step `start`.
            end (int): Stop recording at step `end`.
            name (str): The name of the callback.
        """
        self.controls = []
        self.last_rec = 10000
        self.record_per_step = record_per_step
        self.start = start
        self.end = end
        self.pipe = p
        self.name = name
        self.timesteps = []

    def __call__(self, p: StableDiffusionPipeline, i: int, t, kwargs):
        if i % self.record_per_step == 0 and self.start <= t < self.end:
            down_block_res_samples, mid_block_res_sample = kwargs[
                'down_block_res_samples'], kwargs['mid_block_res_sample']
            self.controls.append([down_block_res_samples, mid_block_res_sample])
            self.timesteps.append(t)
            self.last_rec = t
        return kwargs
    
    
