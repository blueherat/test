"""SiT-native Transformer adapters initialized from the frozen model's tail."""
import copy
from torch import nn


class SiTAdapterHead(nn.Module):
    def __init__(self, model, blocks):
        super().__init__()
        if blocks not in (0, 1, 2): raise ValueError('Use zero, one or two blocks')
        if model.in_channels != 4 or model.patch_size != 2 or model.out_channels not in (4, 8):
            raise ValueError('This experiment requires the SiT-S/2 velocity checkpoint')
        self.output_channels = model.out_channels
        tail = list(model.blocks[-blocks:]) if blocks else []
        self.blocks = nn.ModuleList(copy.deepcopy(tail))
        self.final = copy.deepcopy(model.final_layer)
        # Adapter instrumentation belongs to the source backbone, not its copies.
        for module in self.modules():
            module._forward_pre_hooks.clear()
            module._forward_hooks.clear()
            module._backward_hooks.clear()
        self.copied_layers = list(range(len(model.blocks)-blocks+1, len(model.blocks)+1))
        self.requires_grad_(True)

    def forward(self, tokens, condition):
        for block in self.blocks:
            tokens = block(tokens, condition)
        result = self.final(tokens, condition)
        # The official checkpoint retains a sigma half, discarded by SiT.forward.
        # Channels are interleaved within each patch pixel; slicing [:16] is wrong.
        if self.output_channels == 8:
            result = result.reshape(*result.shape[:2], 4, 8)[..., :4].reshape(*result.shape[:2], 16)
        return result

    def architecture(self):
        return dict(kind='native_sit_transformer_adapter', input_width=384, hidden_width=384,
            transformer_blocks=len(self.blocks), copied_layers=self.copied_layers,
            initialization='pretrained tail blocks and pretrained final layer',
            native_projection_channels=self.output_channels, velocity_channels=4,
            parameters=sum(p.numel() for p in self.parameters()), depth=4,
            output='16 values per patch, unpatchified to 4x32x32 velocity')
