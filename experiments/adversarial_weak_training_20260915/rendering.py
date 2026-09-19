"""Continuous image input aligned with the deployed SiT pixel conversion.

The exported image additionally truncates to uint8. Training retains continuous
values so gradients exist; both paths share the same offset and range clipping.
This module is prepared for a separate subsequent binary run and is not imported
by endpoint_binary_gan_v1's recorded source snapshot.
"""

from experiments import train_imagenet100_sit_flow as base


def sit_images(adapter, latents):
    decoded = adapter.rt.vae.decode(
        latents.float() / base.SD_VAE_SCALING_FACTOR
    ).sample
    return (127.5 * decoded + 128.0).clamp(0.0, 255.0) / 255.0
