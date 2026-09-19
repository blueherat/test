"""Binary GAN: real images versus actual guided endpoints; W fools D.

Manual gradient averaging is intentional: the discrete-adjoint autograd function
recomputes many head calls internally, and must not register DDP forward hooks.
All ranks use the same deterministic critic, equal local batches and one optimizer
update per globally averaged gradient. Only the MLP weak head is changed at inference.
"""

import argparse
import copy
import json
import time
import numpy as np
import torch
import torch.distributed as dist
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from experiments.guidance_dynamic_50k_20260915.sampling import integrate
from experiments.adversarial_guidance_endpoint_20260915.discrete_adjoint import recomputed_rollout
from experiments.adversarial_guidance_endpoint_20260915.sampler import GuidanceSteps
from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
from . import common as c
from .binary_critic import BinaryCritic, discriminator_loss as logistic_discriminator_loss, weak_loss
from .rgb_data import RawImageEndpointData
from .rendering import sit_images as decoded


DATA_PROTOCOL = 'real_rgb_official_continuous_v2'


def main(args):
    rank, world = c.setup()
    assert args.global_batch % world == 0
    count = args.global_batch // world
    run = c.ROOT / args.run
    request = c.read(run / "request.json")
    assert request['data_protocol'] == DATA_PROTOCOL
    for filename, digest in request["sources"].items():
        assert c.sha(filename) == digest, filename
    torch.manual_seed(args.seed)
    adapter = Adapter("sit_small")
    adapter.rt.vae.eval().requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())
    head = adapter.loaded_head(args.initial_head).requires_grad_(True)
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    critic = BinaryCritic().cuda().train()
    optimizer_w = torch.optim.Adam(head.parameters(), lr=args.lr_w, betas=(.9, .99))
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=args.lr_d, betas=(0., .99))
    data = RawImageEndpointData(args.seed + 1009 * rank)
    if rank == 0:
        c.atomic(run / 'training_data.json', dict(data.provenance, data_protocol=DATA_PROTOCOL))
    start = 0
    if not args.resume and rank == 0:
        # Preserved baseline batch is used only for an engineering parity check.
        bank = c.original.model_root('sit_small') / 'quality_inputs'
        noise_check = torch.from_numpy(np.array(np.load(bank / 'noise.npy', mmap_mode='r')[:8])).cuda()
        labels_check = torch.from_numpy(np.load(bank / 'labels.npy')[:8]).long().cuda()
        deployed, counts = integrate(adapter, head, args.initial_head, args.coefficient, noise_check, labels_check)
        tick = round(args.coefficient * 40)
        baseline_path = c.original.model_root('sit_small') / f'points/{args.initial_head}__c{tick:04d}/batches/00000.npz'
        with np.load(baseline_path) as baseline:
            np.testing.assert_array_equal(deployed.cpu().numpy(), baseline['latents'])
            np.testing.assert_array_equal(adapter.pixels(deployed), baseline['arr_0'])
        with torch.no_grad():
            check_steps = GuidanceSteps(adapter, head, labels_check, args.coefficient, method='real')
            actual = recomputed_rollout(check_steps, len(check_steps), noise_check, head.parameters())
        torch.testing.assert_close(actual, deployed, rtol=0, atol=0)
        c.atomic(run / 'initial_baseline_parity.json', dict(passed=True,
            baseline_batch=str(baseline_path), baseline_batch_sha256=c.sha(baseline_path),
            coefficient=args.coefficient, initial_head_fingerprint=fingerprint(head),
            same_latents=True, same_pixels=True, adjoint_forward_exact=True,
            counts=counts, checked_utc=c.now()))
        del noise_check, labels_check, deployed, actual, check_steps
        adapter.values.clear()
    c.barrier()
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert state['objective'] == 'binary_gan'
        assert state.get('data_protocol') == DATA_PROTOCOL, 'Use a checkpoint trained with the same real-image protocol'
        head.load_state_dict(state["head"])
        ema.load_state_dict(state["ema"])
        critic.load_state_dict(state["critic"])
        optimizer_w.load_state_dict(state["optimizer_w"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        for group in optimizer_w.param_groups:
            group["lr"] = args.lr_w
        for group in optimizer_d.param_groups:
            group["lr"] = args.lr_d
        start = int(state["step"])
        if rank < len(state['rngs']):
            data.load_state_dict(state["rngs"][rank])
        if rank == 0:
            c.atomic(run / 'rng_resume.json', dict(saved_world_size=len(state['rngs']), new_world_size=world,
                rank_sources=[f'saved_rank_{r}' if r < len(state['rngs']) else
                              f'new_independent_seed_{args.seed + 1009 * r}' for r in range(world)],
                global_stream_bitwise_preserved=len(state['rngs']) == world))
    else:
        # Calibrate fixed feature coordinates on random real TRAINING samples.
        # This is normalization only; subsequent training draws from all 126689.
        total = torch.zeros(2048, dtype=torch.float64, device="cuda")
        squares = torch.zeros_like(total)
        with torch.no_grad():
            for _ in range(16):
                real, _, _ = data.draw(count)
                values = feature(real).double()
                total += values.sum(0)
                squares += values.square().sum(0)
            if world > 1:
                dist.all_reduce(total)
                dist.all_reduce(squares)
            n = 16 * count * world
            mean = total / n
            std = (squares / n - mean.square()).clamp_min(0).sqrt().clamp_min(.05)
            critic.feature_mean.copy_(mean.float())
            critic.feature_std.copy_(std.float())
    initial = fingerprint(head)
    if rank == 0:
        c.atomic(run / 'initial_state.json', dict(step=start, head=initial,
            ema=fingerprint(ema), critic=fingerprint(critic),
            resumed_from=args.resume, data_protocol=DATA_PROTOCOL, checked_utc=c.now()))
        c.atomic(run / "progress.json", dict(phase="training", step=start,
            target_step=start + args.updates, world_size=world, global_batch=args.global_batch,
            updated_utc=c.now(), initial_head_fingerprint=initial))
    c.barrier()

    def save(step, phase):
        rngs = [None] * world
        signature = dict(head=fingerprint(head), ema=fingerprint(ema), critic=fingerprint(critic))
        signatures = [None] * world
        if world > 1:
            dist.all_gather_object(rngs, data.state_dict())
            dist.all_gather_object(signatures, signature)
        else:
            rngs[0] = data.state_dict()
            signatures[0] = signature
        assert all(value == signature for value in signatures), signatures
        if rank == 0:
            def cpu(module):
                return {name: value.detach().cpu() for name, value in module.state_dict().items()}
            state = dict(step=step, objective='binary_gan', data_protocol=DATA_PROTOCOL,
                head=cpu(head), ema=cpu(ema), critic=cpu(critic),
                optimizer_w=optimizer_w.state_dict(), optimizer_d=optimizer_d.state_dict(),
                rngs=rngs, request_sha256=c.sha(run / "request.json"), args=vars(args))
            path = run / f"checkpoint_{step:06d}.pt"
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)
            c.atomic(run / "latest.json", dict(phase=phase, step=step, checkpoint=str(path),
                checkpoint_sha256=c.sha(path), updated_utc=c.now(),
                replicas_identical=True, replica_signatures=signatures))
            print("saved", step, path, flush=True)
        c.barrier()

    final_step = start
    elapsed_start = time.perf_counter()
    for step in range(start + 1, start + args.updates + 1):
        should_stop = torch.tensor(int(c.stopped()), device="cuda")
        if world > 1:
            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
        if should_stop.item():
            save(step - 1, "paused")
            break
        begin = time.perf_counter()
        real, noise, labels = data.draw(count)
        with torch.no_grad():
            real_features = feature(real)
        active_generator = step > args.warmup
        steps = GuidanceSteps(adapter, head, labels, args.coefficient, method="real")
        if active_generator:
            generated = recomputed_rollout(steps, len(steps), noise, head.parameters())
            generated_features = feature(decoded(adapter, generated))
        else:
            with torch.no_grad():
                generated = recomputed_rollout(steps, len(steps), noise, head.parameters())
                generated_features = feature(decoded(adapter, generated))
        if not bool(torch.isfinite(generated_features).all()):
            raise FloatingPointError("Nonfinite endpoint features")

        critic.train().requires_grad_(True)
        optimizer_d.zero_grad(set_to_none=True)
        inputs = torch.cat([real_features, generated_features.detach()]).detach().requires_grad_(True)
        classes = labels.repeat(2)
        logits = critic(inputs, classes)
        classification = logistic_discriminator_loss(logits[:count], logits[count:])
        input_gradient = torch.autograd.grad(logits[:count].sum(),
                                             inputs, create_graph=True)[0][:count]
        penalty = input_gradient.square().sum(1).mean()
        discriminator_loss = classification + .5 * args.r1 * penalty
        discriminator_loss.backward()
        c.average_gradients(critic)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
        optimizer_d.step()

        critic.eval().requires_grad_(False)
        optimizer_w.zero_grad(set_to_none=True)
        new_logits = critic(generated_features, labels)
        generator_loss = weak_loss(new_logits)
        gradient_norm = 0.
        if active_generator:
            generator_loss.backward()
            c.average_gradients(head)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True))
            optimizer_w.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), head.parameters()):
                    target.lerp_(source, 1. - args.ema)
        real_accuracy = float((logits[:count] > 0).float().mean())
        fake_accuracy = float((logits[count:] < 0).float().mean())
        values = c.mean_values([float(classification.detach()), float(penalty.detach()),
            float(generator_loss.detach()), gradient_norm, real_accuracy, fake_accuracy,
            float(generated.detach().square().mean().sqrt())])
        torch.cuda.synchronize()
        duration = time.perf_counter() - begin
        row = dict(step=step, d_ce=values[0], r1=values[1], g_loss=values[2],
            head_gradient_norm=values[3], real_accuracy=values[4], fake_accuracy=values[5],
            endpoint_rms=values[6], seconds=duration,
            updated_utc=c.now(), generator_updated=active_generator,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
        if rank == 0:
            with (run / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            c.atomic(run / "progress.json", dict(row, phase="training", target_step=start + args.updates,
                world_size=world, global_batch=args.global_batch,
                elapsed_seconds=time.perf_counter() - elapsed_start))
            print(row, flush=True)
        final_step = step
        del generated, generated_features, real_features, inputs, logits, new_logits
        del discriminator_loss, generator_loss, classification, penalty, input_gradient
        adapter.values.clear()
        if step % args.save_every == 0:
            save(step, "training")
    else:
        save(final_step, "complete")
        if rank == 0:
            c.atomic(run / 'progress.json', dict(phase='complete', step=final_step, updated_utc=c.now()))
            c.atomic(run / "complete.json", dict(complete=True, step=final_step,
                initial_head_fingerprint=initial, final_head_fingerprint=fingerprint(head),
                request_sha256=c.sha(run / "request.json"), updated_utc=c.now()))
    c.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--updates", type=int, default=416)
    parser.add_argument("--global-batch", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--lr-w", type=float, default=1e-6)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--r1", type=float, default=1.)
    parser.add_argument("--ema", type=float, default=.99)
    parser.add_argument("--coefficient", type=float, default=1.05)
    parser.add_argument("--initial-head", default="guided_weak")
    parser.add_argument("--seed", type=int, default=2026091597)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume")
    main(parser.parse_args())
