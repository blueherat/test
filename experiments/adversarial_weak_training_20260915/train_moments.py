"""Full-sampler weak training with current-critic Fréchet feature matching.

Manual gradient averaging is intentional: the discrete-adjoint autograd function
recomputes many head calls internally, and must not register DDP forward hooks.
All ranks use the same deterministic critic, equal local batches and one optimizer
update per globally averaged gradient. Only the MLP weak head is changed at inference.
"""

import argparse
import copy
import json
from pathlib import Path
import time
import torch
import torch.distributed as dist
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from experiments.guidance_dynamic_50k_20260915.endpoints import weak_integrate
from experiments.adversarial_guidance_endpoint_20260915.discrete_adjoint import recomputed_rollout
from experiments.adversarial_guidance_endpoint_20260915.sampler import GuidanceSteps
from experiments.adversarial_guidance_endpoint_20260915.objectives import source_discriminator_loss
from experiments.advfd_cleanroom.feature_extractors import DifferentiableInception2048
from . import common as c
from .critic import SourceCritic
from .data import EndpointData, decoded
from .moment_objective import MomentFeedback
from .joint_moment_objective import JointMomentFeedback


def main(args):
    rank, world = c.setup()
    assert args.global_batch % world == 0
    count = args.global_batch // world
    run = c.ROOT / args.run
    request = c.read(run / "request.json")
    for filename, digest in request["sources"].items():
        assert c.sha(filename) == digest, filename
    torch.manual_seed(args.seed)
    adapter = Adapter("sit_small")
    adapter.rt.vae.eval().requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in adapter.model.parameters())
    head = adapter.loaded_head(args.initial_head).requires_grad_(True)
    ema = copy.deepcopy(head).eval().requires_grad_(False)
    feature = DifferentiableInception2048().cuda().eval().requires_grad_(False)
    critic = SourceCritic().cuda().train()
    optimizer_w = torch.optim.Adam(head.parameters(), lr=args.lr_w, betas=(.9, .99))
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=args.lr_d, betas=(0., .99))
    data = EndpointData(args.seed + 1009 * rank)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
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
        if rank < len(state["rngs"]):
            data.load_state_dict(state["rngs"][rank])
        if rank == 0:
            c.atomic(run / 'rng_resume.json', dict(saved_world_size=len(state['rngs']),
                new_world_size=world,
                rank_sources=[f'saved_rank_{r}' if r < len(state['rngs']) else
                              f'new_independent_seed_{args.seed + 1009 * r}' for r in range(world)],
                global_stream_bitwise_preserved=len(state['rngs']) == world,
                note='Existing ranks resume their saved streams; added ranks use distinct seeds'))
    else:
        # Calibrate fixed feature coordinates on random real TRAINING samples.
        # This is normalization only; subsequent training draws from all 126689.
        total = torch.zeros(2048, dtype=torch.float64, device="cuda")
        squares = torch.zeros_like(total)
        with torch.no_grad():
            for _ in range(16):
                real, _, _, _ = data.draw(count)
                values = feature(decoded(adapter, real)).double()
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
    feedback = (JointMomentFeedback(capacity=args.moment_memory, classes=adapter.cfg['classes'])
                if args.conditional_moments else MomentFeedback(capacity=args.moment_memory))
    assert args.moment_memory >= args.global_batch
    if args.resume and "moment_feedback" in state and not args.reset_moment_memory:
        feedback.load_state_dict(state["moment_feedback"])
    if feedback.size < feedback.capacity:
        # Independent fill RNG preserves the resumed optimizer-step data stream.
        fill_data = EndpointData(args.seed + 77777 + 1009 * rank)
        critic.eval().requires_grad_(False)
        with torch.no_grad():
            while feedback.size < feedback.capacity:
                should_stop = torch.tensor(int(c.stopped()), device="cuda")
                if world > 1:
                    dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
                if should_stop.item():
                    raise InterruptedError("Stopped during moment-memory fill before any new update")
                real_fill, _, noise_fill, labels_fill = fill_data.draw(count)
                fill_steps = GuidanceSteps(adapter, head, labels_fill, args.coefficient, method="real")
                generated_fill = recomputed_rollout(fill_steps, len(fill_steps), noise_fill, head.parameters())
                feedback.append(feature(decoded(adapter, real_fill)), feature(decoded(adapter, generated_fill)), labels_fill)
                if rank == 0:
                    c.atomic(run / "progress.json", dict(phase="filling_moment_memory",
                        step=start, memory_size=feedback.size, memory_capacity=feedback.capacity,
                        updated_utc=c.now()))
                adapter.values.clear()
        del fill_data, real_fill, noise_fill, labels_fill, generated_fill, fill_steps
    initial = fingerprint(head)
    if rank == 0:
        c.atomic(run / "progress.json", dict(phase="training", step=start,
            target_step=start + args.updates, world_size=world, global_batch=args.global_batch,
            updated_utc=c.now(), initial_head_fingerprint=initial))
    c.barrier()

    def save(step, phase):
        rngs = [None] * world
        signature = dict(head=fingerprint(head), critic=fingerprint(critic),
            real_memory=c.original.array_sha(feedback.real.cpu().numpy()),
            generated_memory=c.original.array_sha(feedback.generated.cpu().numpy()))
        if args.conditional_moments:
            signature['label_memory'] = c.original.array_sha(feedback.labels.cpu().numpy())
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
            state = dict(step=step, head=cpu(head), ema=cpu(ema), critic=cpu(critic),
                optimizer_w=optimizer_w.state_dict(), optimizer_d=optimizer_d.state_dict(),
                rngs=rngs, moment_feedback=feedback.state_dict(),
                request_sha256=c.sha(run / "request.json"), args=vars(args))
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
        real, strong, noise, labels = data.draw(count)
        with torch.no_grad():
            weak = weak_integrate(adapter, head, noise, labels)
            detached_features = [feature(decoded(adapter, value)) for value in (real, strong, weak)]
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
        inputs = torch.cat([*detached_features, generated_features.detach()]).detach().requires_grad_(True)
        classes = labels.repeat(4)
        sources = torch.arange(4, device="cuda").repeat_interleave(count)
        logits = critic(inputs, classes)
        classification = source_discriminator_loss(logits, sources)
        input_gradient = torch.autograd.grad((logits[:count, 3] - logits[:count, 0]).sum(),
                                             inputs, create_graph=True)[0][:count]
        penalty = input_gradient.square().sum(1).mean()
        discriminator_loss = classification + .5 * args.r1 * penalty
        discriminator_loss.backward()
        c.average_gradients(critic)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10., error_if_nonfinite=True)
        optimizer_d.step()

        critic.eval().requires_grad_(False)
        optimizer_w.zero_grad(set_to_none=True)
        feedback.append(detached_features[0], generated_features, labels)
        generator_loss, moment_values = feedback.loss(critic, generated_features)
        gradient_norm = 0.
        if active_generator:
            generator_loss.backward()
            c.average_gradients(head)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True))
            optimizer_w.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), head.parameters()):
                    target.lerp_(source, 1. - args.ema)
        accuracy = float((logits.argmax(1) == sources).float().mean())
        real_pair = float((logits[:count, 0] > logits[:count, 3]).float().mean())
        guided_pair = float((logits[3 * count:, 3] > logits[3 * count:, 0]).float().mean())
        values = c.mean_values([float(classification.detach()), float(penalty.detach()),
            float(generator_loss.detach()), gradient_norm, accuracy, real_pair, guided_pair,
            float(generated.detach().square().mean().sqrt())])
        torch.cuda.synchronize()
        duration = time.perf_counter() - begin
        row = dict(step=step, d_ce=values[0], r1=values[1], g_loss=values[2],
            head_gradient_norm=values[3], source_accuracy=values[4], real_pair_accuracy=values[5],
            guided_pair_accuracy=values[6], endpoint_rms=values[7], seconds=duration,
            updated_utc=c.now(), generator_updated=active_generator,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3, **moment_values)
        if rank == 0:
            with (run / "train.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            c.atomic(run / "progress.json", dict(row, phase="training", target_step=start + args.updates,
                world_size=world, global_batch=args.global_batch,
                elapsed_seconds=time.perf_counter() - elapsed_start))
            print(row, flush=True)
        final_step = step
        del generated, generated_features, detached_features, inputs, logits
        del discriminator_loss, generator_loss, classification, penalty, input_gradient
        adapter.values.clear()
        if step % args.save_every == 0:
            save(step, "training")
    else:
        save(final_step, "complete")
        if rank == 0:
            c.atomic(run / "progress.json", dict(phase="complete", step=final_step, updated_utc=c.now()))
            c.atomic(run / "complete.json", dict(complete=True, step=final_step,
                initial_head_fingerprint=initial, final_head_fingerprint=fingerprint(head),
                request_sha256=c.sha(run / "request.json"), updated_utc=c.now()))
    c.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--moment-memory", type=int, default=2048)
    parser.add_argument("--conditional-moments", action="store_true")
    parser.add_argument("--reset-moment-memory", action="store_true")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--global-batch", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--lr-w", type=float, default=1e-6)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--r1", type=float, default=1.)
    parser.add_argument("--ema", type=float, default=.99)
    parser.add_argument("--coefficient", type=float, default=1.)
    parser.add_argument("--initial-head", default="real")
    parser.add_argument("--seed", type=int, default=2026091597)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--resume")
    main(parser.parse_args())
