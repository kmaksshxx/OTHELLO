from src.self_play.self_play import *
from src.buffer import *
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig

saved_path = ROOT / 'checkpoint' / 'checkpoint.tar'
state_path = ROOT / 'checkpoint' / 'checkpoint_state.tar'


def save_checkpoint(model, best_model, optimizer, elo_tracker):
    torch.save({
        "model": model.state_dict(),
        "best_model": best_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "elo": elo_tracker.state_dict()
    }, saved_path)


def save_state(buffer: ReplayBuffer):
    state = buffer.return_state()
    torch.save({'state': state}, state_path)


def load_checkpoint():
    if DEVICE == 'cpu':
        ck = torch.load(saved_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        ck = torch.load(saved_path, weights_only=False)

    return ck


def load_state():
    ck = torch.load(state_path, weights_only=False)
    return ck['state']


def train_step(
        model: OthelloResNet, optimizer,
        replay_buffer: ReplayBuffer,
        value_coef, clip_grad):
    if len(replay_buffer) < BATCH_SIZE:
        return None

    states, pis, zs = replay_buffer.sample()
    policy_logits, values = model(states)  # (B, 65), (B, 1)

    log_p = F.log_softmax(policy_logits, dim=1)
    pl = - torch.mean(torch.sum(pis * log_p, dim=1))
    vl = torch.mean((values.squeeze(1) - zs) ** 2)
    loss = pl + value_coef * vl

    optimizer.zero_grad()
    loss.backward()
    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    return {"loss": loss.item(), "pl": pl.item(), "vl": vl.item()}


def train_with_mcts(
    best_model: OthelloResNet, model: OthelloResNet,
    replay_buffer: ReplayBuffer,
    optimizer, elo_agent: EloAgent,
    train_steps_per_iter, value_coef, clip_grad,
    eval_interval, n_games, goal_elo, n_workers,
    timer=None
):
    BEST_ID = "best"
    RANDOM_ID = "random"
    it = 0

    while True:
        it += 1
        CURRENT_ID = f"iteration_{it}"
        if timer:
            timer.reset(CURRENT_ID)

        train_stats = []

        with timed(timer, 'generate_self_play'):
            data, _ = generate_self_play(best_model, n_workers=n_workers)
            for own, opp, pi, z, _ in data:
                replay_buffer.add(own, opp, pi, z)

        for _ in range(train_steps_per_iter):
            with timed(timer, 'train_step'):
                out = train_step(model, optimizer, replay_buffer, value_coef, clip_grad)
                if out is not None:
                    train_stats.append(out)

        pl = np.mean([s["pl"] for s in train_stats])
        pl_std = np.std([s["pl"] for s in train_stats])
        vl = np.mean([s["vl"] for s in train_stats])
        vl_std = np.std([s["vl"] for s in train_stats])

        if it % eval_interval != 0:
            continue

        with timed(timer, 'duel_with_random'):
            stats = duel(None, model,
                         id_a=RANDOM_ID, id_b=CURRENT_ID,
                         elo_agent=elo_agent, n_workers=n_workers)

            win_rate_random = stats['win_rate_b']

        with timed(timer, 'duel'):
            stats_best = duel(
                best_model, model,
                id_a=BEST_ID, id_b=CURRENT_ID,
                elo_agent=elo_agent,
                n_games=n_games, n_workers=n_workers
            )

        if stats_best["win_rate_b"] >= 0.55:
            best_model.load_state_dict(model.state_dict())
            elo_agent.elos[BEST_ID] = elo_agent.elos[CURRENT_ID]
            print("✅ Updated BEST model\n")

            if elo_agent.elos[BEST_ID] >= goal_elo:
                break

        save_checkpoint(model, best_model, optimizer, elo_agent)
        save_state(replay_buffer)

        timer.report()
        print(
            f'\nwin rate random: {win_rate_random:.2f} |',
            f'pl: {pl:.2f} ({pl_std:.2f}) |',
            f'vl: {vl:.2f} ({vl_std:.2f}) |',
            f'current elo: {int(elo_agent.elos[CURRENT_ID])} |',
            f'buffer len: {len(replay_buffer)}\n'
        )


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Device: {DEVICE}")

    # Models & Optimizer
    model = OthelloResNet(num_blocks=4, channels=64)
    model.to(DEVICE)
    model.train()

    best_model = OthelloResNet(num_blocks=4, channels=64)
    best_model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    # load checkpoint
    try:
        checkpoint = load_checkpoint()
        model.load_state_dict(checkpoint['model'])
        best_model.load_state_dict(checkpoint['best_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        elo_agent = EloAgent.load_state_dict(checkpoint["elo"])
        print('Checkpoint Loaded')

    except Exception as e:
        print(e)
        elo_agent = EloAgent()
        print('Checkpoint Not Loaded')

    try:
        state = load_state()
        buffer = ReplayBuffer.load_state(state)
        print('State Loaded')

    except Exception as e:
        print(e)
        buffer = ReplayBuffer()
        print('State Not Loaded')

    print('Warm-up replay buffer...')
    while len(buffer) < BATCH_SIZE * 10:
        data, _ = generate_self_play(model, n_workers=cfg.mcts.n_workers)
        for own, opp, pi, z, _ in data:
            buffer.add(own, opp, pi, z)

    print('start training...')
    train_with_mcts(
        best_model, model, buffer, optimizer, elo_agent,
        cfg.train.train_steps_per_iter,
        cfg.train.value_coef,
        cfg.train.clip_grad,
        cfg.train.eval_interval,
        cfg.train.n_games,
        cfg.train.goal_elo,
        cfg.mcts.n_workers,
        timer=timer
    )


if __name__ == "__main__":
    main()
