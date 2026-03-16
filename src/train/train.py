from src.self_play.self_play import *
from src.buffer.buffer import ReplayBuffer
import torch.nn.functional as F

saved_path = ROOT / 'checkpoint' / 'checkpoint.tar'

train_param = config['train_param']
VALUE_COEF = train_param['VALUE_COEF']
CLIP_GRAD = train_param['CLIP_GRAD']
LR = train_param['LR']
WEIGHT_DECAY = train_param['WEIGHT_DECAY']
TRAIN_STEPS_PER_ITER = train_param['TRAIN_STEPS_PER_ITER']
N_GAMES = train_param['N_GAMES']

NUM_ITERATIONS = 5000


def save_checkpoint(model, best_model, optimizer, elo_tracker):
    torch.save({
        "model": model.state_dict(),
        "best_model": best_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "elo": elo_tracker.state_dict()
    }, saved_path)


def load_checkpoint():
    if DEVICE == 'cpu':
        ck = torch.load(saved_path, map_location=torch.device('cpu'), weights_only=False)
    else:
        ck = torch.load(saved_path, weights_only=False)

    return ck


def alphazero_loss(policy_logits, value, target_pi, target_z, value_coef=1.0):
    log_p = F.log_softmax(policy_logits, dim=1)  # (B, 65)
    policy_loss = - torch.mean(torch.sum(target_pi * log_p, dim=1))
    value_loss = torch.mean((value - target_z)**2)
    loss = policy_loss + value_coef * value_loss
    return loss, policy_loss.item(), value_loss.item()


def train_step(
        model: OthelloResNet, optimizer,
        replay_buffer: ReplayBuffer,
        batch_size=BATCH_SIZE,
        value_coef=VALUE_COEF,
        clip_grad=CLIP_GRAD
):
    if len(replay_buffer) < batch_size:
        return None
    model.train()
    states, pis, zs = replay_buffer.sample(batch_size)
    policy_logits, values = model(states)
    loss, pl, vl = alphazero_loss(
        policy_logits, values, pis, zs, value_coef
    )
    optimizer.zero_grad()
    loss.backward()
    if clip_grad is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    return {"loss": loss.item(), "policy_loss": pl, "value_loss": vl}


def train_with_mcts(
    best_model: OthelloResNet, model: OthelloResNet,
    replay_buffer: ReplayBuffer,
    optimizer,
    elo_agent: EloAgent,
    num_iterations=NUM_ITERATIONS,
    train_steps_per_iter=TRAIN_STEPS_PER_ITER,
    eval_interval=10,
    n_games=N_GAMES,
    timer=None
):
    BEST_ID = "best"
    RANDOM_ID = "random"

    if timer:
        timer.reset('current_0')

    for it in range(num_iterations):
        CURRENT_ID = f"current_{it}"

        with timed(timer, 'duel_with_random'):
            model.eval()
            stats = duel(None, model,
                         id_a=RANDOM_ID, id_b=CURRENT_ID,
                         elo_agent=elo_agent)

            win_rate_random = stats['win_rate_b']

        train_stats = []
        for _ in range(train_steps_per_iter):
            with timed(timer, 'generate_self_play'):
                model.eval()
                data, _ = generate_self_play(model)
                for own, opp, pi, z, _ in data:
                    replay_buffer.add(own, opp, pi, z)

            with timed(timer, 'train_step'):
                model.train()
                out = train_step(model, optimizer, replay_buffer)
                if out is not None:
                    train_stats.append(out)

        pl = np.mean([s["policy_loss"] for s in train_stats])
        vl = np.mean([s["value_loss"] for s in train_stats])

        if it % eval_interval != 0 or it == 0:
            continue

        with timed(timer, 'duel'):
            model.eval()
            stats_best = duel(
                best_model, model,
                id_a=BEST_ID, id_b=CURRENT_ID,
                elo_agent=elo_agent,
                n_games=n_games,
            )

        if (
            stats_best["win_rate_b"] >= 0.55
            and win_rate_random > 0.8
        ):
            best_model.load_state_dict(model.state_dict())
            elo_agent.elos[BEST_ID] = elo_agent.elos[CURRENT_ID]
            print("✅ Updated BEST model")

        save_checkpoint(model, best_model, optimizer, elo_agent)

        timer.report()
        print(
            f'\nwin rate random: {win_rate_random:.1f} |',
            f'pl: {pl:.2f} |',
            f'vl: {vl:.2f} |',
            f'best elo: {int(elo_agent.elos[BEST_ID])} |',
            f'current elo: {int(elo_agent.elos[CURRENT_ID])}\n'
        )

        if timer:
            timer.reset(CURRENT_ID)
    return model


if __name__ == "__main__":
    buffer = ReplayBuffer()
    checkpoint = load_checkpoint()

    # Models & Optimizer
    model = OthelloResNet(num_blocks=4, channels=64)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)

    best_model = OthelloResNet(num_blocks=4, channels=64)
    best_model.load_state_dict(checkpoint['best_model'])
    best_model.to(DEVICE)
    best_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(checkpoint['optimizer'])

    # elo_agent = EloAgent.load_state_dict(checkpoint["elo"])
    elo_agent = EloAgent()

    print('Warm-up replay buffer...')
    while len(buffer) < BATCH_SIZE:
        data, _ = generate_self_play(model)
        for own, opp, pi, z, _ in data:
            buffer.add(own, opp, pi, z)

    print('start training...')
    trained_model = train_with_mcts(
        best_model, model, buffer, optimizer, elo_agent,
        eval_interval=1,
        timer=timer
    )
