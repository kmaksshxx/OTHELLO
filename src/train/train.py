from src.self_play.self_play import *
from src.buffer.buffer import ReplayBuffer
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Parameters')

saved_path = ROOT / 'checkpoint' / 'checkpoint.tar'
state_path = ROOT / 'checkpoint' / 'checkpoint_state.tar'

train_param = config['train_param']
value_coef = train_param['VALUE_COEF']
clip_grad = train_param['CLIP_GRAD']
lr = train_param['LR']
WEIGHT_DECAY = train_param['WEIGHT_DECAY']
train_steps_per_iter = train_param['TRAIN_STEPS_PER_ITER']
n_games = train_param['N_GAMES']

parser.add_argument('--value_coef', default=value_coef, type=float)
parser.add_argument('--clip_grad', default=clip_grad, type=float)
parser.add_argument('--lr', default=lr, type=float)
parser.add_argument('--train_steps_per_iter', default=train_steps_per_iter, type=int)
parser.add_argument('--n_games', default=n_games, type=int)

args = parser.parse_args()


NUM_ITERATIONS = 5000


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


def alphazero_loss(policy_logits, value, target_pi, target_z, value_coef=1.0):
    """
    Calculate loss

    Returns
      - loss, policy_loss, value_loss
    """
    log_p = F.log_softmax(policy_logits, dim=1)  # (B, 65)
    policy_loss = - torch.mean(torch.sum(target_pi * log_p, dim=1))
    value_loss = torch.mean((value - target_z)**2)
    loss = policy_loss + value_coef * value_loss
    return loss, policy_loss.item(), value_loss.item()


def train_step(
        model: OthelloResNet, optimizer,
        replay_buffer: ReplayBuffer,
        batch_size=BATCH_SIZE,
        value_coef=args.value_coef,
        clip_grad=args.clip_grad
):
    if len(replay_buffer) < batch_size:
        return None
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
    train_steps_per_iter=args.train_steps_per_iter,
    eval_interval=1,
    n_games=args.n_games,
    timer=None
):
    BEST_ID = "best"
    RANDOM_ID = "random"

    if timer:
        timer.reset('current_0')

    for it in range(num_iterations):
        CURRENT_ID = f"current_{it}"

        with timed(timer, 'duel_with_random'):
            stats = duel(None, model,
                         id_a=RANDOM_ID, id_b=CURRENT_ID,
                         elo_agent=elo_agent)

            win_rate_random = stats['win_rate_b']

        train_stats = []
        for _ in range(train_steps_per_iter):
            with timed(timer, 'generate_self_play'):
                data, _ = generate_self_play(model)
                for own, opp, pi, z, _ in data:
                    replay_buffer.add(own, opp, pi, z)

            with timed(timer, 'train_step'):
                model.train()
                for _ in range(10):
                    out = train_step(model, optimizer, replay_buffer)
                    if out is not None:
                        train_stats.append(out)

        pl = np.mean([s["policy_loss"] for s in train_stats])
        vl = np.mean([s["value_loss"] for s in train_stats])

        if it % eval_interval != 0 or it == 0:
            continue

        if win_rate_random > 0.8:
            with timed(timer, 'duel'):
                stats_best = duel(
                    best_model, model,
                    id_a=BEST_ID, id_b=CURRENT_ID,
                    elo_agent=elo_agent,
                    n_games=n_games,
                )

            if stats_best["win_rate_b"] >= 0.55:
                best_model.load_state_dict(model.state_dict())
                elo_agent.elos[BEST_ID] = elo_agent.elos[CURRENT_ID]
                print("✅ Updated BEST model")

        save_checkpoint(model, best_model, optimizer, elo_agent)
        save_state(buffer)

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
    # Models & Optimizer
    model = OthelloResNet(num_blocks=4, channels=64)
    model.to(DEVICE)

    best_model = OthelloResNet(num_blocks=4, channels=64)
    best_model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

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
    while len(buffer) < BATCH_SIZE * 100:
        data, _ = generate_self_play(model)
        for own, opp, pi, z, _ in data:
            buffer.add(own, opp, pi, z)

    print('start training...')
    trained_model = train_with_mcts(
        best_model, model, buffer, optimizer, elo_agent,
        timer=timer
    )
