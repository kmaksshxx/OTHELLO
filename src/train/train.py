from src.self_play.self_play import *


def save_checkpoint(model, best_model, optimizer, elo_tracker):
  torch.save({
    "model": model.state_dict(),
    "best_model": best_model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "elo": elo_tracker.state_dict()
  }, saved_path)


def load_checkpoint():
  if DEVICE == 'cpu':
    ck = torch.load(saved_path, map_location=torch.device('cpu'))
  elif DEVICE == 'cuda':
    ck = torch.load(saved_path)

  return ck


def alphazero_loss(policy_logits, value, target_pi, target_z, value_coef=1.0):
    logp = F.log_softmax(policy_logits, dim=1) #(B, 65)
    policy_loss = - torch.mean(torch.sum(target_pi * logp, dim=1))
    value_loss = torch.mean((value - target_z)**2)
    loss = policy_loss + value_coef * value_loss
    return loss, policy_loss.item(), value_loss.item()


def train_step(
        model, optimizer,
        replay_buffer: ReplayBuffer,
        batch_size=TRAIN_BATCH,
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
    best_model,
    mcts: MCTS,
    replay_buffer: ReplayBuffer,
    optimizer,
    elo_agent: EloAgent,
    num_iterations=NUM_ITERATIONS,
    train_steps_per_iter=TRAIN_STEPS_PER_ITER,
    eval_interval=10,
    eval_games=100, # Increased from 50 for more reliable Elo evaluation
):
    BEST_ID = "best"
    RANDOM_ID = "random"

    # ---- Elo init ----
    elo_agent.ensure(RANDOM_ID)
    elo_agent.ensure(BEST_ID)

    history_policy_loss, history_value_loss = [], []
    history_rand_wr, history_elo_best, history_elo_current = [], [], []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    display.display(fig)
    display.clear_output(wait=True)

    #  tq = tqdm(range(num_iterations))

    for it in range(num_iterations):
        CURRENT_ID = f"current_{it}"
        elo_agent.ensure(CURRENT_ID)

        # ==================================================
        # 0. Measure strength vs random (NO buffer)
        # ==================================================
        mcts.model.eval()
        win_rand = 0
        stats = duel(old_model=None, new_model=mcts.model,
                     old_id=RANDOM_ID, new_id=CURRENT_ID,
                     elo_agent=elo_agent)
        n_probe = 10



        for _ in range(n_probe):
            _, winner = generate_game({
                1: {'mcts': mcts}, -1: {'mcts': None},
            })
            if winner == 1:
                win_rand += 1

        win_rate_random = win_rand / n_probe
        history_rand_wr.append(win_rate_random)

        # ==================================================
        # 1. Decide opponent mix
        # ==================================================
        if win_rate_random < 0.4:
            p_random = 1.0
        elif win_rate_random < 0.6:
            p_random = 0.5
        elif win_rate_random < 0.8:
            p_random = 0.2
        else:
            p_random = 0.0

        # ==================================================
        # 2. Self-play / Random-play
        # ==================================================
        mcts.reset_tree()
        t0 = time.time()

        # ADDED
        p_random = -1

        if np.random.rand() < p_random:
            # ---------- vs Random (NO buffer) ----------
            data, _ = generate_game({
                1: {'mcts': mcts},
                -1: {'mcts': None},
            })
        else:
            # ---------- Self-play (buffer ON) ----------
            data, _ = generate_game(
                {
                    1: {'mcts': mcts, 'record': True},
                    -1: {'mcts': mcts, 'record': True},
                },
                reuse_tree=True
            )

            for own, opp, pi, z, p in data:
                replay_buffer.add(own, opp, pi, z, p)

        selfplay_time = int(time.time() - t0)

        # ==================================================
        # 3. Training
        # ==================================================
        mcts.model.train()
        t1 = time.time()
        train_stats = []

        for _ in range(train_steps_per_iter):
            out = train_step(mcts.model, optimizer, replay_buffer)
            if out is not None:
                train_stats.append(out)

        pl = train_stats[-1]["policy_loss"] if train_stats else 0.0
        vl = train_stats[-1]["value_loss"] if train_stats else 0.0

        history_policy_loss.append(pl)
        history_value_loss.append(vl)

        training_time = int(time.time() - t1)

        # ==================================================
        # 4. Evaluation vs BEST
        # ==================================================
        if it % eval_interval != 0 or it == 0:
            continue

        mcts.model.eval()

        stats_best = duel(
            elo_agent,
            old_model=best_model,
            new_model=mcts.model,
            old_id=BEST_ID,
            new_id=CURRENT_ID,
            n_games=eval_games,
        )

        history_elo_best.append(elo_agent.elos[BEST_ID])
        history_elo_current.append(elo_agent.elos[CURRENT_ID])

        # Plotting section
        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.plot(range(len(history_policy_loss)), history_policy_loss, label='Policy Loss')
        ax1.plot(range(len(history_value_loss)), history_value_loss, label='Value Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Policy and Value Loss Over Training Steps')
        ax1.legend()

        ax2.plot(range(len(history_rand_wr)), history_rand_wr, label='Random Win Rate', color='orange')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Win Rate')
        ax2.set_title('Win Rate Against Random Player')
        ax2.legend()

        ax3.plot(range(len(history_elo_best)), history_elo_best, label='Best Model Elo', color='green')
        ax3.plot(range(len(history_elo_current)), history_elo_current, label='Current Model Elo', color='red')
        ax3.set_xlabel('Iterations (Evaluation Points)')
        ax3.set_ylabel('Elo Rating')
        ax3.set_title('Elo Ratings Over Iterations')
        ax3.legend()

        fig.suptitle(f'Training Progress (Iteration {it})')
        fig.tight_layout(rect=(0, 0.03, 1, 0.95)) # Adjust layout to prevent title overlap
        display.clear_output(wait=True)
        display.display(fig)


        # ==================================================
        # 5. Update BEST model
        # ==================================================
        if (
            stats_best["elo_delta_new"] > 20
            and stats_best["win_rate_old"] < 0.45
            and win_rate_random > 0.8
        ):
            best_model.load_state_dict(mcts.model.state_dict())
            elo_agent.elos[BEST_ID] = elo_agent.elos[CURRENT_ID]
            print("✅ Updated BEST model")

        # ==================================================
        # 6. Plateau handling
        # ==================================================
        # if stats_best["plateau"]:
        #     replay_buffer.soft_reset()

        save_checkpoint(mcts.model, best_model, optimizer, elo_agent)
    return mcts.model

#@title MAIN

if __name__ == "__main__":
    buffer = ReplayBuffer()
    checkpoint = load_checkpoint()

    # Models & Optimzer
    model = OthelloResNet(num_blocks=4, channels=64)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    model.train()

    best_model = OthelloResNet(num_blocks=4, channels=64)
    best_model.load_state_dict(checkpoint['best_model'])
    best_model.to(DEVICE)
    best_model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(checkpoint['optimizer'])

    # EloAgent
    elo_agent = EloAgent.load_state_dict(checkpoint["elo"])
    # elo_agent = EloAgent()
    # Random Policy
    elo_agent.elos['random'] = 1500
    elo_agent.ensure('best')
    elo_agent.ensure('current')

    # MCTS
    mcts = MCTS(model=model, n_sim=MCTS_SIMS, batch_eval=MCTS_BATCH)

    # warm up: do a small self-play to fill buffer a bit (optional)
    print('Warm-up replay buffer...')
    while len(buffer) < TRAIN_BATCH:
        data, _ = generate_game({
            1: {'mcts': mcts}, -1: {'mcts': mcts}
        })
        for s, pi, z, p in data:
            buffer.add(s, pi, z, p)

    # start training
    trained_model = train_with_mcts(
        best_model, mcts, buffer, optimizer, elo_agent,
        num_iterations=5000,
        train_steps_per_iter=20,
        eval_interval=10,
        eval_games=100,
    )

# GPU : selfplay=421, train=1
# CPU : selfplay=31, train=7

