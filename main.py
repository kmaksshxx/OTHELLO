from src.train.train import *


if __name__ == "__main__":
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
        data, _ = generate_self_play(model)
        for own, opp, pi, z, _ in data:
            buffer.add(own, opp, pi, z)

    print('start training...')
    trained_model = train_with_mcts(
        best_model, model, buffer, optimizer, elo_agent,
        timer=timer
    )
