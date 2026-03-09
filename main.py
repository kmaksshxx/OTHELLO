import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

print(config['MCTS'])
