import hydra
from trainer.train_graphnet import GraphNetTrainer


@hydra.main(config_path='../config', config_name='graph_nn')
def main(config):
    trainer = GraphNetTrainer(config, None)
    trainer.inference()


if __name__ == "__main__":
    main()
