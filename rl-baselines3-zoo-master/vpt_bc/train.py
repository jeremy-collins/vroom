# Train one model for each task
from behavioural_cloning import behavioural_cloning_train
from torch.utils.tensorboard import SummaryWriter
# https://github.com/minerllabs/basalt-2022-behavioural-cloning-baseline

'''
cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
conda activate vpt
cd vpt_bc
python3 train.py

cd /home/codysoccerman/Documents/classes/Fall_22/Deep_Learning/Project/rl-baselines3-zoo-master
cd vpt_bc
tensorboard --logdir .

'''


def main():
    print("===Training PandaPickAndPlace-v1 model===")
    behavioural_cloning_train(
        data_dir="data/PandaPickAndPlace-v1",     #"data/MineRLBasaltFindCave-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/PandaPickAndPlace-v1.weights"
    )
    print("=== Done Training PandaPickAndPlace-v1 model===")

    '''
    print("===Training MakeWaterfall model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltMakeWaterfall-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltMakeWaterfall.weights"
    )

    print("===Training CreateVillageAnimalPen model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltCreateVillageAnimalPen-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltCreateVillageAnimalPen.weights"
    )

    print("===Training BuildVillageHouse model===")
    behavioural_cloning_train(
        data_dir="data/MineRLBasaltBuildVillageHouse-v0",
        in_model="data/VPT-models/foundation-model-1x.model",
        in_weights="data/VPT-models/foundation-model-1x.weights",
        out_weights="train/MineRLBasaltBuildVillageHouse.weights"
    )
    '''


if __name__ == "__main__":
    main()

