# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A subset of the D4RL dataset, used for training Decision Transformers"""


import pickle

import datasets
import numpy as np

_DESCRIPTION = """\
A subset of the D4RL dataset, used for training Decision Transformers
"""

_HOMEPAGE = "https://github.com/rail-berkeley/d4rl"

_LICENSE = "Apache-2.0"



class DecisionTransformerGymDataset(datasets.GeneratorBasedBuilder):
    """The dataset comprises of tuples of (Observations, Actions, Rewards, Dones) sampled
    by an expert policy for various continuous control tasks (halfcheetah, hopper, walker2d)"""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="PandaPickAndPlace-v1",
            version=VERSION,
            description="Data sampled from an expert policy in the halfcheetah Mujoco environment",
        ),
        datasets.BuilderConfig(
            name="PandaReach-v1",
            version=VERSION,
            description="Data sampled from an expert policy in the halfcheetah Mujoco environment",
        ),
        datasets.BuilderConfig(
            name="halfcheetah-expert-v2",
            version=VERSION,
            description="Data sampled from an expert policy in the halfcheetah Mujoco environment",
        )
    ]

    def _info(self):

        features = datasets.Features(
            {
                "observations": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))), #obs
                "actions": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),
                "rewards": datasets.Sequence(datasets.Value("float32")),
                "dones": datasets.Sequence(datasets.Value("bool")),
                # These are the features of your dataset like images, labels ...
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            # Here we define them above because they are different between the two configurations
            features=features,
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        data_dir = "decision_transformer_gym_replay/PandaPickAndPlace-v1.pkl"   #PandaReach-v1.pkl"       #PandaPickAndPlace-v1.pkl"      #halfcheetah-expert-v2.pkl"       #PandaPickAndPlace-v1.pkl"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, "rb") as f:
            trajectories = pickle.load(f)

            for idx, traj in enumerate(trajectories):
                yield idx, {
                    "observations": traj["observations"], #observations
                    "actions": traj["actions"],
                    "rewards": np.expand_dims(traj["rewards"], axis=1),
                    "dones": np.expand_dims(traj["dones"], axis=1),             # "dones": np.expand_dims(traj.get("dones", traj.get("terminals")), axis=1),   #"dones": np.expand_dims(traj["dones"], axis=1),
                }
