import mowl

mowl.init_jvm("8g", "1g", 8)

from mowl.base_models.elmodel import EmbeddingELModel
from mowl.nn import ELModule
from mowl.projection.factory import projector_factory
from elembeddings_losses import *
from data_utils.dataloader import OntologyDataLoader
import torch as th
from torch import nn
from torch.nn.functional import relu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
import numpy as np
from mowl.datasets import PathDataset


class ELEmModule(ELModule):
    def __init__(
        self,
        nb_ont_classes,
        nb_rels,
        embed_dim=50,
        margin=0.1,
        loss_type="leaky_relu",
        reg_norm=1,
        reg_mode="relaxed",
        test_gci="gci2",
    ):
        super().__init__()
        self.nb_ont_classes = nb_ont_classes
        self.nb_rels = nb_rels
        self.reg_norm = reg_norm
        self.embed_dim = embed_dim
        self.test_gci = test_gci

        self.class_embed = nn.Embedding(self.nb_ont_classes, embed_dim)
        nn.init.uniform_(self.class_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(self.class_embed.weight.data, axis=1)
        weight_data_normalized = weight_data_normalized.reshape(-1, 1)
        self.class_embed.weight.data /= weight_data_normalized

        self.class_rad = nn.Embedding(self.nb_ont_classes, 1)
        nn.init.uniform_(self.class_rad.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(
            self.class_rad.weight.data, axis=1
        ).reshape(-1, 1)
        self.class_rad.weight.data /= weight_data_normalized

        self.rel_embed = nn.Embedding(nb_rels, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, a=-1, b=1)
        weight_data_normalized = th.linalg.norm(
            self.rel_embed.weight.data, axis=1
        ).reshape(-1, 1)
        self.rel_embed.weight.data /= weight_data_normalized

        self.margin = margin
        if loss_type in ["relu", "leaky_relu"]:
            self.loss_type = loss_type
        else:
            raise ValueError('"loss_type" should be one of ["relu", "leaky_relu"]')
        if reg_mode in ["relaxed", "original"]:
            self.reg_mode = reg_mode
        else:
            raise ValueError('"reg_mode" should be one of ["relaxed", "original"]')

    def class_reg(self, x):
        if self.reg_norm is None:
            res = th.zeros(x.size()[0], 1)
        else:
            if self.reg_mode == "original":
                res = th.abs(th.linalg.norm(x, axis=1) - self.reg_norm)
            else:
                res = relu(th.linalg.norm(x, axis=1) - self.reg_norm)
            res = th.reshape(res, [-1, 1])
        return res

    def gci0_loss(self, data, neg=False):
        return gci0_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci0_bot_loss(self, data, neg=False):
        return gci0_bot_loss(data, self.class_rad)

    def gci1_loss(self, data, neg=False):
        return gci1_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci1_bot_loss(self, data, neg=False):
        return gci1_bot_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci2_loss(self, data, neg=False, idxs_for_negs=None):
        return gci2_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.rel_embed,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci3_loss(self, data, neg=False):
        return gci3_loss(
            data,
            self.class_embed,
            self.class_rad,
            self.rel_embed,
            self.class_reg,
            self.margin,
            self.loss_type,
            neg=neg,
        )

    def gci3_bot_loss(self, data, neg=False):
        return gci3_bot_loss(data, self.class_rad)

    def eval_method(self, data):
        if self.test_gci == "gci0":
            return gci0_loss(
                data,
                self.class_embed,
                self.class_rad,
                self.class_reg,
                self.margin,
                self.loss_type,
                neg=False,
            )
        elif self.test_gci == "gci1":
            return gci1_loss(
                data,
                self.class_embed,
                self.class_rad,
                self.class_reg,
                self.margin,
                self.loss_type,
                neg=False,
            )
        elif self.test_gci == "gci1_bot":
            return gci1_bot_loss(
                data,
                self.class_embed,
                self.class_rad,
                self.class_reg,
                self.margin,
                self.loss_type,
                neg=False,
            )
        elif self.test_gci == "gci2":
            return gci2_loss(
                data,
                self.class_embed,
                self.class_rad,
                self.rel_embed,
                self.class_reg,
                self.margin,
                self.loss_type,
                neg=False,
            )
        elif self.test_gci == "gci3":
            return gci3_loss(
                data,
                self.class_embed,
                self.class_rad,
                self.rel_embed,
                self.class_reg,
                self.margin,
                self.loss_type,
                neg=neg,
            )


class ELEmbeddings(EmbeddingELModel):
    def __init__(
        self,
        dataset,
        embed_dim=50,
        margin=0,
        reg_norm=1,
        loss_type="leaky_relu",
        learning_rate=0.001,
        epochs=1000,
        batch_size=4096 * 8,
        model_filepath=None,
        device="cpu",
        reg_mode="relaxed",
        test_gci="gci2",
        eval_property=None,
    ):
        super().__init__(
            dataset, embed_dim, batch_size, extended=True, model_filepath=model_filepath
        )

        self.embed_dim = embed_dim
        self.margin = margin
        self.reg_norm = reg_norm
        self.loss_type = loss_type
        self.reg_mode = reg_mode
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.test_gci = test_gci
        self.eval_property = eval_property
        self._loaded = False
        self._loaded_eval = False
        self.extended = False
        self.init_model()

    def init_model(self):
        self.module = ELEmModule(
            len(self.class_index_dict),
            len(self.object_property_index_dict),
            embed_dim=self.embed_dim,
            margin=self.margin,
            loss_type=self.loss_type,
            reg_norm=self.reg_norm,
            reg_mode=self.reg_mode,
            test_gci=self.test_gci,
        ).to(self.device)
        self.eval_method = self.module.eval_method

    def load_eval_data(self):
        if self._loaded_eval:
            return

        eval_classes = self.dataset.evaluation_classes.as_str

        self._head_entities = set(list(eval_classes)[:])
        self._tail_entities = set(list(eval_classes)[:])

        if self.test_gci == "gci0":
            eval_projector = projector_factory("taxonomy")
        elif self.test_gci == "gci2":
            eval_projector = projector_factory(
                "taxonomy_rels", taxonomy=False, relations=[self.eval_property]
            )
        else:
            raise NotImplementedError

        self._training_set = eval_projector.project(self.dataset.ontology)
        self._testing_set = eval_projector.project(self.dataset.testing)

        self._loaded_eval = True

    def get_embeddings(self):
        self.init_model()

        print("Load the best model", self.model_filepath)
        self.load_best_model()

        ent_embeds = {
            k: v
            for k, v in zip(
                self.class_index_dict.keys(),
                self.module.class_embed.weight.cpu().detach().numpy(),
            )
        }
        rel_embeds = {
            k: v
            for k, v in zip(
                self.object_property_index_dict.keys(),
                self.module.rel_embed.weight.cpu().detach().numpy(),
            )
        }
        return ent_embeds, rel_embeds

    def load_best_model(self):
        self.init_model()
        self.module.load_state_dict(th.load(self.model_filepath))
        self.module.eval()

    @property
    def training_set(self):
        self.load_eval_data()
        return self._training_set

    @property
    def testing_set(self):
        self.load_eval_data()
        return self._testing_set

    @property
    def head_entities(self):
        self.load_eval_data()
        return self._head_entities

    @property
    def tail_entities(self):
        self.load_eval_data()
        return self._tail_entities

    def train(self):
        raise NotImplementedError


class ELEmPPI(ELEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(
        self,
        patience=10,
        epochs_no_improve=20,
        loss_weight=True,
        path_to_dc=None,
    ):
        optimizer = th.optim.Adam(self.module.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
        no_improve = 0
        best_loss = float("inf")

        if path_to_dc is not None:
            train_dataloader = OntologyDataLoader(
                self.training_datasets["gci0"][:],
                self.training_datasets["gci1"][:],
                self.training_datasets["gci2"][:],
                self.training_datasets["gci3"][:],
                self.training_datasets["gci0_bot"][:],
                self.training_datasets["gci1_bot"][:],
                self.training_datasets["gci3_bot"][:],
                self.batch_size,
                self.device,
                negative_mode="filtered",
                path_to_dc=path_to_dc,
                class_index_dict=self.class_index_dict,
                object_property_index_dict=self.object_property_index_dict,
                evaluation_classes=self.dataset.evaluation_classes.as_str,
            )
        else:
            train_dataloader = OntologyDataLoader(
                self.training_datasets["gci0"][:],
                self.training_datasets["gci1"][:],
                self.training_datasets["gci2"][:],
                self.training_datasets["gci3"][:],
                self.training_datasets["gci0_bot"][:],
                self.training_datasets["gci1_bot"][:],
                self.training_datasets["gci3_bot"][:],
                self.batch_size,
                self.device,
                negative_mode="random",
                class_index_dict=self.class_index_dict,
                object_property_index_dict=self.object_property_index_dict,
                evaluation_classes=self.dataset.evaluation_classes.as_str,
            )
        num_steps = train_dataloader.num_steps
        steps = train_dataloader.steps
        all_steps = sum(list(steps.values()))
        if loss_weight:
            weights = [steps[k] / all_steps for k in steps.keys()]
        else:
            weights = [1] * 7

        for epoch in trange(self.epochs):
            self.module.train()

            train_loss = 0

            for batch in train_dataloader:
                cur_loss = 0
                (
                    gci0,
                    gci0_neg,
                    gci1,
                    gci1_neg,
                    gci2,
                    gci2_neg,
                    gci3,
                    gci3_neg,
                    gci0_bot,
                    gci0_bot_neg,
                    gci1_bot,
                    gci1_bot_neg,
                    gci3_bot,
                    gci3_bot_neg,
                ) = batch
                if len(gci0) > 0:
                    pos_loss = self.module(gci0, "gci0")
                    l = th.mean(pos_loss)
                    # l = th.mean(pos_loss) + th.mean(
                    #     self.module(gci0_neg, "gci0", neg=True)
                    # )
                    cur_loss += weights[0] * l
                if len(gci1) > 0:
                    pos_loss = self.module(gci1, "gci1")
                    l = th.mean(pos_loss)
                    # l = th.mean(pos_loss) + th.mean(
                    #     self.module(gci1_neg, "gci1", neg=True)
                    # )
                    cur_loss += weights[1] * l
                if len(gci2) > 0:
                    pos_loss = self.module(gci2, "gci2")
                    # l = th.mean(pos_loss)
                    l = th.mean(pos_loss) + th.mean(
                        self.module(gci2_neg, "gci2", neg=True)
                    )
                    cur_loss += weights[2] * l
                if len(gci3) > 0:
                    pos_loss = self.module(gci3, "gci3")
                    l = th.mean(pos_loss)
                    # l = th.mean(pos_loss) + th.mean(
                    #     self.module(gci3_neg, "gci3", neg=True)
                    # )
                    cur_loss += weights[3] * l
                if len(gci0_bot) > 0:
                    pos_loss = self.module(gci0_bot, "gci0_bot")
                    l = th.mean(pos_loss)
                    cur_loss += weights[4] * l
                if len(gci1_bot) > 0:
                    pos_loss = self.module(gci1_bot, "gci1_bot")
                    l = th.mean(pos_loss)
                    cur_loss += weights[5] * l
                if len(gci3_bot) > 0:
                    pos_loss = self.module(gci3_bot, "gci3_bot")
                    l = th.mean(pos_loss)
                    cur_loss += weights[6] * l
                train_loss += cur_loss.detach().item()
                optimizer.zero_grad()
                cur_loss.backward()
                optimizer.step()

            train_loss /= num_steps

            loss = 0
            with th.no_grad():
                self.module.eval()
                valid_loss = 0
                if self.test_gci == "gci0":
                    gci0_data = self.validation_datasets["gci0"][:]
                    loss = th.mean(self.module(gci0_data, "gci0"))
                elif self.test_gci == "gci1":
                    gci1_data = self.validation_datasets["gci1"][:]
                    loss = th.mean(self.module(gci1_data, "gci1"))
                elif self.test_gci == "gci2":
                    gci2_data = self.validation_datasets["gci2"][:]
                    loss = th.mean(self.module(gci2_data, "gci2"))
                elif self.test_gci == "gci3":
                    gci3_data = self.validation_datasets["gci3"][:]
                    loss = th.mean(self.module(gci3_data, "gci3"))
                elif self.test_gci == "gci1_bot":
                    gci1_bot_data = self.validation_datasets["gci1_bot"][:]
                    loss = th.mean(self.module(gci1_bot_data, "gci1_bot"))
                valid_loss += loss.detach().item()
                scheduler.step(valid_loss)

            if best_loss > valid_loss:
                best_loss = valid_loss
                th.save(self.module.state_dict(), self.model_filepath)
                print(f"Best loss: {best_loss}, epoch: {epoch}")
                no_improve = 0
            else:
                no_improve += 1

            if no_improve == epochs_no_improve:
                print(f"Stopped at epoch {epoch}")
                break

    def eval_method(self, data):
        return self.module.eval_method(data)
