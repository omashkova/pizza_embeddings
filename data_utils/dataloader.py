import math
import numpy as np
import torch as th
from mowl.datasets import PathDataset
from mowl.datasets.el import ELDataset
from uk.ac.manchester.cs.owl.owlapi import OWLClassImpl

class OntologyDataLoader:
    def __init__(
        self,
        gci0,
        gci1,
        gci2,
        gci3,
        gci0_bot,
        gci1_bot,
        gci3_bot,
        batch_size,
        device,
        evaluation_classes,
        negative_mode="random",
        path_to_dc=None,
        class_index_dict=None,
        object_property_index_dict=None,
    ):
        self.gci0 = gci0
        self.gci1 = gci1
        self.gci2 = gci2
        self.gci3 = gci3
        self.gci0_bot = gci0_bot
        self.gci1_bot = gci1_bot
        self.gci3_bot = gci3_bot
        self.batch_size = batch_size
        self.device = device
        self.class_index_dict = class_index_dict
        self.evaluation_classes = evaluation_classes
        if negative_mode in ["random", "filtered"]:
            self.negative_mode = negative_mode
        else:
            raise ValueError('"negative_mode" should be one of ["random", "filtered"]')
        self.steps = {
            "gci0": int(math.ceil(len(gci0) / batch_size)),
            "gci1": int(math.ceil(len(gci1) / batch_size)),
            "gci2": int(math.ceil(len(gci2) / batch_size)),
            "gci3": int(math.ceil(len(gci3) / batch_size)),
            "gci0_bot": int(math.ceil(len(gci0_bot) / batch_size)),
            "gci1_bot": int(math.ceil(len(gci1_bot) / batch_size)),
            "gci3_bot": int(math.ceil(len(gci3_bot) / batch_size)),
        }
        self.remainders = {
            "gci0": 0,
            "gci1": 0,
            "gci2": 0,
            "gci3": 0,
            "gci0_bot": 0,
            "gci1_bot": 0,
            "gci3_bot": 0,
        }
        self.num_steps = max(self.steps.values())
        self.path_to_dc = path_to_dc
        if self.negative_mode == "filtered":
            self.deductive_closure = ELDataset(
                PathDataset(self.path_to_dc).ontology,
                class_index_dict,
                object_property_index_dict,
                device=self.device,
            ).get_gci_datasets()

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        return self.next()

    def generate_negatives(self, gci_name, pos_batch):
        np.random.seed(0)
        if gci_name == "gci0":
            data = pos_batch[:, 1]
            classes = [self.class_index_dict[p] for p in self.evaluation_classes]
            corrupted = np.random.choice(classes, size=len(data), replace=True)
            corrupted = th.tensor(corrupted).to(self.device)
            neg_batch = th.cat([pos_batch[:, :1], corrupted.unsqueeze(1)], dim=1)
            # if self.negative_mode == "filtered":
            #     arr_max = np.maximum(
            #         neg_batch.cpu().numpy().max(0) + 1,
            #         self.deductive_closure["gci0"][:].cpu().numpy().max(0) + 1,
            #     )
            #     neg_batch = neg_batch[
            #         ~np.isin(
            #             neg_batch.cpu().numpy().dot(arr_max),
            #             self.deductive_closure["gci0"][:].cpu().numpy().dot(arr_max),
            #         )
            #     ]
        elif gci_name in ["gci1", "gci2", "gci3"]:
            data = pos_batch[:, 2]
            classes = [self.class_index_dict[p] for p in self.evaluation_classes]
            corrupted = np.random.choice(classes, size=len(data), replace=True)
            corrupted = th.tensor(corrupted).to(self.device)
            neg_batch = th.cat([pos_batch[:, :2], corrupted.unsqueeze(1)], dim=1)
            if self.negative_mode == "filtered":
                # if gci_name == "gci1":
                #     arr_max = np.maximum(
                #         neg_batch.cpu().numpy().max(0) + 1,
                #         self.deductive_closure["gci1"][:].cpu().numpy().max(0) + 1,
                #     )
                #     neg_batch = neg_batch[
                #         ~np.isin(
                #             neg_batch.cpu().numpy().dot(arr_max),
                #             self.deductive_closure["gci1"][:]
                #             .cpu()
                #             .numpy()
                #             .dot(arr_max),
                #         )
                #     ]
                if gci_name == "gci2":
                    arr_max = np.maximum(
                        neg_batch.cpu().numpy().max(0) + 1,
                        self.deductive_closure["gci2"][:].cpu().numpy().max(0) + 1,
                    )
                    neg_batch = neg_batch[
                        ~np.isin(
                            neg_batch.cpu().numpy().dot(arr_max),
                            self.deductive_closure["gci2"][:]
                            .cpu()
                            .numpy()
                            .dot(arr_max),
                        )
                    ]
                # elif gci_name == "gci3":
                #     arr_max = np.maximum(
                #         neg_batch.cpu().numpy().max(0) + 1,
                #         self.deductive_closure["gci3"][:].cpu().numpy().max(0) + 1,
                #     )
                #     neg_batch = neg_batch[
                #         ~np.isin(
                #             neg_batch.cpu().numpy().dot(arr_max),
                #             self.deductive_closure["gci3"][:]
                #             .cpu()
                #             .numpy()
                #             .dot(arr_max),
                #         )
                #     ]
        elif gci_name in ["gci0_bot", "gci1_bot"]:
            data = pos_batch[:, 0]
            classes = [self.class_index_dict[p] for p in self.evaluation_classes]
            corrupted = np.random.choice(classes, size=len(data), replace=True)
            corrupted = th.tensor(corrupted).to(self.device)
            neg_batch = th.cat([corrupted.unsqueeze(1), pos_batch[:, 1:]], dim=1)
            # if self.negative_mode == "filtered":
            #     if gci_name == "gci0_bot":
            #         neg_batch = None
            #     if gci_name == "gci1_bot":
            #         arr_max = np.maximum(
            #             neg_batch.cpu().numpy().max(0) + 1,
            #             self.deductive_closure["gci1_bot"][:].cpu().numpy().max(0) + 1,
            #         )
            #         neg_batch = neg_batch[
            #             ~np.isin(
            #                 neg_batch.cpu().numpy().dot(arr_max),
            #                 self.deductive_closure["gci1_bot"][:]
            #                 .cpu()
            #                 .numpy()
            #                 .dot(arr_max),
            #             )
            #         ]
        elif gci_name == "gci3_bot":
            data = pos_batch[:, 1]
            classes = [self.class_index_dict[p] for p in evaluation_classes]
            corrupted = np.random.choice(classes, size=len(data), replace=True)
            corrupted = th.tensor(corrupted).to(self.device)
            neg_batch = th.cat(
                [
                    pos_batch[:, :1],
                    corrupted.unsqueeze(1),
                    pos_batch[:, 2:],
                ],
                dim=1,
            )
            if self.negative_mode == "filtered":
                neg_batch = None
        return neg_batch

    def next(self):
        if self.i < self.num_steps:
            if self.gci0.size()[0] > 0:
                j = self.remainders["gci0"]
                gci0_batch = self.gci0[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci0"] = (j + 1) % self.steps["gci0"]
                gci0_neg_batch = self.generate_negatives("gci0", gci0_batch)
            else:
                gci0_batch = self.gci0
                gci0_neg_batch = self.gci0
            if self.gci1.size()[0] > 0:
                j = self.remainders["gci1"]
                gci1_batch = self.gci1[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci1"] = (j + 1) % self.steps["gci1"]
                gci1_neg_batch = self.generate_negatives("gci1", gci1_batch)
            else:
                gci1_batch = self.gci1
                gci1_neg_batch = self.gci1
            if self.gci2.size()[0] > 0:
                j = self.remainders["gci2"]
                gci2_batch = self.gci2[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci2"] = (j + 1) % self.steps["gci2"]
                gci2_neg_batch = self.generate_negatives("gci2", gci2_batch)
            else:
                gci2_batch = self.gci2
                gci2_neg_batch = self.gci2
            if self.gci3.size()[0] > 0:
                j = self.remainders["gci3"]
                gci3_batch = self.gci3[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci3"] = (j + 1) % self.steps["gci3"]
                gci3_neg_batch = self.generate_negatives("gci3", gci3_batch)
            else:
                gci3_batch = self.gci3
                gci3_neg_batch = self.gci3
            if self.gci0_bot.size()[0] > 0:
                j = self.remainders["gci0_bot"]
                gci0_bot_batch = self.gci0_bot[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci0_bot"] = (j + 1) % self.steps["gci0_bot"]
                gci0_bot_neg_batch = self.generate_negatives("gci0_bot", gci0_bot_batch)
            else:
                gci0_bot_batch = self.gci0_bot
                gci0_bot_neg_batch = self.gci0_bot
            if self.gci1_bot.size()[0] > 0:
                j = self.remainders["gci1_bot"]
                gci1_bot_batch = self.gci1_bot[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci1_bot"] = (j + 1) % self.steps["gci1_bot"]
                gci1_bot_neg_batch = self.generate_negatives("gci1_bot", gci1_bot_batch)
            else:
                gci1_bot_batch = self.gci1_bot
                gci1_bot_neg_batch = self.gci1_bot
            if self.gci3_bot.size()[0] > 0:
                j = self.remainders["gci3_bot"]
                gci3_bot_batch = self.gci3_bot[
                    self.batch_size * j : self.batch_size * (j + 1), :
                ]
                self.remainders["gci3_bot"] = (j + 1) % self.steps["gci3_bot"]
                gci3_bot_neg_batch = self.generate_negatives("gci3_bot", gci3_bot_batch)
            else:
                gci3_bot_batch = self.gci3_bot
                gci3_bot_neg_batch = self.gci3_bot
            self.i += 1
            return (
                gci0_batch,
                gci0_neg_batch,
                gci1_batch,
                gci1_neg_batch,
                gci2_batch,
                gci2_neg_batch,
                gci3_batch,
                gci3_neg_batch,
                gci0_bot_batch,
                gci0_bot_neg_batch,
                gci1_bot_batch,
                gci1_bot_neg_batch,
                gci3_bot_batch,
                gci3_bot_neg_batch,
            )
        else:
            raise StopIteration
