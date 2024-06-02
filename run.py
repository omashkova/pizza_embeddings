import mowl

mowl.init_jvm("100g", "1g", 8)

from mowl.datasets import PathDataset

from elembeddings_pizza import ELEmPPI
from evaluation_utils import ModelRankBasedEvaluator
import torch as th
import numpy as np
import random
from org.semanticweb.owlapi.model import OWLClass
from mowl.projection.factory import projector_factory


class Entities:
    """Abstract class containing OWLEntities indexed by they IRIs"""

    def __init__(self, collection):
        self._collection = self.check_owl_type(collection)
        self._collection = sorted(self._collection, key=lambda x: x.toStringID())
        self._name_owlobject = self.to_dict()
        self._index_dict = self.to_index_dict()

    def __getitem__(self, idx):
        return self._collection[idx]

    def __len__(self):
        return len(self._collection)

    def __iter__(self):
        self.ind = 0
        return self

    def __next__(self):
        if self.ind < len(self._collection):
            item = self._collection[self.ind]
            self.ind += 1
            return item
        raise StopIteration

    def check_owl_type(self, collection):
        """This method checks whether the elements in the provided collection
        are of the correct type.
        """
        raise NotImplementedError

    def to_str(self, owl_class):
        raise NotImplementedError

    def to_dict(self):
        """Generates a dictionaty indexed by OWL entities IRIs and the values
        are the corresponding OWL entities.
        """
        dict_ = {self.to_str(ent): ent for ent in self._collection}
        return dict_

    def to_index_dict(self):
        """Generates a dictionary indexed by OWL objects and the values
        are the corresponding indicies.
        """
        dict_ = {v: k for k, v in enumerate(self._collection)}
        return dict_

    @property
    def as_str(self):
        """Returns the list of entities as string names."""
        return list(self._name_owlobject.keys())

    @property
    def as_owl(self):
        """Returns the list of entities as OWL objects."""
        return list(self._name_owlobject.values())

    @property
    def as_dict(self):
        """Returns the dictionary of entities indexed by their names."""
        return self._name_owlobject

    @property
    def as_index_dict(self):
        """Returns the dictionary of entities indexed by their names."""
        return self._index_dict


class OWLClasses(Entities):
    """
    Iterable for :class:`org.semanticweb.owlapi.model.OWLClass`
    """

    def check_owl_type(self, collection):
        for item in collection:
            if not isinstance(item, OWLClass):
                raise TypeError("Type of elements in collection must be OWLClass.")
        return collection

    def to_str(self, owl_class):
        name = str(owl_class.toStringID())
        return name


class PizzaDataset(PathDataset):
    def init(self, args, **kwargs):
        super().init(args, **kwargs)

    @property
    def evaluation_classes(self):
        """Classes that are used in evaluation"""

        if self._evaluation_classes is None:
            classes = set()
            for owl_name, owl_cls in self.classes.as_dict.items():
                classes.add(owl_cls)
            self._evaluation_classes = OWLClasses(classes)

        return self._evaluation_classes


random.seed(0)
th.manual_seed(0)

dataset = PizzaDataset(
    "../geometric_models_data/pizza_data/gci0/ontology.owl",
    "../geometric_models_data/pizza_data/gci0/valid.owl",
    "../geometric_models_data/pizza_data/gci0_ontology_2_no_thing.owl",
)

# dc = PizzaDataset("../geometric_models_data/pizza_data/gci2/gci2_ontology.owl")
dc = PizzaDataset("../geometric_models_data/pizza_data/gci0/test_2.owl")
# dc = PizzaDataset("../geometric_models_data/pizza_data/gci0/test_2.owl")
eval_projector = projector_factory("taxonomy")
# eval_projector = projector_factory("taxonomy_rels", taxonomy=False, relations=["http://www.co-ode.org/ontologies/pizza/pizza.owl#hasTopping"])
dc_set = eval_projector.project(dc.ontology)

model = ELEmPPI(
    dataset,
    embed_dim=50,
    # gamma=0,
    # delta=2,
    margin=0.1,
    loss_type="relu",
    reg_norm=1,
    learning_rate=0.001,
    epochs=2000,
    batch_size=4096 * 8,
    model_filepath="model.pt",
    device="cuda",
    reg_mode="original",
    test_gci="gci0",
    # eval_property="http://www.co-ode.org/ontologies/pizza/pizza.owl#hasTopping"
)

model.train(
    loss_weight=True,
    # path_to_dc='../geometric_models_data/pizza_data/gci2/gci2_ontology.owl',
)

with th.no_grad():
    model.load_best_model()
    evaluator = ModelRankBasedEvaluator(
        model,
        dc_set=dc_set,
        device="cuda",
        eval_method=model.eval_method,
    )

    evaluator.evaluate(
        show=True,
        filename1=None,
        filename2=None,
        test_gci="gci0",
    )
