import json
import numpy as np
from os import path


class VRDEvaluation(object):
    def __init__(self, triples):
        self.triples = triples
        print("Loaded %d relation triples" % len(self.triples))

    def resolve_against_dict(self, embeddings):
        dataset = []
        print("Resolving triples against embeddings")
        for object, predicate, subject in self.triples:
            resolved_triple = [embeddings.get(object)]+[embeddings.get(ele) for ele in predicate.split(" ")]+[embeddings.get(subject)]
            if embeddings.unk in resolved_triple:
                # print("  %s - %s - %s relation is skipped (%s)" % (object, predicate, subject, str(resolved_triple)))
                continue
            dataset.append(resolved_triple)
        print("Done. Skipped %d triples" % (len(self.triples) - len(dataset)))

        return np.array(dataset)

    @staticmethod
    def from_directory(directory_path):
        # Load a specific json file from the directory
        def open_file(file): return json.load(open(path.join(directory_path, file), "rt", encoding="utf-8"))

        objects = open_file("objects.json")
        predicates = open_file("predicates.json")
        test = open_file("annotations_test.json")
        train = open_file("annotations_train.json")

        # Extract all triples object-predicate-subject and resolve them against dictionary
        def extract_triples(dataset):
            triples = []
            for image in dataset:
                relations = dataset[image]
                for relation in relations:
                    triples.append((
                        objects[relation["object"]["category"]],
                        predicates[relation["predicate"]],
                        objects[relation["subject"]["category"]]
                    ))
            return triples

        return VRDEvaluation(extract_triples(test) + extract_triples(train))
