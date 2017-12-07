import json
import numpy as np
from os import path


class EvaluationDataGenerator:
    def __init__(self, evaluation, batch_size, max_epoch):
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.evaluation = evaluation

        self._epoch = 0

    @property
    def progress(self):
        return self._epoch

    def __iter__(self):
        batch = []
        targets = []
        for self._epoch in range(self.max_epoch):
            for sentence, label in self.evaluation.sentences:
                batch.append(sentence)
                targets.append(label)

                if len(batch) == self.batch_size:
                    inputs, inputs_length = self.prepare_batch(batch)
                    yield inputs, inputs_length, targets
                    batch = []
                    targets = []

        if len(batch) > 0:
            inputs, inputs_length = self.prepare_batch(batch)
            yield inputs, inputs_length, targets

    def prepare_batch(self, batch):
        max_length = 0
        lengths = []

        for sentence in batch:
            cur_len = len(sentence)
            if max_length < cur_len:
                max_length = cur_len

            lengths.append(cur_len)
        lengths = np.array(lengths)

        return np.array([sentence + [0]*(max_length - len(sentence)) for sentence in batch]), lengths


class VRDEvaluation(object):
    def __init__(self, sentences, objects):
        self.sentences = sentences
        self.objects = objects
        print("Loaded %d sentences" % len(self.sentences))
        print("Loaded %d unique objects" % len(self.objects))

    def __len__(self):
        return len(self.sentences)

    @staticmethod
    def from_empty():
        return VRDEvaluation([], [])

    @staticmethod
    def from_directory(directory_path, embeddings, include_train=False, include_test=False):
        # Defining needed constants
        comma = embeddings.get(",")
        noun = embeddings.get("<noun>")

        # Load a specific json file from the directory
        def open_file(file): return json.load(open(path.join(directory_path, file), "rt", encoding="utf-8"))

        objects = open_file("objects.json")
        predicates = open_file("predicates.json")
        test = open_file("annotations_test.json")
        train = open_file("annotations_train.json")

        # Prepare a sentence for each image and produce a dataset with target words replaced with <noun>
        def extract_sentences(dataset):
            sentences = []
            for image in dataset:
                image_participants = set()
                image_sentence = []
                relations = dataset[image]
                for relation in relations:
                    object = embeddings.get(objects[relation["object"]["category"]])
                    subject = embeddings.get(objects[relation["subject"]["category"]])
                    relation = predicates[relation["predicate"]].split(" ")
                    relation = [embeddings.get(ele) for ele in relation]

                    if object != embeddings.unk:
                        image_participants.add(object)
                    if subject != embeddings.unk:
                        image_participants.add(subject)

                    if len(image_sentence) > 0:
                        image_sentence.append(comma)
                    image_sentence.extend([subject] + relation + [object])

                for participant in image_participants:
                    sentences.append(([noun if ele == participant else ele for ele in image_sentence], participant))
            return sentences

        resolved_objects = []
        for object in objects:
            if object in embeddings.dictionary:
                resolved_objects.append(embeddings.get(object))

        sentences = []
        if include_train:
            sentences.extend(extract_sentences(train))
        if include_test:
            sentences.extend(extract_sentences(test))
        return VRDEvaluation(sentences, resolved_objects)
