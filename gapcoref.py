#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scores system output for the GAP challenge.
"""
from __future__ import division
from __future__ import print_function

import argparse
from collections import defaultdict
import csv

from constants import Gender
from constants import GOLD_FIELDNAMES
from constants import PRONOUNS
from constants import SYSTEM_FIELDNAMES
import spacy
import neuralcoref
filename = "/Users/leo/PycharmProjects/gap-coreference/gap-development.tsv"


class Annotation(object):
    """Container class for storing annotations of an example.

    Attributes:
      gender(None): The gender of the annotation. None indicates that gender was
        not determined for the given example.
      name_a_coref(None): bool reflecting whether Name A was recorded as
        coreferential with the target pronoun for this example. None indicates
        that no annotation was found for the given example.
      name_b_coref(None): bool reflecting whether Name B was recorded as
        coreferential with the target pronoun for this example. None indicates
        that no annotation was found for the given example.
    """

    def __init__(self):
        self.gender = None
        self.name_a_coref = None
        self.name_b_coref = None


class Scores(object):
    """Container class for storing scores, and generating evaluation metrics.

    Attributes:
      true_positives: Tally of true positives seen.
      false_positives: Tally of false positives seen.
      true_negatives: Tally of true negatives seen.
      false_negatives: Tally of false negatives seen.
    """

    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def recall(self):
        """Calculates recall based on the observed scores.

        Returns:
          float, the recall.
        """
        numerator = self.true_positives
        denominator = self.true_positives + self.false_negatives
        return 100.0 * numerator / denominator if denominator else 0.0

    def precision(self):
        """Calculates precision based on the observed scores.

        Returns:
          float, the precision.
        """
        numerator = self.true_positives
        denominator = self.true_positives + self.false_positives
        return 100.0 * numerator / denominator if denominator else 0.0

    def f1(self):
        """Calculates F1 based on the observed scores.

        Returns:
          float, the F1 score.
        """
        recall = self.recall()
        precision = self.precision()

        numerator = 2 * precision * recall
        denominator = precision + recall
        return numerator / denominator if denominator else 0.0


def read_annotations(filename, is_gold):
    """Reads coreference annotations for the examples in the given file.

    Args:
      filename: Path to .tsv file to read.
      is_gold: Whether or not we are reading the gold annotations.

    Returns:
      A dict mapping example ID strings to their Annotation representation. If
      reading gold, 'Pronoun' field is used to determine gender.
    """

    def is_true(value):
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        else:
            print('Unexpected label!', value)
            return None

    fieldnames = GOLD_FIELDNAMES if is_gold else SYSTEM_FIELDNAMES

    annotations = defaultdict(Annotation)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter='\t')

        # Skip the header line in the gold data
        if is_gold:
            next(reader, None)

        for row in reader:
            example_id = row['ID']
            if example_id in annotations:
                print('Multiple annotations for', example_id)
                continue

            annotations[example_id].name_a_coref = is_true(row['A-coref'])
            annotations[example_id].name_b_coref = is_true(row['B-coref'])
            if is_gold:
                gender = PRONOUNS.get(row['Pronoun'].lower(), Gender.UNKNOWN)
                assert gender != Gender.UNKNOWN, row
                annotations[example_id].gender = gender
    return annotations


def get_system_annotations(filename):
    fieldnames = SYSTEM_FIELDNAMES
    first = True
    nlp = spacy.load("en_core_web_lg")
    neuralcoref.add_to_pipe(nlp, greedyness=0.5)
    system_annotations = defaultdict(Annotation)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter='\t')
        for row in reader:
            if first:
                first = False
                continue
            doc = nlp(row['Text'])
            A_coref = False
            B_coref = False
            found_pronoun = False
            for cluster in doc._.coref_clusters:
                for token in cluster.mentions:
                    print(row['ID'], token.text, token.start_char)
                    if token.start_char == int(row['Pronoun-offset']):
                        found_pronoun = True
                        # find the coref-A and coref-B
                        for token_coref in cluster.mentions:
                            if token_coref.start_char == int(row['A-offset']):
                                A_coref = True
                            elif token_coref.start_char == int(row['B-offset']):
                                B_coref = True
                        break
            if found_pronoun:
                system_annotations[row["ID"]].name_a_coref = A_coref
                system_annotations[row["ID"]].name_b_coref = B_coref
            else:
                system_annotations[row["ID"]].name_a_coref = None
                system_annotations[row["ID"]].name_b_coref = None
    for key in system_annotations:
        if system_annotations[key]:
            print(key, system_annotations[key].name_a_coref, system_annotations[key].name_b_coref)
        else:
            print(key, system_annotations[key])
    return system_annotations


def my_get_system_annotations(filename):
    fieldnames = SYSTEM_FIELDNAMES
    first = True
    nlp = spacy.load("en_core_web_lg")
    neuralcoref.add_to_pipe(nlp, greedyness=0.5)
    system_annotations = defaultdict(Annotation)
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter='\t')
        for row in reader:
            if first:
                first = False
                continue
            doc = nlp(row['Text'])
            clusters = doc._.coref_clusters
            merge_identical_entity(clusters)
            merge_potential_coreference(clusters)
            A_coref = False
            B_coref = False
            found_pronoun = False
            for cluster in clusters:
                for token in cluster.mentions:
                    print(row['ID'], token.text, token.start_char)
                    if token.start_char == int(row['Pronoun-offset']):
                        found_pronoun = True
                        # find the coref-A and coref-B
                        for token_coref in cluster.mentions:
                            if token_coref.start_char == int(row['A-offset']):
                                A_coref = True
                            elif token_coref.start_char == int(row['B-offset']):
                                B_coref = True
                        break
            if found_pronoun:
                system_annotations[row["ID"]].name_a_coref = A_coref
                system_annotations[row["ID"]].name_b_coref = B_coref
            else:
                system_annotations[row["ID"]].name_a_coref = None
                system_annotations[row["ID"]].name_b_coref = None
    for key in system_annotations:
        if system_annotations[key]:
            print(key, system_annotations[key].name_a_coref, system_annotations[key].name_b_coref)
        else:
            print(key, system_annotations[key])
    return system_annotations


def calculate_scores(gold_annotations, system_annotations):
    """Score the system annotations against gold.

    Args:
      gold_annotations: dict from example ID to its gold Annotation.
      system_annotations: dict from example ID to its system Annotation.

    Returns:
      A dict from gender to a Scores object for that gender. None is used to
        denote no specific gender, i.e. overall scores.
    """
    scores = {}
    for example_id, gold_annotation in gold_annotations.items():
        system_annotation = system_annotations[example_id]

        name_a_annotations = [
            gold_annotation.name_a_coref, system_annotation.name_a_coref
        ]
        name_b_annotations = [
            gold_annotation.name_b_coref, system_annotation.name_b_coref
        ]
        for gender in [None, gold_annotation.gender]:
            if gender not in scores:
                scores[gender] = Scores()

            for (gold, system) in [name_a_annotations, name_b_annotations]:
                if system is None:
                    print('Missing output for', example_id)
                    scores[gender].false_negatives += 1
                elif gold and system:
                    scores[gender].true_positives += 1
                elif not gold and system:
                    scores[gender].false_positives += 1
                elif not gold and not system:
                    scores[gender].true_negatives += 1
                elif gold and not system:
                    scores[gender].false_negatives += 1
    return scores


def make_scorecard(scores):
    """Returns a human-readable scorecard of the given scores.

    Args:
      scores: dict from gender to its Scores object. None is used to denote no
        specific gender, i.e. overall scores.

    Returns:
      A string, the scorecard.
    """
    scorecard = []

    display_names = [(None, 'Overall'), (Gender.MASCULINE, 'Masculine'),
                     (Gender.FEMININE, 'Feminine')]

    bias_terms = {}
    for gender, display_name in display_names:
        gender_scores = scores.get(gender, Scores())

        recall = gender_scores.recall()
        precision = gender_scores.precision()
        f1 = gender_scores.f1()
        bias_terms[gender] = f1

        scorecard.append('{} recall: {:.1f} precision: {:.1f} f1: {:.1f}'.format(
            display_name, recall, precision, f1))
        scorecard.append('\t\ttp {:d}\tfp {:d}'.format(
            gender_scores.true_positives, gender_scores.false_positives))
        scorecard.append('\t\tfn {:d}\ttn {:d}'.format(
            gender_scores.false_negatives, gender_scores.true_negatives))

    bias = '-'
    if bias_terms[Gender.MASCULINE] and bias_terms[Gender.FEMININE]:
        bias = '{:.2f}'.format(
            bias_terms[Gender.FEMININE] / bias_terms[Gender.MASCULINE])
    scorecard.append('Bias (F/M): {}\n'.format(bias))
    return '\n'.join(scorecard)


def run_scorer(gold_tsv, system_tsv):
    """Run the scorer.

    Args:
      gold_tsv: Gold annotations to score against.
      system_tsv: System output to score.

    Returns:
      A string, the scorecard.
    """
    gold_annotations = read_annotations(gold_tsv, is_gold=True)
    assert gold_annotations, 'No gold annotations read!'

    system_annotations = read_annotations(system_tsv, is_gold=False)
    assert system_annotations, 'No system annotations read!'

    scores = calculate_scores(gold_annotations, system_annotations)
    return make_scorecard(scores)


def main(args):
    """Score system output against gold and display the scorecard.

    Args:
      args: argparse namespace containing gold_tsv and system_tsv.
    """
    scorecard = run_scorer(args.gold_tsv, args.system_tsv)
    print(scorecard)


def merge_identical_entity(clusters):
    a_dict = {}
    for idx, cluster in enumerate(clusters):
        if cluster.main.text.strip() not in a_dict.keys():
            a_dict[cluster.main.text.strip()] = cluster
        else:
            a_dict[cluster.main.text.strip()].mentions.extend(cluster.mentions)
            clusters.pop(idx)


def find_overlaping(clusters):
    a_dict = {}  # key is index of cluster in clusters, value is soome clusters overlapping with it
    clusters.sort(key=lambda cluster: cluster.main.text)
    for idx, cluster in enumerate(clusters):
        is_substring = False
        for key in a_dict.keys():
            if len(key.main.text) >= len(cluster.main.text):
                if cluster.main.text in key.main.text:
                    a_dict[key].append(cluster)
                    is_substring = True
            else:
                if key.main.text in cluster.main.text:
                    a_dict[key].append(cluster)
                    is_substring = True
        if not is_substring:
            a_dict[cluster] = [cluster]
    return a_dict


def merge_potential_coreference(clusters):
    a_dict = find_overlaping(clusters)
    for key in a_dict:
        if len(a_dict[key]) > 1:
            main_entity = a_dict[key][0]
            main_entity_len = len(main_entity.main)
            print("main_entity: ", main_entity.main)
            print("--------------")
            for idx, entity in enumerate(a_dict[key][1:]):
                # find entities whose start is identical to main entity
                if entity.main[:main_entity_len].text == main_entity.main.text:
                    # if main entity follows with 's, conjunction, or preposition then don't merge
                    if entity.main[1].pos_ in ["PART", "ADP", "CCONJ"]:
                        print(entity.main.text, "don't merge")
                    else:
                        print(entity.main.text, "to merge")
                        main_entity.mentions.extend(entity.mentions)
                        clusters.remove(entity)

                    print()
            print("*************")


if __name__ == '__main__':
    gold_annotations = read_annotations(filename, True)
    system_annotations = get_system_annotations(filename)
    scores1 = calculate_scores(gold_annotations, system_annotations)
    scores1 = scores1[None]
    my_system_annotations = my_get_system_annotations(filename)
    scores2 = calculate_scores(gold_annotations, my_system_annotations)
    scores2 = scores2[None]
    print(scores1.f1())
    print(scores2.f1())

    # dataset = load_dataset("quoref")
    # print(dataset)

