# coding=utf-8
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
"""TODO: Add a description here."""

import datasets

from Levenshtein import distance

# TODO: Add BibTeX citation
_CITATION = """\
@InProceedings{huggingface:metric,
title = {A great new metric},
authors={huggingface, Inc.},
year={2020}
}
"""

# TODO: Add description of the metric here
_DESCRIPTION = """\
This new metric is designed to solve this great NLP task and is crafted with a lot of care.
"""


# TODO: Add description of the arguments of the metric here
_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score.
    references: list of reference for each prediction.
Returns:
    exact_match: description of the first score,
    distance: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=['ab', 'cd'], predictions=['ab', 'cc'])
    >>> print(results)
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "http://url/to/external/resource/bad_words.txt"


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class JaSeq2SeqMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string"),
                }
            ),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"],
        )

    # def _download_and_prepare(self, dl_manager):
    #     """Optional: download external resources useful to compute the scores"""
    # TODO: Download external resources if needed
    # bad_words_path = dl_manager.download_and_extract(BAD_WORDS_URL)
    # self.bad_words = set(
    #     [w.strip() for w in open(bad_words_path, "r", encoding="utf-8")]
    # )

    def _compute(self, predictions, references):
        """Returns the scores"""
        mean_exact_match = sum(
            pred == ref for pred, ref in zip(predictions, references)
        ) / len(predictions)
        mean_distance = sum(
            distance(pred, ref) for pred, ref in zip(predictions, references)
        ) / len(predictions)

        # if self.config_name == "max":
        #     second_score = max(
        #         abs(len(i) - len(j))
        #         for i, j in zip(predictions, references)
        #         if i not in self.bad_words
        #     )
        return {
            "exact_match": mean_exact_match,
            "distance": mean_distance,
        }