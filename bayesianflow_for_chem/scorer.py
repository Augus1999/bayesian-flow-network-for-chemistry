# -*- coding: utf-8 -*-
# Author: Nianze A. TAO (Omozawa SUENO)
"""
Define essential scorers.
"""
from typing import List, Callable, Union, Optional
import torch
from torch import Tensor
from rdkit import RDLogger
from rdkit.Contrib.SA_Score import sascorer  # type: ignore
from rdkit.Chem import MolFromSmiles, QED

RDLogger.DisableLog("rdApp.*")  # type: ignore


def smiles_valid(smiles: str) -> int:
    """
    Return the validity of a SMILES string.

    :param smiles: SMIlES string
    :type smiles: str
    :return: validity
    :rtype: int
    """
    return 1 if (MolFromSmiles(smiles) and smiles) else 0


def qed_score(smiles: str) -> float:
    """
    Return the quantitative estimate of drug-likeness score of a SMILES string.

    :param smiles: SMILES string
    :type smiles: str
    :return: QED score
    :rtype: float
    """
    return QED.qed(MolFromSmiles(smiles))


def sa_score(smiles: str) -> float:
    """
    Return the synthetic accessibility score of a SMILES string.

    :param smiles: SMILES string
    :type smiles: str
    :return: SA score
    :rtype: float
    """
    return sascorer.calculateScore(MolFromSmiles(smiles))


class Scorer:
    def __init__(
        self,
        scorers: List[Callable[[str], Union[int, float]]],
        score_criteria: List[Callable[[Union[int, float]], float]],
        vocab_keys: List[str],
        vocab_separator: str = "",
        valid_checker: Optional[Callable[[str], int]] = None,
        eta: float = 1e-2,
        name: str = "scorer",
    ) -> None:
        """
        Scorer class.
        e.g.

        ```python
        scorer = Scorer(
            scorers=[smiles_valid, qed_score],
            score_criteria=[lambda x: float(x == 1), lambda x: float(x > 0.5)],
            vocab_keys=VOCAB_KEYS,
        )
        ```

        :param scorers: a list of scorer(s)
        :param score_criteria: a list of score criterion (or criteria) in the same order of scorers
        :param vocab_keys: a list of (ordered) vocabulary
        :param vocab_separator: token separator; default is `""`
        :param valid_checker: a callable to check the validity of sequences; default is `None`
        :param eta: the coefficient to be multiplied to the loss
        :param name: the name of this scorer
        :type scorers: list
        :type score_criteria: list
        :type vocab_keys: list
        :type vocab_separator: str
        :type eta: float
        :type name: str
        :type valid_checker: typing.Callable | None
        """
        assert len(scorers) == len(
            score_criteria
        ), "The number of scores should match that of criteria."
        self.scorers = scorers
        self.score_criteria = score_criteria
        self.vocab_keys = vocab_keys
        self.vocab_separator = vocab_separator
        self.valid_checker = valid_checker
        self.eta = eta
        self.name = name

    def calc_score_loss(self, p: Tensor) -> Tensor:
        """
        Calculate the score loss.

        :param p: token probability distributions;  shape: (n_b, n_t, n_vocab)
        :type p: torch.Tensor
        :return: score loss;                        shape: ()
        :rtype: torch.Tensor
        """
        tokens = p.argmax(-1)
        e_k = torch.nn.functional.one_hot(tokens, len(self.vocab_keys)).float()
        seqs = [
            self.vocab_separator.join([self.vocab_keys[i] for i in j])
            .split("<start>" + self.vocab_separator)[-1]
            .split(self.vocab_separator + "<end>")[0]
            .replace("<pad>", "")
            for j in tokens
        ]
        valid = [
            1 if self.valid_checker is None else self.valid_checker(i) for i in seqs
        ]
        scores = [
            [
                1 if valid[j] == 0 else 1 - self.score_criteria[i](scorer(seq))
                for j, seq in enumerate(seqs)
            ]
            for i, scorer in enumerate(self.scorers)
        ]
        loss = (e_k * p).sum(2).mean(1) * p.new_tensor(scores).mean(0)
        return loss.mean()


if __name__ == "__main__":
    ...
