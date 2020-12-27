from strsimpy.weighted_levenshtein import WeightedLevenshtein


class BeginningLevenshtein(WeightedLevenshtein):
    def __init__(self, *args, **kwargs):
        self.weighted_levenshtein = WeightedLevenshtein(*args, **kwargs)

    def distance(self, text: str, reference: str) -> float:
        return self(text, reference)["distance"]

    def __call__(self, text: str, reference: str) -> float:
        ref_beginnigs = [reference[:i] for i in range(len(reference) + 1)]
        distances = [
            self.weighted_levenshtein.distance(text, ref) for ref in ref_beginnigs
        ]
        candidates = [
            {"match_ref": ref, "match_len": len(ref), "distance": d}
            for ref, d in zip(ref_beginnigs, distances)
        ]
        best = min(candidates, key=lambda c: c["distance"])
        return best
