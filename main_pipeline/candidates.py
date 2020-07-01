class Candidates:

    def __init__(self, sobel_candidates=None, opening_candidates=None, color_candidates=None):
        if color_candidates is None:
            color_candidates = []
        if opening_candidates is None:
            opening_candidates = []
        if sobel_candidates is None:
            sobel_candidates = []
        self.sobel_candidates = sobel_candidates
        self.opening_candidates = opening_candidates
        self.color_candidates = color_candidates

    @property
    def all(self):
        return self.sobel_candidates + self.opening_candidates + self.color_candidates
