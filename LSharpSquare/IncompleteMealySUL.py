from aalpy.automata import MealyMachine
from aalpy.base.SUL import SUL

from LSharpSquare.CacheTree import CacheTree


class MealySUL(SUL):
    def __init__(self, automaton: MealyMachine):
        super().__init__()
        self.outputs = []
        self.automaton: MealyMachine = automaton
        self.num_successful_queries = 0

    def pre(self):
        self.automaton.reset_to_initial()
        self.outputs = []

    def step(self, letter=None):
        out = self.automaton.step(letter)
        self.outputs.append(out)
        return out

    def post(self):
        return tuple(self.outputs)

    def query(self, word: tuple) -> list:
        """
        Performs an output query on the SUL. Before the query, pre() method is called and after the query post()
        method is called. Each letter in the word (input in the input sequence) is executed using the step method.

        Args:

            word: output query (word consisting of letters/inputs)

        Returns:

            final output

        """
        self.pre()
        for letter in word:
            self.step(letter)
        out = self.post()
        self.num_queries += 1
        self.num_successful_queries += 1
        self.num_steps += len(word)
        if isinstance(out, (list, tuple)) and len(out) > 0 and all(val in (None, "unknown") for val in out):
            self.num_successful_queries -= 1
        return out


class IncompleteMealySUL(MealySUL):
    def __init__(self, words, automaton: MealyMachine = None):
        super().__init__(automaton)
        self.input_walk = None
        self.cache = CacheTree()

        for word, output in words:
            self.add_word(word, output)
        for word, output in words:
            if self.word_known(word) != tuple(output):
                raise Exception(f"Word {word} with output {output} inconsistent with cache")

    def add_word(self, word, outputs):
        word = tuple(word)
        outputs = tuple(outputs)
        if len(word) != len(outputs):
            raise ValueError(f"Input and output lengths differ for word {word}: {len(word)} != {len(outputs)}")

        self.cache.reset()
        if len(word) == 0:
            self.cache.step_in_cache(None, ())
            return

        for input_val, output_val in zip(word, outputs):
            self.cache.step_in_cache(input_val, output_val)

    def word_known(self, word):
        word = tuple(word)
        outputs = self.cache.in_cache(word)

        if type(outputs) != list and type(outputs) != tuple:
            return outputs

        outputs = tuple(outputs)
        if len(outputs) != len(word):
            return None

        return outputs

    def pre(self):
        self.input_walk = []
        if self.automaton is not None:
            self.automaton.reset_to_initial()
            self.outputs = []

    def step(self, letter=None):
        self.input_walk.append(letter)
        if self.automaton is None:
            self.outputs.append(None)
            return None
        else:
            out = self.automaton.step(letter)
            self.outputs.append(out)
            return out

    def post(self):
        saved_output = self.word_known(self.input_walk)
        if saved_output is None:
            if self.automaton is not None:
                self.add_word(self.input_walk, tuple(self.outputs))
                return tuple(self.outputs)
            else:
                unknown_trace = tuple(None for _ in self.input_walk)
                self.add_word(self.input_walk, unknown_trace)
                return unknown_trace
        else:
            return saved_output
