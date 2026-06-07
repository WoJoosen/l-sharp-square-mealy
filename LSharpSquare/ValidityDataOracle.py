from aalpy.base import Automaton
from aalpy.base.Oracle import Oracle


class ValidityDataOracle(Oracle):
    def __init__(self, data):
        """
        Give data in format: [(["a", "b"], ["out1", "out2"]), (["b", "a", "a"], ["o1", "o2", "o3"])]
        For Mealy machines: input sequence mapped to output sequence
        """
        super().__init__(None, None)
        self.data = data
        self.num_queries = 0
        self.num_steps = 0

    def find_cex(self, hypothesis: Automaton):
        for inputs, expected_outputs in self.data:
            hypothesis.reset_to_initial()
            actual_outputs = []
            for input_val in inputs:
                output = hypothesis.step(input_val)
                actual_outputs.append(output)
                self.num_steps += 1
            self.num_queries += 1
            
            # Check for counterexample: only compare positions where expected output is not "unknown"
            for i, (expected, actual) in enumerate(zip(expected_outputs, actual_outputs)):
                if expected != "unknown" and expected != actual:
                    print(f"Counterexample found!")
                    print(f"Expected outputs: {expected_outputs} vs. Actual outputs: {tuple(actual_outputs)}")
                    return inputs
        return None