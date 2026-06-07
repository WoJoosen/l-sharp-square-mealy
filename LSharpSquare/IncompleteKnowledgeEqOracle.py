from aalpy.base import Oracle, DeterministicAutomaton
from aalpy.automata import Dfa, MooreMachine, MealyMachine, MooreState, MealyState, DfaState
from queue import Queue
from typing import Tuple, Union
import itertools as it


def _outputs_match_with_unknowns(out1, out2) -> bool:
    """
    Check if outputs match, treating "unknown" as wildcard.
    Returns True if outputs are compatible.
    """
    if out1 == "unknown" or out2 == "unknown":
        return True
    
    return out1 == out2


def bisimilar_with_unknowns(a1: DeterministicAutomaton, a2: DeterministicAutomaton, 
                            return_cex=False) -> Union[bool, None, tuple]:
    """
    Checks whether the provided automata are bisimilar, treating "unknown" as wildcard.
    If return_cex the function returns a counter example or None, otherwise a Boolean is returned.
    
    This is a modified version of aalpy.utils.bisimilar() that handles incomplete knowledge.
    Based on: https://github.com/DES-Lab/AALpy

    Returns:
        object: true or false if return_cex is set to False, otherwise None (no counterexample) or a counterexample
    """

    if a1.__class__ != a2.__class__:
        raise ValueError("tried to check bisimilarity of different automaton types")
    
    if a1 is a2:
        a2 = a1.copy()

    supported_automaton_types = (Dfa, MooreMachine, MealyMachine)
    if not isinstance(a1, supported_automaton_types):
        raise NotImplementedError(
            f"bisimilarity is not implemented for {a1.__class__.__name__}. Supported: {', '.join(t.__name__ for t in supported_automaton_types)}")

    to_check: Queue[Tuple] = Queue()
    to_check.put((a1.initial_state, a2.initial_state))
    requirements = dict()
    requirements[(a1.initial_state, a2.initial_state)] = ()

    while not to_check.empty():
        s1, s2 = to_check.get()

        if isinstance(s1, (DfaState, MooreState)) and s1.output != s2.output:
            return requirements[(s1, s2)] if return_cex else False

        t1, t2 = s1.transitions, s2.transitions
        for t in it.chain(t1.keys(), filter(lambda x: x not in t1.keys(), t2.keys())):
            common = t in t1.keys() and t in t2.keys()
            
            if isinstance(s1, MealyState):
                outputs_match = _outputs_match_with_unknowns(s1.output_fun[t], s2.output_fun[t])
                if (not common) or (not outputs_match):
                    return requirements[(s1, s2)] + (t,) if return_cex else False
            else:
                if (not common) or (isinstance(s1, MealyState) and s1.output_fun[t] != s2.output_fun[t]):
                    return requirements[(s1, s2)] + (t,) if return_cex else False

        for t in t1.keys():
            c1, c2 = t1[t], t2[t]
            if (c1, c2) not in requirements:
                requirements[(c1, c2)] = requirements[(s1, s2)] + (t,)
                to_check.put((c1, c2))

    return None if return_cex else True


class IncompleteKnowledgeEqOracle(Oracle):
    """
    Oracle for learning from incomplete/partial knowledge about a Mealy machine.
    Modified version of AALpy's PerfectKnowledgeEqOracle that treats "unknown" outputs as wildcards.
    Based on aalpy.utils.bisimilar() function.
    """
    def __init__(self, alphabet: list, sul, model_under_learning: DeterministicAutomaton):
        super().__init__(alphabet, sul)
        self.model_under_learning = model_under_learning
        self.num_queries = 0
        self.num_steps = 0

    def find_cex(self, hypothesis):
        """Find counterexample by comparing hypothesis with incomplete model."""
        self.num_queries += 1

        cex = bisimilar_with_unknowns(hypothesis, self.model_under_learning, return_cex=True)

        if cex is not None:
            self.num_steps += len(cex)

        return cex