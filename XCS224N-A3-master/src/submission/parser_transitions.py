class PartialParse(object):
    def __init__(self, sentence):
        """Initializes this partial parse.

        Your code should initialize the following fields:
            1. self.stack: 
                The current stack represented as a list with the top of the stack as the last element of the list.
            2. self.buffer: 
                The current buffer represented as a list with the first item on the buffer as the first item of the list
            3. self.dependencies: 
                The list of dependencies produced so far. 
                Represented as a list of tuples where each tuple is of the form (head, dependent).
                Order for this list doesn't matter.
        The root token should be represented with the string "ROOT"
        Args:
            sentence:
                The sentence to be parsed as a list of words.
                Your code should not modify the sentence.
        """
        # The sentence being parsed is kept for bookkeeping purposes.
        # Do not use it in your code.
        
        self.sentence = sentence

        ### START CODE HERE
        # Initially, the stack only contains ROOT
        # The root token should be represented with the string "ROOT"
        self.stack = ["ROOT"]
        # Initially, the dependencies list is empty
        self.dependencies = []
        # Initially, the buffer contains all words of the sentence in order.
        self.buffer = [ word for word in sentence ]
        ### END CODE HERE

    def parse_step(self, transition):
        """Performs a single parse step by applying the given transition to this partial parse

        Args:
            transition: 
                1. A string that equals "S" representing the shift transitions.
                2. A string that equals "LA" representing the left-arc transitions.
                3. A string that equals "RA" representing the right-arc transitions.
                You can assume the provided transition is a legal transition.
        """
        ### START CODE HERE
        if transition == 'S':
            # 1. A string that equals "S" representing the SHIFT transitions.
            if self.buffer:
                first_word = self.buffer.pop(0) # step 1-1 : removes the 1st word from the buffer
                self.stack.append( first_word ) # step 1-2 : pushes it onto the stack.
        elif transition == 'LA':
            # 2. A string that equals "LA" representing the LEFT-ARC transitions.
            #    step 2-1 : marks the 2nd (2nd most recently added) item on the stack as a dependent of the 1st item
            #    step 2-2 : removes the 2nd item from the stack.
            if len(self.stack) >= 2:
                first_item = self.stack[-1]
                second_item = self.stack.pop(-2)
                self.dependencies.append( (first_item,second_item) )
        elif transition == 'RA':
            # 3. A string that equals "RA" representing the RIGHT-ARC transitions.
            #    step 3-1 : marks the 1st (most recently added) item on the stack as a dependent of the 2nd item
            #    step 3-2 : removes the 1st item from the stack.
            if len(self.stack) >= 2:
                second_item = self.stack[-2]
                first_item = self.stack.pop(-1)
                self.dependencies.append( (second_item,first_item) )
        ### END CODE HERE

    def parse(self, transitions):
        """Applies the provided transitions to this PartialParse

        Args:
            transitions: The list of transitions in the order they should be applied
        Returns:
            dependencies: The list of dependencies produced when parsing the sentence. Represented
                          as a list of tuples where each tuple is of the form (head, dependent)
        """
        for transition in transitions:
            self.parse_step(transition)
        return self.dependencies


def minibatch_parse(sentences, model, device, batch_size):
    """Parses a list of sentences in minibatches using a model.

    Args:
        1. sentences: 
            A list of sentences to be parsed each sentence is a list of words
        2. model:
            The model that makes parsing decisions.
            It is assumed to have a function model.predict(partial_parses) that 
             - takes in a list of PartialParses as input 
             - returns a list of transitions predicted for each parse. 
            That is, after calling transitions = model.predict(partial_parses)
            transitions[i] will be the next transition to apply to partial_parses[i].
        3. device: The device to be used
        4. batch_size: The number of PartialParses to include in each minibatch
    Returns:
        dependencies: 
            A list where each element is the dependencies list for a parsed sentence.
            Ordering should be the same as in sentences 
            (i.e., dependencies[i] should contain the parse for sentences[i]).
    """

    ### START CODE HERE
    # Initialize partial_parses as a list of PartialParses
    # one for each sentence in sentences
    partial_parses = [ PartialParse(sentence) for sentence in sentences ]
    # 1. Initialize unfinished parses as a shallow copy of partial parses
    # 2. shallowly copy with the list() factory function
    unfinished_parses = list(partial_parses)
    # while unfinished parses is "not empty" do
    while unfinished_parses:
        # step 1. Take the first batch size parses in unfinished parses as a minibatch
        minibatch = unfinished_parses[:batch_size]
        # step 2. Use the model to predict the next transition for each partial parse in the minibatch 
        predicted_transitions = model.predict(minibatch,device)
        # step 3. Perform a parse step on each partial parse in the minibatch with its predicted transition
        for partial_parse, predicted_transition in zip( minibatch, predicted_transitions ):
            # Performs a single parse step by applying the given transition to this partial parse
            partial_parse.parse_step(predicted_transition)
            # step 4. Remove the completed (empty buffer and stack of size 1) parses from unfinished parses
            # condition 1 : "empty" buffer : not partial_parse.buffer
            # condition 2 : stack of size 1 : len(partial_parse.stack) == 1
            if (not partial_parse.buffer) and len(partial_parse.stack) == 1:
                unfinished_parses.remove(partial_parse)
    # The dependencies for each completed parse in partial_parses.
    dependencies = [ completed_parse.dependencies for completed_parse in partial_parses ]
    ### END CODE HERE

    return dependencies


def test_step(name, transition, stack, buf, deps,
              ex_stack, ex_buf, ex_deps):
    """Tests that a single parse step returns the expected output"""
    pp = PartialParse([])
    pp.stack, pp.buffer, pp.dependencies = stack, buf, deps

    pp.parse_step(transition)
    stack, buf, deps = (tuple(pp.stack), tuple(pp.buffer), tuple(sorted(pp.dependencies)))
    assert stack == ex_stack, \
        "{:} test resulted in stack {:}, expected {:}".format(name, stack, ex_stack)
    assert buf == ex_buf, \
        "{:} test resulted in buffer {:}, expected {:}".format(name, buf, ex_buf)
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)
    print("{:} test passed!".format(name))


def test_parse_step():
    """Simple tests for the PartialParse.parse_step function
    Warning: these are not exhaustive
    """
    test_step("SHIFT", "S", ["ROOT", "the"], ["cat", "sat"], [],
              ("ROOT", "the", "cat"), ("sat",), ())
    test_step("LEFT-ARC", "LA", ["ROOT", "the", "cat"], ["sat"], [],
              ("ROOT", "cat",), ("sat",), (("cat", "the"),))
    test_step("RIGHT-ARC", "RA", ["ROOT", "run", "fast"], [], [],
              ("ROOT", "run",), (), (("run", "fast"),))


def test_parse():
    """Simple tests for the PartialParse.parse function
    Warning: these are not exhaustive
    """
    sentence = ["parse", "this", "sentence"]
    dependencies = PartialParse(sentence).parse(["S", "S", "S", "LA", "RA", "RA"])
    dependencies = tuple(sorted(dependencies))
    expected = (('ROOT', 'parse'), ('parse', 'sentence'), ('sentence', 'this'))
    assert dependencies == expected, \
        "parse test resulted in dependencies {:}, expected {:}".format(dependencies, expected)
    assert tuple(sentence) == ("parse", "this", "sentence"), \
        "parse test failed: the input sentence should not be modified"
    print("parse test passed!")


class DummyModel(object):
    """Dummy model for testing the minibatch_parse function
    First shifts everything onto the stack and then does exclusively right arcs if the first word of
    the sentence is "right", "left" if otherwise.
    """

    def predict(self, partial_parses, device):
        return [("RA" if pp.stack[1] == "right" else "LA") if len(pp.buffer) == 0 else "S"
                for pp in partial_parses]


def test_dependencies(name, deps, ex_deps):
    """Tests the provided dependencies match the expected dependencies"""
    deps = tuple(sorted(deps))
    assert deps == ex_deps, \
        "{:} test resulted in dependency list {:}, expected {:}".format(name, deps, ex_deps)


def test_minibatch_parse():
    """Simple tests for the minibatch_parse function
    Warning: these are not exhaustive
    """
    device = 'cpu'
    sentences = [["right", "arcs", "only"],
                 ["right", "arcs", "only", "again"],
                 ["left", "arcs", "only"],
                 ["left", "arcs", "only", "again"]]
    deps = minibatch_parse(sentences, DummyModel(), device, 2)
    test_dependencies("minibatch_parse", deps[0],
                      (('ROOT', 'right'), ('arcs', 'only'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[1],
                      (('ROOT', 'right'), ('arcs', 'only'), ('only', 'again'), ('right', 'arcs')))
    test_dependencies("minibatch_parse", deps[2],
                      (('only', 'ROOT'), ('only', 'arcs'), ('only', 'left')))
    test_dependencies("minibatch_parse", deps[3],
                      (('again', 'ROOT'), ('again', 'arcs'), ('again', 'left'), ('again', 'only')))
    print("minibatch_parse test passed!")


if __name__ == '__main__':
    test_parse_step()
    test_parse()
    test_minibatch_parse()
