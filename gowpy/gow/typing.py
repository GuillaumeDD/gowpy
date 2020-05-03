from typing import Tuple, Callable, Sequence, Set, Union

Token = str
Tokenized_document = Sequence[Token]
Tokenizer = Callable[[str], Tokenized_document]
Node = int
Nodes = Set[Node]

Edge_label = Tuple[int, int]
Edge = Tuple[Node, Node]
Edge_with_code = Tuple[Node, Node, int]
Edges = Union[Set[Edge], Set[Edge_with_code]]
