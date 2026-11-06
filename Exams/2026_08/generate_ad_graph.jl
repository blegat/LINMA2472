using ExprGraphExplorer
using LinearAlgebra

struct EmptyMetadata end
EmptyMetadata(::Any) = EmptyMetadata()

Node = ExprNode{Any,EmptyMetadata}
x = Node([1, 2])
Q = Node([1 2; 3 4])
s = Q * x
t = Node(:dot, Node[x, s], dot(x.value, s.value))
one = Node(:constant, Node[], 1.0)
f = Node(:-, Node[t, one], t.value - one.value)
graph = ExprGraph(f; names=IdDict(x => "x", Q => "Q", s => "s", t => "t", one => "1", f => "f"))
exam_state = capture_frame(graph, ""; show_metadata=false)
save_svg(joinpath(@__DIR__, "ad_graph.svg"), graph, exam_state; exam=true)
save_png(joinpath(@__DIR__, "ad_graph.png"), graph, exam_state; exam=true)
save_eps(joinpath(@__DIR__, "ad_graph.eps"), graph, exam_state; exam=true)
