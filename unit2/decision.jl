# Plots for decision boundary
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

function sigmoid(x = 0.5)
    return exp(x) / (1 + exp(x))
end

# plot the sigmoid function
xs = -4:0.1:4.0
p = plot(
    xs,
    map(x -> sigmoid(x), xs),
    legend = :top,
    label = L"p(1|x)",
    linewidth = 3,
    color = :blue,
)
plot!(xs, map(x -> 1 - sigmoid(x), xs), label = L"p(0|x)", linewidth = 3, color = :red)
ylabel!(L"p(y|x)")
xlabel!(L"x")
display(p)
