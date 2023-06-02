# Plots for Bayesian model selection and evidence maximization
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

prior = Uniform(0, 1)
posterior = Uniform(0.4, 0.6)
xs = range(start = -0.2, stop = 1.2, length = 100)
p = plot(
    xs,
    map(x -> pdf(posterior, x), xs),
    linewidth = 3,
    color = :red,
    legend = false,
    xaxis = nothing,
    yaxis = nothing,
    xtickfontsize = 14,
    ytickfontsize = 18,
    xguidefontsize = 24,
    yguidefontsize = 24,
)
plot!(xs, map(x -> pdf(prior, x), xs), linewidth = 3, color = :blue)
quiver!(
    [0.6, 0.4],
    [5.25, 5.25],
    quiver = ([-0.2, 0.2], [0, 0]),
    linewidth = 3,
    color = :black,
)
annotate!([0.5], [5.7], text(L"\Delta_\mathrm{posterior}", 24))
quiver!([1, 0], [1.25, 1.25], quiver = ([-1, 1], [0, 0]), linewidth = 3, color = :black)
annotate!([0.75], [1.7], text(L"\Delta_\mathrm{prior}", 24))
ylims!(-0.1, 6)
xlabel!(L"w")
ylabel!(L"P(w|D)")
display(p)
savefig("~/Downloads/bayesian_model_selection.svg")
