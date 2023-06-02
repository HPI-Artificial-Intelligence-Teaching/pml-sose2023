# Plots for Kernels slides
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    polynomial_basis(x,j)

Computes the `j`th basis function value at `x` for polynomial basis functions

# Examples
```jldoctest
julia> polynomial_basis(2.0,3)
8.0

julia> polynomial_basis(2.0,0)
1.0
```
"""
function polynomial_basis(x::Float64, j)
    return x^j
end

"""
    gauss_basis(x,j; σ = 1)

Computes the `j`th basis function value at `x` for the Gaussian basis functions with variance `σ`

# Examples
```jldoctest
julia> gauss_basis(2.0,3)
0.24197072451914337

julia> gauss_basis(2.0,0)
0.05399096651318806

julia> gauss_basis(2.0,0; σ = 2)
0.12098536225957168
```
"""
function gauss_basis(x::Float64, j; σ = 1)
    return pdf(Normal(j, σ), x)
end

"""
    sigmoid_basis(x,j; σ = 1)

Computes the `j`th basis function value at `x` for the sigmoid basis functions at length scale `σ`

# Examples
```jldoctest
julia> sigmoid_basis(2.0,3)
0.2689414213699951

julia> sigmoid_basis(2.0,0)
0.8807970779778824

julia> sigmoid_basis(2.0,0, σ = 2)
0.7310585786300049
```
"""
function sigmoid_basis(x, j; σ = 1)
    return exp((x - j) / σ) / (1 + exp((x - j) / σ))
end

"""
    plot_kernels(ϕ, x; xlim=(-1,1), base_name="~/Downloads/kernel")

Plots the basis functions of `ϕ` and the kernel evaluations at `x` 
"""
function plot_kernels(ϕ, x; xlim = (-1, 1), base_name = "~/Downloads/kernel")
    N = length(ϕ(0))
    xs = range(start = xlim[1], stop = xlim[2], length = 100)
    yss = map(x -> ϕ(x), xs)
    ys = map(ys -> ys' * ϕ(x), yss)
    p = plot(
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"\phi(x)")
    for i = 1:N
        plot!(xs, map(ys -> ys[i], yss), color = i, linewidth = 3)
    end
    display(p)
    savefig(p, base_name * "_basis.svg")

    p = plot(
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"k(x,x^\prime)")
    plot!(xs, ys, color = :blue, linewidth = 3)
    plot!([x, x], [minimum(ys), maximum(ys)], linewdith = 0.5, color = :red)
    display(p)
    savefig(p, base_name * "_kernel.svg")
end

plot_kernels(
    x -> map(j -> polynomial_basis(x, j), 1:11),
    -0.5,
    base_name = "~/Downloads/poly",
)
plot_kernels(
    x -> map(j -> gauss_basis(x, j / 5 - 1, σ = 0.15), 0:10),
    0,
    base_name = "~/Downloads/gauss",
)
plot_kernels(
    x -> map(j -> sigmoid_basis(x, j / 5 - 1, σ = 0.15), 0:10),
    0,
    base_name = "~/Downloads/sigmoid",
)
