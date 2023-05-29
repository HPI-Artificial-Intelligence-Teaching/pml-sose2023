# Plots for linear basis function models
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
"""
function polynomial_basis(x::Float64, j)
    return x^j
end

"""
    fourier_basis(x,j)

Computes the `j`th basis function value at `x` for the Fourier basis functions
"""
function fourier_basis(x::Float64, j)
    return cos(π * j * x)
end

"""
    gauss_basis(x,j)

Computes the `j`th basis function value at `x` for the Gaussian basis functions
"""
function gauss_basis(x::Float64, j)
    return pdf(Normal(j, 1), x)
end

"""
    sigmoid_basis(x,j)

Computes the `j`th basis function value at `x` for the sigmoid basis functions
"""
function sigmoid_basis(x::Float64, j)
    return exp(x - j) / (1 + exp(x - j))
end

"""
    f(x,w,basis)

Computes the value of linear basis function at `x` with a parameter vector `w`` and the first `length(w)` many basis functions using `basis`
"""
function f(x::Float64, w, basis)
    y = 0
    for j = 1:length(w)
        y += w[j] * basis(x, j - 1)
    end
    return y
end

"""
    plot_function_sample(basis,n=99,d=5;min_x=0,max_x=5, color1=:blue, color2=:red)

Plots a sample of `n`+1 linear basis functions with the `d` basis function `basis`
"""
function plot_function_sample(
    basis,
    n = 99,
    d = 5;
    min_x = 0,
    max_x = 5,
    color1 = :blue,
    color2 = :red,
)
    xs = range(min_x, max_x, 1000)
    v = randn(d, 1)
    p = plot(
        xs,
        map(x -> f(x, v, basis), xs),
        legend = false,
        linewidth = 2,
        color = color1,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    for i = 1:n
        v = randn(d, 1)
        plot!(
            xs,
            map(x -> f(x, v, basis), xs),
            legend = false,
            linewidth = 0.3,
            color = color2,
        )
    end
    xlabel!(L"x")
    ylabel!(L"f(x)")

    return p
end

# plot a sample of basis functions
Random.seed!(42)
p = plot_function_sample(polynomial_basis, 99, 5, min_x = -5, max_x = 5)
savefig(p, "~/Downloads/poly.png")
display(p)

p = plot_function_sample(fourier_basis, 99, 5, min_x = -2, max_x = 2)
savefig(p, "~/Downloads/fourier.png")
display(p)

p = plot_function_sample(gauss_basis, 99, 5, min_x = -5, max_x = 10)
savefig(p, "~/Downloads/gauss.png")
display(p)

p = plot_function_sample(sigmoid_basis, 99, 5, min_x = -5, max_x = 10)
savefig(p, "~/Downloads/sigmoid.png")
display(p)
