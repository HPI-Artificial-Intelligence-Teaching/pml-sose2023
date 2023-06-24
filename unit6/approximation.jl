# Plots for aspproximate inference for a 1D Gaussian
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    plot_approximation(μ = 0, σ = 1)

Generates a plot of the true and approximate posterior (marginal) and likelihood (message) for a 1D truncated (at zero) Gaussian. `μ` and `σ` are the mean and variance of the prior, respectively.
"""
function plot_approximation(
    μ = 0,
    σ = 1,
)
    # adds a plot for a function
    function plot_function(f; color=:blue, style=:solid)
        plot!(xs, f, linewidth = 3, color = color, style = style)
    end

    # compute the mean and variance of the best approximation 
    t = μ/σ
    v = pdf(Normal(), t)/cdf(Normal(), t)
    μ_approx_posterior = μ + σ * v
    σ_approx_posterior = sqrt(σ * σ * (1 - (v * (v + t))))

    τ_prior = μ/(σ^2)
    ρ_prior = 1/(σ^2)
    τ_approx_posterior = μ_approx_posterior / (σ_approx_posterior^2)
    ρ_approx_posterior = 1 / (σ_approx_posterior^2)
    τ_approx_likelihood = τ_approx_posterior - τ_prior
    ρ_approx_likelihood = ρ_approx_posterior - ρ_prior
    μ_approx_likelihood = τ_approx_likelihood / ρ_approx_likelihood
    σ_approx_likelihood = sqrt(1 / ρ_approx_likelihood)

    println("σ2 (prior) = ", σ^2, "   σ2 (likel) = ", σ_approx_likelihood^2, "    σ2 (posterior) = ", σ_approx_posterior^2)

    # draw all the plots
    d = Normal(μ, σ)
    xs = range(start = μ-3*σ, stop = μ+3*σ, length=1000);
    # generic plot parameter
    p = plot(
        legend = false,
        color = :blue,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"p(x)")
    
    # plot the prior PDF
    plot_function(x -> pdf(d, x); color = :blue)
    # plot the factor function (true likelihood)
    plot_function(x -> (x > 0) ? 1 : 0; color = :red)
    # plot the true posterior PDF
    Z = 1 - cdf(d, 0)
    plot_function(x -> (x > 0) ?  (1/Z *  pdf(d, x)) : 0; color = :black)

    # plot the approximate PDF of the posterior
    plot_function(x -> pdf(Normal(μ_approx_posterior, σ_approx_posterior), x); color = :black, style = :dot)
    # plot the approximate PDF of the posterior
    plot_function(x -> pdf(Normal(μ_approx_likelihood, σ_approx_likelihood), x); color = :red, style = :dot)

    display(p)
end

plot_approximation(0, 0.8)
savefig("~/Downloads/approximation.svg")