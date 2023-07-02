# Plots for truncated 1D Gaussian
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    plot_v_w_functions(;base = "~/Downloads/")

Generates a plot of the helper functions v and w in the truncated 1D Gaussian case. 
"""
function plot_v_w_functions(;base = "~/Downloads/")
    ts = range(start = -6, stop = 6, length=1000)

    function v(t)
        return (pdf(Normal(), t)/cdf(Normal(), t))
    end

    function w(t)
        vt = v(t)
        return (vt * (vt + t))
    end

    # plot the v function
    p = plot(
        ts,
        v,
        linewidth = 3,
        legend = false,
        color = :blue,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"t")
    ylabel!(L"v(t)")
    display(p)
    savefig(base * "v.svg")

    # plot the w function
    p = plot(
        ts,
        w,
        linewidth = 3,
        legend = false,
        color = :blue,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"t")
    ylabel!(L"w(t)")
    display(p)
    savefig(base * "w.svg")
end

"""
    plot_truncated_integrals(;μ=1, σ=1, base = "~/Downloads/")

Generates plots for the two ways to express the normalization constant 
"""
function plot_truncated_integrals(;μ=1, σ=1, base = "~/Downloads/")
    x_min, x_max = min(-4*σ, μ-4*σ), max(4*σ, μ+4*σ)
    xs = range(start = x_min, stop = x_max, length=1000)
    d1 = Normal(μ,σ)
    d2 = Normal(0,σ)

    # create integral shape for case 1
    pts1 = [(0.0,0.0)]
    xss = range(start = 0, stop = x_max, length=500)
    for i = 1:500
        push!(pts1, (xss[i], pdf(d1, xss[i])))
    end
    for i = 500:-1:1
        push!(pts1, (xss[i], 0))
    end
    push!(pts1, (0.0,0.0))

    # create integral shape for case 2
    pts2 = [(float(μ),0.0)]
    xss = range(start = x_min, stop = μ, length=500)
    for i = 500:-1:1
        push!(pts2, (xss[i], 0))
    end
    for i = 1:500
        push!(pts2, (xss[i], pdf(d2, xss[i])))
    end
    push!(pts2, (μ,0.0))

    # plot the v function
    p = plot(
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        xticks = ([0, μ], [L"0", L"\mu"]),
    )
    plot!(Shape(pts1), fillcolor = :red, fillalpha = 0.2, linewidth = 0.5)
    plot!(xs, x -> pdf(d1, x), linewidth = 3, color = :blue)
    ylabel!(L"N(x;\mu,\sigma^2)")
    display(p)
    savefig(base * "N1.svg")

    # plot the w function
    p = plot(
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        xticks = ([0, μ], [L"0", L"\mu"]),
    )
    plot!(Shape(pts2), fillcolor = :red, fillalpha = 0.2, linewidth = 0.5)
    plot!(xs, x -> pdf(d2, x), linewidth = 3, color = :blue)
    ylabel!(L"N(x;0,\sigma^2)")
    display(p)
    savefig(base * "N2.svg")
end



plot_v_w_functions(base="~/Downloads/")
plot_truncated_integrals(base="~/Downloads/")