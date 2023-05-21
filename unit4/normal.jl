# Plots for multivariate Normal distribution
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    plot_normal(μ, Σ; x1_lim=(-3,3), x2_lim=(-3,3), file_name="marginal.png")

Plots the density function of a 2D Gaussian with mean vector `μ` and covariance `Σ`(with plotting margins at `x1_lim` and `x2_lim`) and stores the picture in the file `file_name`
    
# Examples
```jldoctest
julia> plot_normal(vec([0;0]),[[1 0];[0 1]]; file_name="~/Downloads/std_normal.png")
"/Users/rherbrich/Downloads/std_normal.png"
```
"""
function plot_normal(μ, Σ; x1_lim=(-3,3), x2_lim=(-3,3), file_name="marginal.png")
    # plot the Normal distribution
    x1s = range(x1_lim[1], stop = x1_lim[2], length = 100)
    x2s = range(x2_lim[1], stop = x2_lim[2], length = 100)
    x1s_coarse = range(x1_lim[1], stop = x1_lim[2], length = 30)
    x2s_coarse = range(x2_lim[1], stop = x2_lim[2], length = 30)
    p = surface(x1s, x2s, (x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1], color=:greens, fillalpha=0.1, 
            legend=false, xtickfontsize=14, ytickfontsize=14, ztickfontsize=14,
            xguidefontsize=16, yguidefontsize=16, zguidefontsize=16)
    wireframe!(x1s_coarse, x2s_coarse, (x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1])
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    zlabel!(L"p(x_1,x_2)")
    display(p)
    savefig(p, file_name)
end

"""
    plot_ring(n; α=0.1, file_name="ring.png")

Plots a sample density for a 2D distribution which has zero covariance but strong dependence using a transparency of `α` and outputs the empiricial covariance estaimte of the data on screen as well as saves the file in `file_name`

# Examples
```jldoctest
julia> plot_dist(10000; file_name="~/Downloads/ring.png")
Empirical covariance-0.005441054911227434
"/Users/rherbrich/Downloads/ring.png"
```    
"""
function plot_dist(n; α=0.1, file_name="ring.png")
    θ=2*π*rand(n)
    r = ones(n) + rand(n)
    X = hcat(r.*cos.(θ),r.*sin.(θ))
    μ1, μ2 = map(x ->mean(x), eachcol(X))
    println("Empirical covariance", mean(map(x->(x[1]-μ1)*(x[2]-μ2),eachrow(X))))

    p = plot(X[:,1], X[:,2], seriestype=:scatter, color=:blue, alpha=α, legend=false, aspect_ratio = :equal,
                xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
    savefig(p, file_name)
end

"""
    plot_box_mueller(n; α = 0.1, base_name)

Plots the density function of 2D uniform and the Box-Mueller transform with `n` points at a transparency value of `α` and stores the images in the files with the base name `base_name`

# Examples
```jldoctest
julia> plot_box_mueller(10000, base_name="~/Downloads/")
"/Users/rherbrich/Downloads/normal.png"
```
"""
function plot_box_mueller(n; α = 0.1, base_name)
    X = rand(2, n)
    p = plot(X[1,:],X[2,:],color=:blue, seriestype=:scatter, alpha=α, legend=false, axis_ratio = :equal,
            xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
    savefig(p, base_name * "uniform.png")

    # Box-Mueller transform
    for i = 1:size(X,2)
        X[:,i] = [sqrt(-2*log(X[1,i]))*cos(2*π*X[2,i]);sqrt(-2*log(X[1,i]))*sin(2*π*X[2,i])]
    end
    p = plot(X[1,:],X[2,:],color=:blue, seriestype=:scatter, alpha=0.1, legend=false, axis_ratio = :equal, 
            xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
    savefig(p, base_name * "normal.png")
end

"""
    plot_normal_with_marginal(μ, Σ; x2 = [0.0], x1_lim=(-3,3), x2_lim=(-3,3), base_name="~/Downloads/")

Plots the density function of a 2D Gaussian with mean vector `μ` and covariance `Σ` as well as the marginal distribution at all values of `x2` (with plotting margins at `x1_lim` and `x2_lim`) and storing the image in the directory with base name `base_name`

# Examples
```jldoctest
julia> plot_normal_with_marginal(vec([1;0]),[[1 1];[1 2]]; x2=[-2,0,2], x1_lim=(-5,5), x2_lim=(-5,5), base_name="~/Downloads/")
"/Users/rherbrich/Downloads/marginal.png"
```
"""
function plot_normal_with_marginal(μ, Σ; x2 = [0.0], x1_lim=(-3,3), x2_lim=(-3,3), base_name="~/Downloads/")
    # plot the Normal distribution
    x1s = range(x1_lim[1], stop = x1_lim[2], length = 100)
    x2s = range(x2_lim[1], stop = x2_lim[2], length = 100)
    x1s_coarse = range(x1_lim[1], stop = x1_lim[2], length = 30)
    x2s_coarse = range(x2_lim[1], stop = x2_lim[2], length = 30)
    p = surface(x1s, x2s, (x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1], color=:greens, fillalpha=0.1, 
            legend=false, xtickfontsize=14, ytickfontsize=14, ztickfontsize=14,
            xguidefontsize=16, yguidefontsize=16, zguidefontsize=16)
    wireframe!(x1s_coarse, x2s_coarse, (x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1])
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    zlabel!(L"p(x_1,x_2)")
    display(p)
    savefig(p, base_name * "joint.png")

    # plot all the marginals
    for x in x2
        x2s = x*ones(100)
        pdfs = map((x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1],x1s,x2s)

        path3d!(x1s, x2s, pdfs, color=:red, linewidth=3)
    end
    display(p)
    savefig(p, base_name * "marginal.png")
end

"""
    plot_normal_bayes(μ, σ, a, b, τ; x1 = [], x2 = [], x1_lim=(-3,3), x2_lim=(-3,3), α=0.1, file_name="bayes.png")

Plots the density function of a 2D Gaussian where x1 has mean `μ` and standard deviation `σ` and x2 is given a Normal with mean `a`*x1 + `b` and standard deviation `τ`. Plots the joint distribution with transparency `α` as well as the marginal distributions at all values of `x1` (in blue) and `x2` (in red) using the plotting limits `x1_lim` and `x2_lim`. Stores the images in the file `file_name`.

# Examples
```jldoctest
julia> plot_normal_bayes(0,1,1,1,0.5,x1_lim=(-3,3),x2_lim=(-2,5),x1 = [-2, -1, 0, 1, 2], α=0, file_name="~/Downloads/bayes1.png")
"/Users/rherbrich/Downloads/bayes1.png"
```
"""
function plot_normal_bayes(μ, σ, a, b, τ; x1 = [], x2 = [], x1_lim=(-3,3), x2_lim=(-3,3), α=0.1, file_name="bayes.png")
    μ = [μ; a*μ+b]
    Σ = [[σ^2 σ^2*a];[σ^2*a τ^2+a^2*σ^2]]
    # plot the Normal distribution
    x1s = range(x1_lim[1], stop = x1_lim[2], length = 100)
    x2s = range(x2_lim[1], stop = x2_lim[2], length = 100)
    x1s_coarse = range(x1_lim[1], stop = x1_lim[2], length = 30)
    x2s_coarse = range(x2_lim[1], stop = x2_lim[2], length = 30)
    p = surface(x1s, x2s, (x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1], color=:greens, fillalpha=α, 
            legend=false, xtickfontsize=14, ytickfontsize=14, ztickfontsize=14,
            xguidefontsize=16, yguidefontsize=16, zguidefontsize=16)
    wireframe!(x1s_coarse, x2s_coarse, (x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1])
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    zlabel!(L"p(x_1,x_2)")

    # plot all the marginals
    for x in x1
        x1ss = x * ones(100)
        pdfs = map((x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1], x1ss, x2s)

        path3d!(x1ss,x2s, pdfs, color=:blue, linewidth=3)
    end

    # plot all the marginals
    for x in x2
        x2ss = x * ones(100)
        pdfs = map((x1, x2) -> pdf(MvNormal(μ, Σ),[x1;x2])[1], x1s, x2ss)

        path3d!(x1s,x2ss, pdfs, color=:red, linewidth=3)
    end
    display(p)
    savefig(p, file_name)
end

plot_normal(vec([0;0]),[[1 0];[0 1]]; file_name="~/Downloads/std_normal.png")
plot_dist(10000; file_name="~/Downloads/ring.png")
plot_box_mueller(10000, base_name="~/Downloads/")
plot_normal_with_marginal(vec([1;0]),[[1 1];[1 2]]; x2=[-2,0,2], x1_lim=(-5,5), x2_lim=(-5,5), base_name="~/Downloads/")
plot_normal_bayes(0,1,1,1,0.5,x1_lim=(-3,3),x2_lim=(-2,5),x1 = [-2, -1, 0, 1, 2], α=0, file_name="~/Downloads/bayes1.png")
plot_normal_bayes(0,1,1,1,0.5,x1_lim=(-3,3),x2_lim=(-2,5),x1 = [-2, -1, 0, 1, 2], α=0.1, file_name="~/Downloads/bayes2.png")
plot_normal_bayes(0,1,1,1,0.5,x1_lim=(-3,3),x2_lim=(-2,5),x2 = [-2, -1, 0, 1, 2, 3], α=0.1, file_name="~/Downloads/bayes3.png")