# Plots for the distributions
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Plots
using Distributions
using LaTeXStrings

"""
    var_bernoulli(π=0.5)

Returns the variance of a Bernoulli with success probability `π`.
# Examples
```jldoctest
julia> p = var_bernoulli()
0.25
```
"""
function var_bernoulli(π = 0.5)
    return π * (1 - π)
end

"""
    sigmoid(x=0)

Returns the value of the logistic sigmoid at `x`
# Examples
```jldoctest
julia> sigmoid(1)
0.7310585786300049
```
"""
function sigmoid(x = 0)
    return exp(x) / (1 + exp(x))
end

"""
    normal_density(x,μ=0,σ=1)

Returns the value of the normal density at `x` with mean `μ` and standard deviation `σ`
# Examples
```jldoctest
julia> normal_density(0)
0.3989422804014327
```
"""
function normal_density(x, μ = 0, σ = 1)
    return 1 / (sqrt(2 * π) * σ) * exp(-(x - μ)^2 / (2 * σ^2))
end

# plot the variance of the Bernoulli
πs = 0:0.01:1.0
p = plot(
    legend = false,
    xtickfontsize = 14,
    ytickfontsize = 14,
    xguidefontsize = 16,
    yguidefontsize = 16,
)
plot!(πs, map(π -> var_bernoulli(π), πs), linewidth = 3)
# scatter!(πs,map(π -> var_bernoulli(π),πs))
ylabel!(L"{\mathrm{var}}[X]")
xlabel!(L"π")
display(p)

# plot the sigmoid function
xs = -4:0.1:4.0
p = plot(
    legend = false,
    xtickfontsize = 14,
    ytickfontsize = 14,
    xguidefontsize = 16,
    yguidefontsize = 16,
)
plot!(xs, map(x -> sigmoid(x), xs), linewidth = 3)
# scatter!(πs,map(π -> var_bernoulli(π),πs))
ylabel!(L"{\mathrm{sigmoid}}(x)")
xlabel!(L"x")
display(p)

# plot the normal density function
xs = -3:0.1:3.0
p = plot(
    legend = false,
    xtickfontsize = 14,
    ytickfontsize = 14,
    xguidefontsize = 16,
    yguidefontsize = 16,
)
plot!(xs, map(x -> normal_density(x), xs), linewidth = 3)
# scatter!(πs,map(π -> var_bernoulli(π),πs))
ylabel!(L"N(x)")
xlabel!(L"x")
display(p)

# plot the normal CDF and of the sigmoid function
n = Normal()
xs = -4:0.1:4.0
p = plot(
    xtickfontsize = 14,
    ytickfontsize = 14,
    xguidefontsize = 16,
    yguidefontsize = 16,
    legendfontsize = 16,
)
plot!(
    xs,
    map(x -> cdf(n, sqrt(pi / 8) * x), xs),
    label = L"\Phi\left(\sqrt{\frac{\pi}{8}}x\right)",
    linewidth = 3,
)
plot!(
    xs,
    map(x -> sigmoid(x), xs),
    label = L"{\mathrm{logit}}(x)",
    linewidth = 3,
    linecolor = :red,
)
xlabel!(L"x")
display(p)
