# Plots for Bayesian regression with linear basis function and predictive distributions
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
function polynomial_basis(x::Float64,j)
    return x^j
end 

"""
    fourier_basis(x,j)

Computes the `j`th basis function value at `x` for the Fourier basis functions

# Examples
```jldoctest
julia> fourier_basis(2.0,3)
0.10944260690631982

julia> fourier_basis(2.0,0)
1.0
```
"""
function fourier_basis(x::Float64,j)
    return if(j % 2 == 0) cos(π*j/2*x) else sind(π*(j-1)/2*x) end
end 

"""
    gauss_basis(x,j)

Computes the `j`th basis function value at `x` for the Gaussian basis functions

# Examples
```jldoctest
julia> gauss_basis(2.0,3)
0.24197072451914337

julia> gauss_basis(2.0,0)
0.05399096651318806
```
"""
function gauss_basis(x::Float64,j;σ=1)
    return pdf(Normal(j,σ),x)
end 

"""
    sigmoid_basis(x,j)

Computes the `j`th basis function value at `x` for the sigmoid basis functions

# Examples
```jldoctest
julia> sigmoid_basis(2.0,3)
0.2689414213699951

julia> sigmoid_basis(2.0,0)
0.8807970779778824
```
"""
function sigmoid_basis(x::Float64,j)
    return exp(x-j)/(1+exp(x-j))
end 

"""
    generate_data(n, f, σ=0.1, from=0, to=1)

Generates a dataset of `n` observatiosn using a given function `f` with additive, zero-mean Gaussian noise with standard deviation of `σ` at fixed intervals in the range `from` to `to`

# Examples
```jldoctest
julia> generate_data(11, x -> sin(x*π), σ=0.15, from=0, to=1)
(x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], y = [0.08927038376370136; 0.7746052791966367; … ; 0.6234703379577893; -0.05348756498832998;;])
```    
"""
function generate_data(n::Int64, f; σ=0.1, from=0, to=1)
    xs = collect(range(from,to,n))
    ys = map(f, xs) + randn((n,1))*σ
    return (x = xs, y = ys)
end 

"""
    plot_Bayesian_fit(train_data; σ=0.1, τ=0.5, ϕ=x -> map(j -> polynomial_basis(x,j), 1:2), color=:blue)

Generates a plot of the Bayesian posterior for the training data `train_data` using the feature map `ϕ`. For the prior, it's assumed to be zero mean with standard deviation of `τ`; the likelihood is assumed to have standard deviation of `σ`. Using `color` for the line and ribbon.
"""
function plot_Bayesian_fit(train_data; σ=0.1, τ=0.5, ϕ=x -> map(j -> polynomial_basis(x,j), 1:2), color=:blue)
    # generate the feature representation of the input data
    Φ = transpose(hcat([ϕ(x) for x in train_data.x]...))

    # compute the parameters of the Bayesian posterior over w
    Σinv = 1/τ^2*Diagonal(ones(size(Φ,2)))
    y = train_data.y
    S = inv(Σinv + 1/σ^2*Φ'*Φ)
    μ = S*1/σ^2*Φ'*y
    
    # compute the mean and standard deviation of the predictive distribution over all test points 
    xs = collect(range(-0.3,1.3,100))
    Φ_test = generate_features(xs, ϕ=ϕ)
    pred = map(ϕ -> (vec(μ)'*ϕ, sqrt(σ^2+ϕ'*S*ϕ)), eachrow(Φ_test))
    plot!(xs,map(x->x[1],pred), ribbon=map(x->x[2],pred), fillalpha=0.2, linewidth=3, color=color)
end

# generates training data 
Random.seed!(41)
# train_data = generate_data(5, x -> sin(x*π), σ=0.15, from=0, to=1)
train_data = generate_data(11, x -> sin(x*π), σ=0.15, from=0, to=1)
# train_data = generate_data(11, x -> 1/(1+x^2), σ=0.05, from=-5, to=5)


p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, 
         xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
xlabel!(L"x")
ylabel!(L"y")
plot_Bayesian_fit(train_data, τ=1, color=:blue, ϕ=x -> map(j -> polynomial_basis(x,j), 0:6))
plot_Bayesian_fit(train_data, τ=10, color=:red, ϕ=x -> map(j -> polynomial_basis(x,j), 0:6))
display(p)
savefig("~/Downloads/poly_fit.svg")

# p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, 
#          xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
# xlabel!(L"x")
# ylabel!(L"y")
# plot_Bayesian_fit(train_data, τ=1, color=:blue, ϕ=x -> map(j -> fourier_basis(x,j), 0:15))
# plot_Bayesian_fit(train_data, τ=10, color=:red, ϕ=x -> map(j -> fourier_basis(x,j), 0:15))
# display(p)

p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, 
         xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
xlabel!(L"x")
ylabel!(L"y")
plot_Bayesian_fit(train_data, τ=1, color=:blue, ϕ=x -> map(j -> gauss_basis(x,j/20, σ=0.15), 0:20))
plot_Bayesian_fit(train_data, τ=10, color=:red, ϕ=x -> map(j -> gauss_basis(x,j/20, σ=0.15), 0:20))
display(p)
savefig("~/Downloads/gauss_fit.svg")
