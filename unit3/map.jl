# Plots for polynomial regression with MAP solution
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots


"""
    generate_data(n, f, σ=0.1)

Generates a dataset of n observatiosn using a given function f with additive, zero-mean Gaussian noise with standard deviation of σ
"""
function generate_data(n::Int64, f; σ=0.1, from=0, to=1)
    xs = collect(range(from,to,n))
    ys = map(f, xs) + randn((n,1))*σ
    return (x = xs, y = ys)
end 

"""
    generate_features(data)

Generates the ferature matrix from the raw data
"""
function generate_features(data; max_degree=2, min_degree=1)
    return hcat([data.^i for i=min_degree:max_degree]...)
end 

"""
    fit_polynomial(train_data; degree=2, λ=1, min_degree=1, color=:blue)

Generates a regularized least-square fit of a polynomial of degree and plots the fit
"""
function fit_polynomial(train_data; λ=1, degree=2, min_degree=1, color=:blue, threed=false)
    train_X = generate_features(train_data.x, min_degree=min_degree, max_degree=degree)
    w = inv(train_X' * train_X + Diagonal(λ * ones(size(train_X, 2)))) * train_X' * train_data.y
    xs = collect(range(minimum(train_data.x),maximum(train_data.x),100))
    test_X = generate_features(xs, min_degree=min_degree, max_degree=degree)
    ys = test_X * w
    if (threed)
        plot!(xs,ys, zeros(100), linewidth=3, color=color)
    else        
        plot!(xs,ys, linewidth=3, color=color)
    end
    return w
end

function likelihood(w, X, y; σ=0.15)
    return pdf(MvNormal(X*w, I*σ^2),y)
end 

# generates training data 
Random.seed!(41)
# train_data = generate_data(5, x -> sin(x*π), σ=0.15, from=0, to=1)
train_data = generate_data(11, x -> sin(x*π), σ=0.15, from=0, to=1)
# train_data = generate_data(11, x -> 1/(1+x^2), σ=0.05, from=-5, to=5)

# plot of the training data with best fit of 6th order polynomial
p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
xlabel!(L"x")
ylabel!(L"y")
fit_polynomial(train_data, λ=0, min_degree=0, degree=6, color=:blue)
display(p)

# plot of the training data with best fit of 6th order polynomial
p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
xlabel!(L"x")
ylabel!(L"y")
fit_polynomial(train_data, λ=1e-3, min_degree=0, degree=6, color=:blue)
display(p)

# plot of the training data with best fit of 6th order polynomial
p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
xlabel!(L"x")
ylabel!(L"y")
fit_polynomial(train_data, λ=1e-2, min_degree=0, degree=6, color=:blue)
display(p)

# plot of the training data with best fit of 6th order polynomial
p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
xlabel!(L"x")
ylabel!(L"y")
fit_polynomial(train_data, λ=1e-1, min_degree=0, degree=6, color=:blue)
display(p)

# plot of the training data with best fit of 6th order polynomial
p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
xlabel!(L"x")
ylabel!(L"y")
fit_polynomial(train_data, λ=1e-0, min_degree=0, degree=6, color=:blue)
display(p)

# plot of the training data with best fit of 6th order polynomial for all values of the regularizer
p = plot(train_data.x, train_data.y, seriestype=:scatter, legend=false, color = :orange, xtickfontsize=14, ytickfontsize=14, xguidefontsize=16, yguidefontsize=16)
xlabel!(L"x")
ylabel!(L"y")
fit_polynomial(train_data, λ=0, min_degree=0, degree=6, color=:blue)
fit_polynomial(train_data, λ=1e-3, min_degree=0, degree=6, color=:red)
fit_polynomial(train_data, λ=1e-2, min_degree=0, degree=6, color=:green)
fit_polynomial(train_data, λ=1e-1, min_degree=0, degree=6, color=:black)
fit_polynomial(train_data, λ=1e-0, min_degree=0, degree=6, color=:purple)
display(p)