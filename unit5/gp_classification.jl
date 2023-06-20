# Plots for Gaussian Processes classification
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots
using MLDatasets


"""
    RBF_kernel(x1,x2;λ=1)

Computes the radial basis function kernel value at `x1` and `x2` with length scale `λ`

# Examples
```jldoctest
julia> RBF_kernel([0,0], [1,1], λ=1)
0.1353352832366127

julia> RBF_kernel([0,0], [1,2], λ=1)
0.006737946999085467

julia> RBF_kernel([0,0], [1,1], λ=2)
0.6065306597126334
```
"""
function RBF_kernel(x1, x2; λ = 0.05)
    return exp(-((x1[1] - x2[1])^2 + (x1[2] - x2[2])^2) / λ^2)
end

"""
    X, y = extract_data(n=9; class0=1, class1=8)

Extract a dataset of two times `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the first `n` examples are one class is the digits with the label `class0` and the other `n` examples are from class is the digits with the label `class1`. 
"""
function extract_data(n = 9; class0 = 1, class1 = 8)
    function average_intensity_features(img)
        return [mean(img[1:14, :]), mean(img[15:28, :])]
    end

    # extract the data
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> average_intensity_features(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> average_intensity_features(X[:, :, idx1[i]]), 1:n)...)'
    X = vcat(X0, X1)
    mn = mean(X, dims = 1)
    X -= repeat(mn, size(X, 1))
    X0 -= repeat(mn, size(X0, 1))
    X1 -= repeat(mn, size(X1, 1))
    y = vcat(zeros(n, 1), 1 * ones(n, 1))

    return X, y
end

"""
    plot_gp_probit_regression(n=250; class0_color = :red, class1_color = :blue, class0=1, class1=9 no_iterations=3)

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the one class is the digits with the label `class0` and the other class is the digits with the label `class1` (using the colors `class0_color` and `class1_color`, respectively). Then performs the Gaussian Process (logit) regression learning with Laplace approximation and plots the decision surface.
"""
function plot_gp_logit_regression(
    n = 250;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 9,
    no_iterations = 10,
    kernel = RBF_kernel,
)
    function average_intensity_features(img)
        return [mean(img[1:14, :]), mean(img[15:28, :])]
    end

    # plot the raw data 
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> average_intensity_features(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> average_intensity_features(X[:, :, idx1[i]]), 1:n)...)'
    mn = (sum(X1, dims = 1) + sum(X0, dims = 1)) / (2 * n)
    X0 -= repeat(mn, n)
    X1 -= repeat(mn, n)

    # augment the data with a one column for the bias and concatenate the two datasets
    trainX = vcat(X0, X1)
    x_max, y_max = 1.2 * maximum(trainX, dims = 1)
    x_min, y_min = 1.2 * minimum(trainX, dims = 1)
    trainY = vcat(zeros(n, 1), 1 * ones(n, 1))

    # compute the kernel matrix
    C = [kernel(x1, x2) for x1 in eachrow(trainX), x2 in eachrow(trainX)]
    Cinv = inv(C)
    f = hcat(zeros(size(C)[1]))

    # # run Newton-Raphson to compute the mode of the posterior
    for idx = 1:no_iterations
        g = exp.(f) ./ (1 .+ exp.(f))
        d = (trainY - g) - Cinv * f
        H = Diagonal(vec(-g .* (1 .- g))) - Cinv
        Δ = inv(H) * d
        println("Iteration ", idx, ": e = ", norm(Δ), ": d = ", norm(d))
        f -= Δ
    end
    m = f

    # compute the inverse of the Hessian for the covariance
    g = exp.(m) ./ (1 .+ exp.(m))
    d = (trainY - g) - Cinv * m
    H = Cinv - Diagonal(vec(-g .* (1 .- g)))
    A = inv(H)
    α = Cinv * m
    B = Cinv - Cinv * A * Cinv
    function posterior(x1, x2)
        k = [kernel(x, [x1, x2]) for x in eachrow(trainX)]
        μ = (k'*α)[1]
        σ2 = kernel([x1, x2], [x1, x2]) - k' * B * k + π / 8
        N = Normal(μ, sqrt(σ2))
        return (cdf(N, 0))
    end

    # plot the posterior probability of each class prediction 
    X = range(start = x_min, stop = x_max, length = 100)
    Y = range(start = y_min, stop = y_max, length = 100)
    p = heatmap!(X, Y, (x1, x2) -> posterior(x1, x2), legend = false, color = :berlin)
    contour!(
        X,
        Y,
        (x1, x2) -> posterior(x1, x2),
        levels = [0.5],
        linewidth = 1,
        color = :white,
    )
    plot!(
        X0[:, 1],
        X0[:, 2],
        legend = false,
        seriestype = :scatter,
        color = class0_color,
        alpha = 0.5,
        aspect_ratio = :equal,
    )
    plot!(X1[:, 1], X1[:, 2], seriestype = :scatter, color = class1_color, alpha = 0.5)

    # decorate the plot
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
end

plot_gp_logit_regression(50, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.5))
savefig("~/Downloads/gp_logistic_50_lambda=0.5.png")
plot_gp_logit_regression(50, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.05))
savefig("~/Downloads/gp_logistic_50_lambda=0.05.png")
plot_gp_logit_regression(250, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.5))
savefig("~/Downloads/gp_logistic_250_lambda=0.5.png")
plot_gp_logit_regression(250, kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.05))
savefig("~/Downloads/gp_logistic_250_lambda=0.05.png")
