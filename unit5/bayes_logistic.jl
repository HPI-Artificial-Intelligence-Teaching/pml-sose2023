# Plots for Logistic Regression Learning
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
    plot_logistic_regression(n=250; class0_color = :red, class1_color = :blue, class0=1, class1=9, no_iterations = 10, τ = 0.1)

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the one class is the digits with the label `class0` and the other class is the digits with the label `class1` (using the colors `class0_color` and `class1_color`, respectively). Then performs the logistic regression learning and plots the decision surface.
"""
function plot_bayes_logistic_regression(
    n = 250;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 9,
    no_iterations = 10,
    τ = 0.1,
)
    function average_intensity_features(img)
        return [mean(img[1:14, :]), mean(img[15:28, :])]
    end

    function plot_weight(w; y_max = 0.1, y_min = -0.1, x_min = -0.1, x_max = +0.1)
        w = w ./ sqrt(w' * w) * 0.05
        x_min_computed = y_min * w[2] / w[1]
        x_max_computed = y_max * w[2] / w[1]
        xs = range(
            start = min(x_min, x_min_computed),
            stop = max(x_max, x_max_computed),
            length = 100,
        )
        plot!(xs, x -> -x * w[1] / w[2], linewidth = 2, color = :gray)
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
    X = vcat(X0, X1)
    x_max, y_max = 1.2 * maximum(X, dims = 1)
    x_min, y_min = 1.2 * minimum(X, dims = 1)
    y = vcat(zeros(n, 1), 1 * ones(n, 1))
    μ = zeros(2, 1)

    # run Newton-Raphson to compute the mode of the posterior
    for idx = 1:no_iterations
        t = X * μ
        g = exp.(t) ./ (1 .+ exp.(t))
        R = Diagonal(vec(g .* (1 .- g)))
        μ -= inv(X' * R * X + 2 / τ^2 * Diagonal(ones(2))) * (X' * (g - y) + 2 / τ^2 * μ)
    end

    # compute the inverse of the Hessian for the covariance
    t = X * μ
    g = exp.(t) ./ (1 .+ exp.(t))
    R = Diagonal(vec(g .* (1 .- g)))
    Σ = inv(X' * R * X + 2 / τ^2 * Diagonal(ones(2)))
    function posterior(x1, x2)
        ϕ = [x1 x2]'
        m = (μ'*ϕ)[1]
        s2 = (ϕ'*Σ*ϕ)[1] + π / 8
        N = Normal(m, sqrt(s2))
        return (cdf(N, 0))
    end

    # plot the posterior probability of each class prediction 
    X = range(start = x_min, stop = x_max, length = 100)
    Y = range(start = y_min, stop = y_max, length = 100)
    p = heatmap!(X, Y, (x1, x2) -> posterior(x1, x2), legend = false, color = :berlin)
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
    plot_weight(μ, y_min = y_min, y_max = y_max, x_min = x_min, x_max = x_max)

    # decorate the plot
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
end

"""
    learn_logistic_regression(n=250; class0=1, class1=9, ϵ=1e-4, τ = 0.1)

Learns a logisit regression model for `n` examples of the MNIST examples of the digits with the label `class0` and `class1` and outputs the learned weight vector as an image (using `ϵ` as a stopping criterion)
"""
function learn_bayes_logistic_regression(
    n = 250;
    class0 = 1,
    class1 = 9,
    ϵ = 1e-4,
    τ = 0.1,
    base = "~/Downloads/bayes_log",
)
    # transformation on a weight vector to plot it as an image
    function plot_transform(x)
        return (hcat(map(r -> r[28:-1:1], eachrow(reshape(x[(28*28):-1:1], 28, 28)'))...)')
    end

    # plot the raw data 
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> vec(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> vec(X[:, :, idx1[i]]), 1:n)...)'
    mn = (sum(X1, dims = 1) + sum(X0, dims = 1)) / (2 * n)
    X0 -= repeat(mn, n)
    X1 -= repeat(mn, n)

    # augment the data with a one column for the bias and concatenate the two datasets
    X = vcat(X0, X1)
    y = vcat(zeros(n, 1), 1 * ones(n, 1))
    μ = zeros(size(X, 2), 1)

    Δ = 2 * ϵ
    while (Δ > ϵ)
        t = X * μ
        g = exp.(t) ./ (1 .+ exp.(t))
        R = Diagonal(vec(g .* (1 .- g)))
        Δμ =
            inv(X' * R * X + 2 / τ^2 * Diagonal(ones(size(X, 2)))) *
            (X' * (g - y) + 2 / τ^2 * μ)
        μ -= Δμ
        Δ = norm(Δμ)
        println("Δ = ", Δ)
    end

    # compute the inverse of the Hessian for the covariance
    t = X * μ
    g = exp.(t) ./ (1 .+ exp.(t))
    R = Diagonal(vec(g .* (1 .- g)))
    Σ = pinv(X' * R * X + 2 / τ^2 * Diagonal(ones(size(X, 2))))

    # plot the mean of the weight vector distribution
    heatmap(
        plot_transform(μ),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    savefig(base * "_mean.png")

    # plot the variance of the weight vector distribution
    heatmap(
        plot_transform(diag(Σ)),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    savefig(base * "_variance.png")
end

plot_bayes_logistic_regression(50, τ = 20)
savefig("~/Downloads/bayes_logistic_50_tau=20.png")
plot_bayes_logistic_regression(50, τ = 100)
savefig("~/Downloads/bayes_logistic_50_tau=100.png")
plot_bayes_logistic_regression(250, τ = 5)
savefig("~/Downloads/bayes_logistic_250_tau=5.png")
plot_bayes_logistic_regression(250, τ = 20)
savefig("~/Downloads/bayes_logistic_250_tau=20.png")

learn_bayes_logistic_regression(1000, class0 = 1, class1 = 8, base="~/Downloads/bayes_logistic_1_vs_8")
learn_bayes_logistic_regression(1000, class0 = 1, class1 = 9, base="~/Downloads/bayes_logistic_1_vs_9")
learn_bayes_logistic_regression(1000, class0 = 3, class1 = 4, base="~/Downloads/bayes_logistic_3_vs_4")
