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
    p = plot_negative_log_likelihood(X, ys; scale=1)

Plots the negative log-likelihood for the dataset `X` with 0/1-labels `ys`. The scale of the logistic sigmoind is `scale`. Returns the handle to the plot.
"""
function plot_negative_log_likelihood(X, ys; scale = 1)
    # the negative log-likelihood for this particular dataset
    function negative_log_likelihood(w)
        gs = exp.(X * w) ./ (1 .+ exp.(X * w))
        return (sum(map((g, y) -> (y == 0) ? -log2(1 - g) : -log2(g), gs, ys)))
    end

    # plot the likelihood over the weight space
    θs = range(0, stop = 2 * π, length = 300)
    p = plot(
        θs,
        θ -> negative_log_likelihood([scale * sin(θ); scale * cos(θ)]),
        color = :blue,
        linewidth = 3,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"\theta")
    ylabel!(L"-\log_2\left(P\left(\mathbf{y}|X,\mathbf{w}\left(\theta\right)\right)\right)")

    return p
end

"""
    plot_scale_video(;n=9, y_max=150, filename="~/Downloads/logit_scale.gif")

Creates a video of the effect of the logit scale on the negative log-likelihood. A dataset of `n` points is generated and the y-axis is fixed to 0 .. `y_max`. The video is saved in `filename`.
"""
function plot_scale_video(; n = 9, y_max = 150, filename = "~/Downloads/logit_scale.gif")
    X, y = extract_data(n)
    gr()
    anim = @animate for scale ∈ range(1, 100, length = 200)
        s = round(scale, digits = 1)
        p = plot_negative_log_likelihood(X, y, scale = scale)
        ylims!(0, y_max)
        title!(L"\mathrm{Scale} = %$s")
        display(p)
    end
    gif(anim, filename, fps = 10)
end

"""
    plot_logistic_regression(n=250; class0_color = :red, class1_color = :blue, class0=1, class1=9, no_iterations=3)

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the one class is the digits with the label `class0` and the other class is the digits with the label `class1` (using the colors `class0_color` and `class1_color`, respectively). Then performs the logistic regression learning and plots the decision surface.
"""
function plot_logistic_regression(
    n = 250;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 9,
    no_iterations = 3,
)
    function average_intensity_features(img)
        return [mean(img[1:14, :]), mean(img[15:28, :])]
    end

    function plot_weight(w, idx; y_max = 0.1, y_min = -0.1, x_min = -0.1, x_max = +0.1)
        w = w ./ sqrt(w' * w) * 0.05
        x_min_computed = y_min * w[2] / w[1]
        x_max_computed = y_max * w[2] / w[1]
        xs = range(
            start = min(x_min, x_min_computed),
            stop = max(x_max, x_max_computed),
            length = 100,
        )
        plot!(xs, x -> -x * w[1] / w[2], linewidth = 2, color = :black)
        plot!([0, w[1]], [0, w[2]], color = :black, arrow = :closed, linewidth = 2)
        annotate!(w[1] + 0.0008, w[2] + 0.0008, text(L"\mathbf{w}_{%$idx}", 14))
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
    p = plot(
        X0[:, 1],
        X0[:, 2],
        legend = false,
        seriestype = :scatter,
        color = class0_color,
        alpha = 0.5,
        aspect_ratio = :equal,
    )
    plot!(X1[:, 1], X1[:, 2], seriestype = :scatter, color = class1_color, alpha = 0.5)

    # augment the data with a one column for the bias and concatenate the two datasets
    X = vcat(X0, X1)
    x_max, y_max = 1.2 * maximum(X, dims = 1)
    x_min, y_min = 1.2 * minimum(X, dims = 1)
    y = vcat(zeros(n, 1), 1 * ones(n, 1))
    w = zeros(2, 1)

    for idx = 1:no_iterations
        t = X * w
        g = exp.(t) ./ (1 .+ exp.(t))
        R = Diagonal(vec(g .* (1 .- g)))
        w -= inv(X' * R * X) * X' * (g - y)
        plot_weight(w, idx, y_min = y_min, y_max = y_max, x_min = x_min, x_max = x_max)
    end
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
end

"""
    learn_logistic_regression(n=250; class0=1, class1=9, ϵ=1e-4)

Learns a logisit regression model for `n` examples of the MNIST examples of the digits with the label `class0` and `class1` and outputs the learned weight vector as an image (using `ϵ` as a stopping criterion)
"""
function learn_logistic_regression(
    n = 250;
    class0 = 1,
    class1 = 9,
    no_iterations = 3,
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
    w = zeros(size(X,2), 1)

    # Δ = 2 * ϵ
    # while (Δ > ϵ)
    for i = 1:no_iterations
        t = X * w
        g = exp.(t) ./ (1 .+ exp.(t))
        R = Diagonal(vec(g .* (1 .- g)))
        Δw = pinv(X' * R * X) * (X' * (g - y))
        w -= Δw
        Δ = norm(Δw)
        println("Δ = ", Δ)
    end

    # plot the mean of the weight vector distribution
    p = heatmap(
        plot_transform(w),
        colormap = :grays,
        legend = false,
        aspect_ratio = :equal,
        xaxis = nothing,
        yaxis = nothing,
        bordercolor = :white,
    )
    display(p)
end

X, y = extract_data(500)
p = plot_negative_log_likelihood(X, y)
display(p)

plot_scale_video(filename = "~/Downloads/logit_n=9_scale.gif")
plot_scale_video(filename = "~/Downloads/logit_n=500_scale.gif", n = 500, y_max = 8000)

plot_logistic_regression()
