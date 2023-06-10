# Plots for Perceptron Learning
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
    plot_perceptron_learning(n=8; class0_color = :red, class1_color = :blue, class0=1, class1=8)

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the one class is the digits with the label `class0` and the other class is the digits with the label `class1` (using the colors `class0_color` and `class1_color`, respectively). Then performs the perceptron learning algorithm and plots the time evolution of the learning procedure
"""
function plot_perceptron_learning(
    n = 9;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 8,
)
    function average_intensity_features(img)
        return [mean(img[1:14, :]), mean(img[15:28, :])]
    end

    function plot_weight(w, idx; y_max = 0.1, y_min = -0.1, x_min = -0.1, x_max = +0.1)
        x_min_computed = y_min * w[2] / w[1]
        x_max_computed = y_max * w[2] / w[1]
        xs = range(
            start = max(x_min, x_min_computed),
            stop = min(x_max, x_max_computed),
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
    y = vcat(-1 * ones(n, 1), 1 * ones(n, 1))
    w = zeros(2, 1)

    mistakes = 0
    changes_made = true
    while (changes_made)
        changes_made = false
        for i in eachindex(y)
            if ((y[i]*w'*X[i, :])[1] <= 0)
                w = w + y[i] * X[i, :]
                # println(w)
                mistakes += 1
                plot!(
                    [X[i, 1]],
                    [X[i, 2]],
                    legend = false,
                    seriestype = :scatter,
                    linewidth = 2,
                    color = (y[i] == +1) ? class1_color : class0_color,
                )
                plot_weight(
                    w,
                    mistakes,
                    y_min = y_min,
                    y_max = y_max,
                    x_min = x_min,
                    x_max = x_max,
                )
                changes_made = true
            end
        end
    end
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
end

plot_perceptron_learning()
