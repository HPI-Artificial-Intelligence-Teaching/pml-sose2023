# Plots for Fisher Discriminants classification
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
    plot_MNIST(n=500; class0_color = :red, class1_color = :blue, class0=1, class1=8, w= [1; 1; -0.25], compute_FDA=false)

Plots `n` examples of MNIST image feature vectors (top- and bottom-half average pixel intensity) where the one class is the digits with the label `class0` and the other class is the digits with the label `class1` (using the colors `class0_color` and `class1_color`, respectively). If `compute_FDA` is false, it uses the vector `w` to plot a decision surface; otherwise, it first computes the FDA solution and then plots the decision surface.
"""
function plot_MNIST(
    n = 500;
    class0_color = :red,
    class1_color = :blue,
    class0 = 1,
    class1 = 8,
    w = [1; 1; -0.25],
    compute_FDA = false,
)
    function average_intensity_features(img)
        return [mean(img[1:14, :]), mean(img[15:28, :])]
    end

    # plot the raw data and one separating hyperplane
    y = MNIST(split = :train).targets
    X = MNIST(split = :train).features
    idx0 = range(1, length(y))[y.==class0]
    idx1 = range(1, length(y))[y.==class1]
    X0 = hcat(map(i -> average_intensity_features(X[:, :, idx0[i]]), 1:n)...)'
    X1 = hcat(map(i -> average_intensity_features(X[:, :, idx1[i]]), 1:n)...)'
    p = plot(
        X0[:, 1],
        X0[:, 2],
        legend = false,
        seriestype = :scatter,
        color = class0_color,
        alpha = 0.5,
    )
    plot!(X1[:, 1], X1[:, 2], seriestype = :scatter, color = class1_color, alpha = 0.5)

    if (compute_FDA)
        m0 = mean(X0, dims = 1)'
        m1 = mean(X1, dims = 1)'
        S = zeros(size(X0, 2), size(X0, 2))
        for x in eachrow(X0)
            S += (x - m0) * (x - m0)'
        end
        for x in eachrow(X1)
            S += (x - m1) * (x - m1)'
        end

        w = inv(S) * (m1 - m0)
        w = [w[1]; w[2]; -w' * (1 / 2 * m0 + 1 / 2 * m1)]
        println(w)
    end

    # augment the data with a one column for the bias
    X0 = hcat(X0, ones(size(X0, 1), 1))
    X1 = hcat(X1, ones(size(X1, 1), 1))

    x_min = -(0.35 * w[2] + w[3]) / w[1]
    x_max = -w[3] / w[1]
    xs = range(start = x_min, stop = x_max, length = 100)
    plot!(xs, x -> -(x * w[1] + w[3]) / w[2], linewidth = 2, color = :black)
    display(p)

    # plot the distribution of the projections
    f0, f1 = X0 * w, X1 * w
    P0, P1 = Normal(mean(f0), sqrt(var(f0))), Normal(mean(f1), sqrt(var(f1)))
    mn, mx = min(f0..., f1...), max(f0..., f1...)
    xs, bins = range(mn, mx, length = 100), range(mn, mx, length = 30)

    p = histogram(
        f0,
        label = L"f_0",
        normalize = :pdf,
        bins = bins,
        color = class0_color,
        alpha = 0.5,
    )
    histogram!(
        f1,
        label = L"f_1",
        normalize = :pdf,
        bins = bins,
        color = class1_color,
        alpha = 0.5,
    )
    plot!(xs, x -> pdf(P0, x), label = L"N(f_0)", lw = 3, color = class0_color)
    plot!(xs, x -> pdf(P1, x), label = L"N(f_1)", lw = 3, color = class1_color)
    display(p)
end

plot_MNIST(w = [1; 1; -0.25])
plot_MNIST(w = [1; -1; 0.05])
plot_MNIST(compute_FDA = true)
