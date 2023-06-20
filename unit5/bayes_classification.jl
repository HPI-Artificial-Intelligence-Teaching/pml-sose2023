# Plots for Bayesian regression
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

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
    plot_bayesian_inference(X, y; τ=sqrt(1/2), no_samples=200, base_name="fig")

Generates the plots of data, likelihood, posterior and `no_samples` function samples for a given 2D dataset and stores them in a folder with the base name `base_name`
"""
function plot_bayesian_inference(
    X,
    y;
    σ = 0.2,
    τ = sqrt(1 / 2),
    no_samples = 200,
    base_name = "fig",
)
    # Computes the logistic likelihood of the 2D linear function for the dataset given by `X` and `y` 
    function likelihood(w)
        t = X * w
        z = exp.(t .* 10) ./ (1 .+ exp.(t .* 10))
        return (prod(y .* z .+ (1 .- y) .* (1 .- z)))
    end

    # compute the Bayesian prior
    Σ = τ^2 * Diagonal(ones(2))
    μ = zeros(2)

    # plot the posterior over the weight space
    w1 = range(-3, stop = 3, length = 100)
    w2 = range(-3, stop = 3, length = 100)
    w1_coarse = range(-3, stop = 3, length = 30)
    w2_coarse = range(-3, stop = 3, length = 30)
    p = surface(
        w1,
        w2,
        (w1, w2) -> pdf(MvNormal(μ, Σ), [w1; w2])[1],
        color = :blues,
        fillalpha = 0.1,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        ztickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    wireframe!(w1_coarse, w2_coarse, (w1, w2) -> pdf(MvNormal(μ, Σ), [w1; w2])[1])
    xlabel!(L"w_1")
    ylabel!(L"w_2")
    display(p)
    savefig(p, base_name * "_prior.png")

    # plot the likelihood over the weight space
    w1 = range(-3, stop = 3, length = 100)
    w2 = range(-3, stop = 3, length = 100)
    w1_coarse = range(-3, stop = 3, length = 30)
    w2_coarse = range(-3, stop = 3, length = 30)
    p = surface(
        w1,
        w2,
        (w1, w2) -> likelihood([w1; w2]),
        color = :reds,
        fillalpha = 0.1,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        ztickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    wireframe!(w1_coarse, w2_coarse, (w1, w2) -> likelihood([w1; w2]))
    xlabel!(L"w_1")
    ylabel!(L"w_2")
    display(p)
    savefig(p, base_name * "_likel.png")

    # plot the posterior over the weight space
    w1 = range(-3, stop = 3, length = 100)
    w2 = range(-3, stop = 3, length = 100)
    w1_coarse = range(-3, stop = 3, length = 30)
    w2_coarse = range(-3, stop = 3, length = 30)
    p = surface(
        w1,
        w2,
        (w1, w2) -> pdf(MvNormal(μ, Σ), [w1; w2])[1] * likelihood([w1; w2]),
        color = :greens,
        fillalpha = 0.1,
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        ztickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    wireframe!(
        w1_coarse,
        w2_coarse,
        (w1, w2) -> pdf(MvNormal(μ, Σ), [w1; w2])[1] * likelihood([w1; w2]),
    )
    xlabel!(L"w_1")
    ylabel!(L"w_2")
    display(p)
    savefig(p, base_name * "_post.png")

    # plot the input space
    idx0 = range(1, length(y))[y.==0]
    idx1 = range(1, length(y))[y.==1]
    p = plot(
        X[idx0, 1],
        X[idx0, 2],
        legend = false,
        seriestype = :scatter,
        color = :red,
        alpha = 0.5,
        aspect_ratio = :equal,
    )
    plot!(X[idx1, 1], X[idx1, 2], seriestype = :scatter, color = :blue, alpha = 0.5)
    xlabel!(L"x")
    ylabel!(L"y")
    # xlims!(-1, 1)
    # ylims!(-1, 1)
    display(p)
    savefig(p, base_name * "_data.png")
end

X, y = extract_data(20)

plot_bayesian_inference(
    X[[1, 2, 21, 22], :],
    y[[1, 2, 21, 22]],
    base_name = "~/Downloads/bayes2",
)
plot_bayesian_inference(
    X[[1, 2, 3, 4, 21, 22, 23, 24], :],
    y[[1, 2, 3, 4, 21, 22, 23, 24]],
    base_name = "~/Downloads/bayes4",
)
plot_bayesian_inference(
    X[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], :],
    y[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]],
    base_name = "~/Downloads/bayes10",
)
