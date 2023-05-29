# Plots for polynomial regression
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
function generate_data(n::Int64, f; σ = 0.1, from = 0, to = 1)
    xs = collect(range(from, to, n))
    ys = map(f, xs) + randn((n, 1)) * σ
    return (x = xs, y = ys)
end

"""
    generate_features(data)

Generates the ferature matrix from the raw data
"""
function generate_features(data; max_degree = 2, min_degree = 1)
    return hcat([data .^ i for i = min_degree:max_degree]...)
end

"""
    fit_polynomial(train_data; degree=2, min_degree=1, color=:blue)

Generates a least-square fit of a polynomial of degree and plots the fit
"""
function fit_polynomial(
    train_data;
    degree = 2,
    min_degree = 1,
    color = :blue,
    threed = false,
)
    train_X = generate_features(train_data.x, min_degree = min_degree, max_degree = degree)
    w = inv(train_X' * train_X) * train_X' * train_data.y
    xs = collect(range(minimum(train_data.x), maximum(train_data.x), 100))
    test_X = generate_features(xs, min_degree = min_degree, max_degree = degree)
    ys = test_X * w
    if (threed)
        plot!(xs, ys, zeros(100), linewidth = 3, color = color)
    else
        plot!(xs, ys, linewidth = 3, color = color)
    end
    return w
end

function likelihood(w, X, y; σ = 0.15)
    return pdf(MvNormal(X * w, I * σ^2), y)
end

# generates training data 
Random.seed!(41)
# train_data = generate_data(5, x -> sin(x*π), σ=0.15, from=0, to=1)
train_data = generate_data(11, x -> sin(x * π), σ = 0.15, from = 0, to = 1)
# train_data = generate_data(11, x -> 1/(1+x^2), σ=0.05, from=-5, to=5)

# plot of the training data
p = plot(
    train_data.x,
    train_data.y,
    seriestype = :scatter,
    legend = false,
    color = :orange,
    xtickfontsize = 14,
    ytickfontsize = 14,
)
xlabel!(L"x")
ylabel!(L"y")
display(p)

# plot of the training data with best fit of 2nd order polynomial
p = plot(
    train_data.x,
    train_data.y,
    seriestype = :scatter,
    legend = false,
    color = :orange,
    xtickfontsize = 14,
    ytickfontsize = 14,
)
xlabel!(L"x")
ylabel!(L"y")
w_best = fit_polynomial(train_data, min_degree = 1, degree = 2, color = :blue)
display(p)

# plot of the training data with best fit of 6th order polynomial
p = plot(
    train_data.x,
    train_data.y,
    seriestype = :scatter,
    legend = false,
    color = :orange,
    xtickfontsize = 14,
    ytickfontsize = 14,
)
xlabel!(L"x")
ylabel!(L"y")
fit_polynomial(train_data, min_degree = 0, degree = 6, color = :blue)
display(p)

# plot of the training data with best fit of 10th order polynomial
p = plot(
    train_data.x,
    train_data.y,
    seriestype = :scatter,
    legend = false,
    color = :orange,
    xtickfontsize = 14,
    ytickfontsize = 14,
    ztickfontsize = 14,
)
xlabel!(L"x")
ylabel!(L"y")
fit_polynomial(train_data, min_degree = 0, degree = 10, color = :blue)
display(p)

# plot the prior over the weight space
mv = MvNormal([0, 0], I * 2)
x = y = range(-5, stop = 5, length = 100)
x_coarse = y_coarse = range(-5, stop = 5, length = 40)
p = surface(
    x,
    y,
    (x, y) -> pdf(mv, [x, y]),
    color = :blues,
    fillalpha = 0.1,
    legend = false,
    xtickfontsize = 14,
    ytickfontsize = 14,
    ztickfontsize = 14,
)
wireframe!(x_coarse, y_coarse, (x, y) -> pdf(mv, [x, y]))
xlabel!(L"w_1")
ylabel!(L"w_2")
display(p)

# plot the likelihood over the weight space
train_X = generate_features(train_data.x, min_degree = 1, max_degree = 2)
w1 = range(4, stop = 5, length = 100)
w2 = range(-5, stop = -4, length = 100)
w1_coarse = range(3, stop = 5, length = 30)
w2_coarse = range(-5, stop = -3, length = 30)
p = surface(
    w1,
    w2,
    (w1, w2) -> likelihood([w1, w2], train_X, train_data.y)[1],
    color = :reds,
    fillalpha = 0.1,
    legend = false,
    xtickfontsize = 14,
    ytickfontsize = 14,
    ztickfontsize = 14,
)
wireframe!(w1_coarse, w2_coarse, (w1, w2) -> likelihood([w1, w2], train_X, train_data.y)[1])
xlabel!(L"w_1")
ylabel!(L"w_2")
display(p)

# plot the posterior over the weight space
train_X = generate_features(train_data.x, min_degree = 1, max_degree = 2)
w1 = range(4, stop = 5, length = 100)
w2 = range(-5, stop = -4, length = 100)
w1_coarse = range(3, stop = 5, length = 30)
w2_coarse = range(-5, stop = -3, length = 30)
p = surface(
    w1,
    w2,
    (w1, w2) -> likelihood([w1, w2], train_X, train_data.y)[1] * pdf(mv, [w1, w2]),
    color = :greens,
    fillalpha = 0.1,
    legend = false,
    xtickfontsize = 14,
    ytickfontsize = 14,
    ztickfontsize = 14,
)
wireframe!(
    w1_coarse,
    w2_coarse,
    (w1, w2) -> likelihood([w1, w2], train_X, train_data.y)[1] * pdf(mv, [w1, w2]),
)
xlabel!(L"w_1")
ylabel!(L"w_2")
display(p)

# plot the predictive distribution
x = range(0, stop = maximum(train_data.x), length = 100)
y = range(minimum(train_data.y), stop = maximum(train_data.y), length = 100)
x_coarse = range(minimum(train_data.x), stop = maximum(train_data.x), length = 30)
y_coarse =
    range(1.5 * minimum(train_data.y), stop = 1.5 * maximum(train_data.y), length = 30)
p = plot(train_data.x, train_data.y, zeros(11), seriestype = :scatter, color = :orange)
fit_polynomial(train_data, min_degree = 1, degree = 2, color = :blue, threed = true)
# surface!(x, y, (x, y) -> pdf(Normal(w_best[1]*x+w_best[2]*x*x,0.15),y), color=:greys, fillalpha=0.1, legend=false, xtickfontsize=14, ytickfontsize=14, ztickfontsize=14)
surface!(
    x,
    y,
    (x, y) -> pdf(Normal(w_best[1] * x + w_best[2] * x * x, 0.15), y),
    color = :greys,
    fillalpha = 0.1,
    legend = false,
)
wireframe!(
    x_coarse,
    y_coarse,
    (x, y) -> pdf(Normal(w_best[1] * x + w_best[2] * x * x, 0.15), y),
)
xlabel!(L"x")
ylabel!(L"y")
display(p)
