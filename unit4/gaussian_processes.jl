# Plots for Gaussian Processes
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Random
using LinearAlgebra
using Distributions
using LaTeXStrings
using Plots

"""
    RBF_kernel(x1,x2;λ=1)

Computes the radial basis function kernel value at `x1` and `x2` with length scale `λ`

# Examples
```jldoctest
julia> RBF_kernel(0, 1; λ = 1)
0.36787944117144233

julia> RBF_kernel(0, 1; λ = 2)
0.7788007830714049
```
"""
function RBF_kernel(x1, x2; λ = 1)
    return exp(-(x1 - x2)^2 / λ^2)
end

"""
    OU_kernel(x1,x2;λ=1)

Computes the Ornstein-Uhlenbeck kernel value at `x1` and `x2` with length scale `1/λ`
# Examples
```jldoctest
julia> OU_kernel(0, 1; λ = 1)
0.36787944117144233

julia> OU_kernel(0, 1; λ = 2)
0.6065306597126334
```
"""
function OU_kernel(x1, x2; λ = 1)
    return exp(-abs(x1 - x2) / λ)
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
function generate_data(n::Int64, f; σ = 0.1, from = 0, to = 1)
    xs = collect(range(from, to, n))
    ys = map(f, xs) + randn((n, 1)) * σ
    return (x = xs, y = ys)
end

"""
    plot_GP_prior(kernel; no_samples=300, alpha=0.1, color=:red, xlim=(-1,1))

Plots a Gaussian Process prior with a given kernel using `no_samples` many functions with an alpha channel of `alpha` and color `color`
"""
function plot_GP_prior(
    kernel;
    no_samples = 300,
    highlight_idx = 300,
    alpha = 0.1,
    color = :red,
    xlim = (-1, 1),
    file_name = "~/Downloads/GP.svg",
)
    # plot the empty canvas
    p = plot(
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"y")

    # compute the parameters of the GP instanciations at the sample points
    n = 300
    xs = range(xlim[1], stop = xlim[2], length = n)
    μ = zeros(n)
    Σ = [kernel(x1, x2) for x1 in xs, x2 in xs]

    # approximate the √Σ
    U, S, V = svd(Σ)
    Σ_half = U * Diagonal(sqrt.(S)) * V'

    # sample all the standard Gaussians at once and then transform them for each sample
    yss = rand(MvNormal(μ, I(n)), no_samples)
    for j = 1:no_samples
        width, α = if (j == highlight_idx)
            3, 1
        else
            1, alpha
        end
        ys = Σ_half * yss[:, j]
        plot!(xs, ys, linewidth = width, alpha = α, color = :red)
    end
    display(p)
    savefig(p, file_name)
end

"""
    plot_GP_fit(train_data; σ=0.1, kernel=RBF_kernel, color=:blue)

Generates a plot of the GP posterior for the training data `train_data` using the kernel `kernel`. For the data noise, a standard deviation of `σ`. Using `color` for the line and ribbon.
"""
function plot_GP_fit(train_data; σ = 0.1, kernel = RBF_kernel, color = :blue)
    function kern(x1, x2)
        if (x1 == x2)
            return kernel(x1, x2) + σ^2
        else
            return kernel(x1, x2)
        end
    end
    
    # compute the parameters of the GP posterior 
    C = [kern(x1, x2) for x1 in train_data.x, x2 in train_data.x]
    C_inv = inv(C)
    w = C_inv * train_data.y

    # compute the mean and standard deviation of the predictive distribution over all test points 
    xs = collect(range(-0.3, 1.3, 100))
    K = map(x -> map(xt -> kern(x, xt), train_data.x), xs)
    pred = map((k, x) -> (w' * vec(k), kern(x, x) - vec(k)' * C_inv * vec(k)), K, xs)
    plot!(
        xs,
        map(x -> x[1][1], pred),
        ribbon = map(x -> x[2], pred),
        fillalpha = 0.2,
        linewidth = 3,
        color = color,
    )
end

"""
    plot_GP_evidence_maximization(train_data; base_name="~/Downloads/")

Generates a plot of the GP log2-evidence for the training data `train_data`.
"""
function plot_GP_evidence_maximization(train_data; base_name = "~/Downloads/")
    # compute the log evidence for a given set of noise parameters `σ` and RBF kernel length scale `λ`
    function log2_evidence(σ, λ)
        kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = λ)

        function kern(x1, x2)
            if (x1 == x2)
                return kernel(x1, x2) + σ^2
            else
                return kernel(x1, x2)
            end
        end
        C = [kern(x1, x2) for x1 in train_data.x, x2 in train_data.x]

        return (
            (
                -1 / 2 * log(det(C)) - (1/2*train_data.y'*inv(C)*train_data.y)[1] -
                size(C, 1) / 2 * log(2π)
            ) / log(2)
        )
    end

    # plot the evidence
    σs = range(start = 0.05, stop = 2, length = 100)
    λs = range(start = 0.01, stop = 1, length = 100)
    σs_coarse = range(start = 0.05, stop = 2, length = 30)
    λs_coarse = range(start = 0.01, stop = 1, length = 30)
    p = surface(
        σs,
        λs,
        log2_evidence,
        color = :purple,
        fillalpha = 0.5,
        camera = (45, 30),
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        ztickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        zguidefontsize = 16,
    )
    wireframe!(σs_coarse, λs_coarse, log2_evidence)
    xlabel!(L"\sigma")
    ylabel!(L"\lambda")
    zlabel!(L"\log_2(P(D|\sigma,\lambda))")

    # maximize the evidence by brute-force search
    p2 = plot(
        train_data.x,
        train_data.y,
        seriestype = :scatter,
        legend = false,
        color = :orange,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    xlabel!(L"x")
    ylabel!(L"y")
    max_cnt = 1
    for i in eachindex(σs)
        for j in eachindex(λs)
            logE = log2_evidence(σs[i], λs[j])
            if (
                (i == 1 || log2_evidence(σs[i-1], λs[j]) <= logE) &&
                (i == length(σs) || log2_evidence(σs[i+1], λs[j]) <= logE) &&
                (j == 1 || log2_evidence(σs[i], λs[j-1]) <= logE) &&
                (j == length(λs) || log2_evidence(σs[i], λs[j+1]) <= logE)
            )
                println(σs[i], ",", λs[j])
                plot_GP_fit(
                    train_data,
                    σ = σs[i],
                    kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = λs[j]),
                    color = max_cnt,
                )
                max_cnt += 1
            end
        end
    end
    display(p)
    display(p2)
    savefig(p2, base_name * "GP_evidence_max.svg")
    savefig(p, base_name * "GP_evidence.png")
end

# plot the GP prior
Random.seed!(42)
plot_GP_prior(
    (x1, x2) -> RBF_kernel(x1, x2, λ = 0.000005),
    color = :red,
    file_name = "~/Downloads/GP_prior_RBF.svg",
)plot_GP_prior(
    (x1, x2) -> OU_kernel(x1, x2, λ = 20000),
    color = :red,
    file_name = "~/Downloads/GP_prior_OU.svg",
)

# plot the GP posterior
Random.seed!(41)
train_data = generate_data(101, x -> sin(x * π), σ = 0.55, from = 0, to = 1)

p = plot(
    train_data.x,
    train_data.y,
    seriestype = :scatter,
    legend = false,
    color = :orange,
    xtickfontsize = 14,
    ytickfontsize = 14,
    xguidefontsize = 16,
    yguidefontsize = 16,
)
xlabel!(L"x")
ylabel!(L"y")
plot_GP_fit(
    train_data,
    σ = 0.25,
    kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 1),
    color = :blue,
)
plot_GP_fit(
    train_data,
    σ = 0.25,
    kernel = (x1, x2) -> RBF_kernel(x1, x2, λ = 0.5),
    color = :red,
)
display(p)
savefig(p, "~/Downloads/GP_RBF_fit.svg")

p = plot(
    train_data.x,
    train_data.y,
    seriestype = :scatter,
    legend = false,
    color = :orange,
    xtickfontsize = 14,
    ytickfontsize = 14,
    xguidefontsize = 16,
    yguidefontsize = 16,
)
xlabel!(L"x")
ylabel!(L"y")
plot_GP_fit(
    train_data,
    σ = 0.25,
    kernel = (x1, x2) -> OU_kernel(x1, x2, λ = 2),
    color = :blue,
)
plot_GP_fit(
    train_data,
    σ = 0.25,
    kernel = (x1, x2) -> OU_kernel(x1, x2, λ = 0.1),
    color = :red,
)
display(p)
savefig(p, "~/Downloads/GP_OU_fit.svg")

# plot the GP evidence maximization
Random.seed!(41)
train_data = generate_data(11, x -> sin(x * π), σ = 0.15, from = 0, to = 1)
plot_GP_evidence_maximization(train_data; base_name = "~/Downloads/")

