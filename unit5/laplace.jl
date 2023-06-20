# Plots for Laplace approximation
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using LaTeXStrings
using Plots


function plot_laplace(;
    a = 20,
    b = 4,
    xmin = -2,
    xmax = 4,
    color1 = :red,
    color2 = :blue,
    base = "~/Downloads/laplace",
)

    # plots a pair of functions
    function plot_functions(f1, f2; shaded = false, y_label = L"p(w|D)")
        xs = range(start = xmin, stop = xmax, length = 1000)
        # plot the frame
        p = plot(
            legend = false,
            xtickfontsize = 14,
            ytickfontsize = 14,
            xguidefontsize = 16,
            yguidefontsize = 16,
        )
        if (shaded)
            # plot the two functions
            plot!(xs, f1, color = color1, linewidth = 3)
            plot!(xs, f2, color = color2, linewidth = 3)

            # fill the ranges
            plot!(xs, map(x -> 0, xs), fillrange = f1, color = color1, fillalpha = 0.5)
            plot!(xs, map(x -> 0, xs), fillrange = f2, color = color2, fillalpha = 0.5)
        else
            # plot the two functions
            plot!(xs, f1, color = color1, linewidth = 3)
            plot!(xs, f2, color = color2, linewidth = 3)
        end

        xlabel!(L"w")
        ylabel!(y_label)
        display(p)
    end

    # sigmoid function
    sigmoid(t) = exp(t) / (1 + exp(t))

    # original negative log density to be approximated
    f(x) = -x^2 / 2 + log(sigmoid(a * x + b))
    f′(x) = -x + (1 - sigmoid(a * x + b)) * a
    f′′(x) = -1 - a^2 * sigmoid(a * x + b) * (1 - sigmoid(a * x + b))

    xxs = range(start = -10, stop = 10, length = 10000)
    log_Z = log(sum(map(x -> (xxs[2] - xxs[1]) * exp(f(x)), xxs)))

    # run Newton-Raphson to find the maximizer
    w0 = -b / a
    while (abs(f′(w0)) > 1e-4)
        println("w0 = ", w0, ", f(w0) = ", f(w0), ", f'(w0) = ", f′(w0))
        w0 = w0 - f′(w0) / f′′(w0)
    end

    μ = w0
    σ2 = 1 / -f′′(w0)
    println(σ2)

    plot_functions(x -> (x - μ)^2 / (2 * σ2), x -> -f(x), y_label = L"-\log(p(w|D))")
    savefig(base * "_neg_log_density.svg")
    plot_functions(
        x -> 1 / sqrt(2 * π * σ2) * exp(-(x - μ)^2 / (2 * σ2)),
        x -> exp(-log_Z + f(x)),
        shaded = true,
        y_label = L"p(w|D)",
    )
    savefig(base * "_density.svg")
end

plot_laplace(a = 20.0, b = 4.0, base = "~/Downloads/laplace_20_4")
