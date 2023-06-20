# Plots for optimization with the Newton-Raphson algorithm
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using LaTeXStrings
using Plots

function plot_newton_raphson(f, f′, f′′, x_ts)
    xs = range(start = -1.5, stop = 2, length = 100)
    # plot the function
    p = plot(
        xs,
        f,
        color = :blue,
        linewidth = 5,
        legend = false,
        framestyle = :zerolines,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    # plot the first derivative
    plot!(xs, f′, color = :red, linewidth = 3)

    for x_t in x_ts
        for i = 1:2
            # plot the line from x_t to f′(x_t)
            plot!(
                [x_t, x_t],
                [0, f′(x_t)],
                color = :black,
                linestyle = :dot,
                arrow = :closed,
                linewidth = 1,
            )
            # linear approximation at point x_t
            g(x) = f′′(x_t) * x + (f′(x_t) - f′′(x_t) * x_t)

            x1 = (20 - f′(x_t) + f′′(x_t) * x_t) / f′′(x_t)
            x2 = (-20 - f′(x_t) + f′′(x_t) * x_t) / f′′(x_t)
            xss = range(
                start = max(min(x1, x2), minimum(xs)),
                stop = min(max(x1, x2), maximum(xs)),
                length = 100,
            )

            # plot the linear approximation to the first derivative
            plot!(xss, g, color = :black, linestyle = :dash, linewidth = 1)

            # update to the next value of the 
            x_t_new = x_t - f′(x_t) / f′′(x_t)

            # plot the line from x_t to f′(x_t)
            plot!(
                [x_t, x_t_new],
                [-1, -1],
                color = :black,
                linestyle = :dot,
                arrow = :closed,
                linewidth = 1,
            )

            # update x_t
            x_t = x_t_new
        end
        # plot the line from x_t to f(x_t)
        plot!([x_t, x_t], [0, f(x_t)], color = :blue, arrow = :closed, linewidth = 1)
    end

    ylims!((-20, 20))
    xlabel!(L"x")
    ylabel!(L"y")
    display(p)
end

f(x) = 3 * x^3 - 2 * x^2 - 3 * x - 3
f′(x) = 9 * x^2 - 4 * x - 3
f′′(x) = 18 * x - 4
plot_newton_raphson(f, f′, f′′, [-1, 1.5])
