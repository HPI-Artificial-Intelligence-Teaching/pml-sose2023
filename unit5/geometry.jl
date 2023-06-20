# Plots for geometry of linear classification
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using LaTeXStrings
using Plots

"""
    plot_sigmoid()

Plots the sigmoid function
"""
function plot_sigmoid(; λ = 1, show_λ = false, filename = "~/Downloads/sigmoid.svg")
    # plot the sigmoid function
    xs = range(start = -4, stop = 4, length = 500)
    p = plot(
        legend = false,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
    )
    plot!(xs, map(x -> exp(λ * x) / (1 + exp(λ * x)), xs), linewidth = 3)
    ylims!((0, 1))
    if (show_λ)
        λ = round(λ, digits = 1)
        ylabel!(L"g(%$λ x)")
        xlabel!(L"x")
        display(p)
    else
        ylabel!(L"g(x)")
        xlabel!(L"x")
        display(p)
        savefig(filename)
    end
end

"""
    plot_sigmoid_video(;λ_start=-5, λ_end=5, filename="~/Downloads/anim_sigmoid.gif")

Plots the sigmoid function with a video of how it grows (from 2^`λ_start` to 2^`λ_end`)
"""
function plot_sigmoid_video(;
    λ_start = -1,
    λ_end = 6,
    filename = "~/Downloads/anim_sigmoid.gif",
)
    gr()
    anim = @animate for λ ∈ range(λ_start, λ_end, length = 200)
        plot_sigmoid(λ = 2^λ, show_λ = true)
    end
    gif(anim, filename, fps = 10)

end

"""
    plot_geometry()

Plots helper graphs for explaining the geometry
"""
function plot_geometry()
    xs = range(start = -1, stop = 2, length = 100)
    w = [1, 1]
    w0 = -1
    # plot the function
    p = plot(
        xs,
        x -> -x * w[1] / w[2] - w0 / w[2],
        color = :blue,
        linewidth = 2,
        legend = false,
        framestyle = :zerolines,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        aspect_ratio = :equal,
    )
    annotate!([2], [-2 * w[1] / w[2] - w0 / w[2]], text(L"S(\mathbf{w},w_0)", 14))

    x_a = [0.5, -0.5 * w[1] / w[2] - w0 / w[2]]
    x_b = [1.0, -1.0 * w[1] / w[2] - w0 / w[2]]
    plot!([x_a[1], x_b[1]], [x_a[2], x_b[2]], color = :red, arrow = :closed, linewidth = 2)
    scatter!([x_a[1], x_b[1]], [x_a[2], x_b[2]], color = :red)
    annotate!([x_a[1] - 0.1], [x_a[2] - 0.1], text(L"\mathbf{x}_A", 14))
    annotate!([x_b[1] - 0.1], [x_b[2] - 0.1], text(L"\mathbf{x}_B", 14))
    # annotate!([(x_a[1]+x_b[1])/2-.1], [(x_a[2]+x_b[2])/2-.1], text(L"\mathbf{x}_B-\mathbf{x}_A", 10, rotation = -35))
    plot!(
        [x_a[1], x_a[1] + 0.5 * w[1]],
        [x_a[2], x_a[2] + 0.5 * w[2]],
        color = :red,
        arrow = :closed,
        linewidth = 2,
    )
    annotate!(
        [x_a[1] + 0.25 * w[1] + 0.1],
        [x_a[2] + 0.25 * w[2] - 0.1],
        text(L"\mathbf{w}", 14),
    )
    ylims!((-1, 2))
    xlims!((-1, 2))
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)

    xs = range(start = -1, stop = 2, length = 100)
    w = [1, 1]
    w0 = -1
    # plot the function
    p = plot(
        xs,
        x -> -x * w[1] / w[2] - w0 / w[2],
        color = :blue,
        linewidth = 2,
        legend = false,
        framestyle = :zerolines,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        aspect_ratio = :equal,
    )
    annotate!([2], [-2 * w[1] / w[2] - w0 / w[2]], text(L"S(\mathbf{w},w_0)", 14))

    α = -w0 / (w' * w)
    x = [α * w[1], α * w[2]]
    plot!(
        [0, 3 * x[1]],
        [0, 3 * x[2]],
        color = :black,
        arrow = :closed,
        linestyle = :dot,
        linewidth = 1,
    )
    plot!([0, x[1]], [0, x[2]], color = :red, arrow = :closed, linewidth = 2)
    scatter!([x[1]], [x[2]], color = :red)
    annotate!([x[1]], [x[2] + 0.1], text(L"\mathbf{x}", 14))
    annotate!([3 * x[1] + 0.1], [3 * x[2] + 0.1], text(L"\mathbf{w}", 14))
    ylims!((-1, 2))
    xlims!((-1, 2))
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)

    xs = range(start = -1, stop = 2, length = 100)
    w = [1, 1]
    x = [1.5, 0.5]
    w0 = -1
    # plot the function
    p = plot(
        xs,
        x -> -x * w[1] / w[2] - w0 / w[2],
        color = :blue,
        linewidth = 2,
        legend = false,
        framestyle = :zerolines,
        xtickfontsize = 14,
        ytickfontsize = 14,
        xguidefontsize = 16,
        yguidefontsize = 16,
        aspect_ratio = :equal,
    )
    annotate!([2], [-2 * w[1] / w[2] - w0 / w[2]], text(L"S(\mathbf{w},w_0)", 14))

    α = -w0 / (w' * w)
    r = ((w' * x) + w0) / (w' * w)
    x1 = α * w
    x2 = r * w
    x⊥ = x - (x1 + x2)
    plot!(
        [0, 3 * x1[1]],
        [0, 3 * x1[2]],
        color = :black,
        arrow = :closed,
        linestyle = :dot,
        linewidth = 1,
    )
    plot!([0, x1[1]], [0, x1[2]], color = :red, arrow = :closed, linewidth = 2)
    plot!(
        [x1[1], x1[1] + x2[1]],
        [x1[2], x1[2] + x2[2]],
        color = :red,
        arrow = :closed,
        linewidth = 2,
    )
    plot!(
        [x1[1] + x2[1], x[1]],
        [x1[2] + x2[2], x[2]],
        color = :green,
        arrow = :closed,
        linewidth = 2,
    )
    scatter!([x[1]], [x[2]], color = :purple)
    plot!([0, x[1]], [0, x[2]], color = :purple, arrow = :closed, linewidth = 2)
    annotate!([x[1] + 0.15], [x[2]], text(L"\mathbf{x}", 14))
    annotate!([x1[1] / 2], [x1[2] / 2 + 0.17], text(L"\mathbf{x}_1", 14))
    annotate!([x1[1] + x2[1] / 2], [x1[2] + x2[2] / 2 + 0.17], text(L"\mathbf{x}_2", 14))
    annotate!(
        [x1[1] + x2[1] + x⊥[1] * 3 / 4],
        [x1[2] + x2[2] + x⊥[2] / 2 + 0.1],
        text(L"\mathbf{x}_\bot", 14),
    )
    annotate!([3 * x1[1] + 0.1], [3 * x1[2] + 0.1], text(L"\mathbf{w}", 14))
    ylims!((-1, 2))
    xlims!((-1, 2))
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    display(p)
end

plot_sigmoid()
plot_sigmoid_video()
plot_geometry()
