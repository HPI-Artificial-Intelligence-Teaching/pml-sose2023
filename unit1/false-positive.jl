# False-Positive Puzzle
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Plots

"""
    posterior(α=0.95,β=0.001)

Returns the posterior for a positive outcome with error probability `α` and scarcity `β`
```
"""
function posterior(α = 0.95, β = 0.001)
    return α * β / (α * β + (1 - α) * (1 - β))
end

# plot the key distribution on the screen
αs = 0.5:0.01:1.0
p = plot(αs, map(α -> posterior(α), αs), legend = false, linewidth = 3, yaxis = :log)
scatter!(αs, map(α -> posterior(α), αs))
ylabel!("P(disease|positive test)")
xlabel!("test accuracy")
display(p)

βs = 0.001:0.01:0.5
p = plot(βs, map(β -> posterior(0.95, β), βs), legend = false, linewidth = 3, xaxis = :log)
scatter!(βs, map(β -> posterior(0.95, β), βs))
ylabel!("P(disease|positive test)")
xlabel!("scarcity")
display(p)
