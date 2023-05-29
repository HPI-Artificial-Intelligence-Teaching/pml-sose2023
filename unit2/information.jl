# Plots for the information theory section
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

using Plots

# computes the binary entropy
function binary_entropy(p = 0.5)
    if (p == 0)
        return 0
    elseif (p == 1)
        return 0
    end

    return -p * log(2, p) - (1 - p) * log(2, 1 - p)
end

# reads a file and returns the entropy of the ascii code distribution
function average_ascii_frequency(file_name)
    letter_count = Dict{Char,Int}()
    open(file_name, "r") do file
        for line in eachline(file)
            for c in line
                # c = Char(rand(UInt8))
                if (haskey(letter_count, c))
                    letter_count[c] += 1
                else
                    letter_count[c] = 1
                end
            end
        end
    end

    return letter_count
end

# computes the entropy of a dictionary-based distribution
function entropy(d)
    n = sum(values(d))
    return sum(map(k -> -k / n * log(2, k / n), values(d)))
end

# plot the binary entropy
ps = 0:0.01:1.0
p = plot(ps, map(p -> binary_entropy(p), ps), legend = false, linewidth = 3)
# scatter!(πs,map(π -> var_bernoulli(π),πs))
ylabel!("H[p]")
xlabel!("p")
display(p)

# plots the entropy of letters in an actual file
d = average_ascii_frequency("bible.txt")
n = sum(values(d))
sorted_keys = sort(collect(keys(d)), by = key -> -d[key])
p = bar(sorted_keys[1:15], map(k -> d[k] / n, sorted_keys[1:15]), legend = false)
ylabel!("Frequency")
xlabel!("ASCII codes")
display(p)

println("H = ", entropy(d))
