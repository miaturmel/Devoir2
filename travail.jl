# ---
# title: Devoir 2
# repository: tpoisot/BIO245-modele
# auteurs:
#    - nom: Lafontaine
#      prenom: Laurianne
#      matricule: 20275756
#      github: LaurianneLafontaine
#    - nom: Turmel
#      prenom: Mia
#      matricule: 20277557
#      github: miaturmel
# ---

# # Introduction

# # Présentation du modèle

# # Implémentation

# ## Packages nécessaires

import Random
Random.seed!(123456)
using CairoMakie
using Distributions

# ## Documentation

"""
    check_transition_matrix!(arg1)

Vérifie que la somme des lignes d'une matrice données correspond à 1.
Si cette condition n'est pas respectée, la fonction modifie les valeurs des lignes en les normalisant.
Un avertissement est émis pour aviser de la modification.

# Arguments 
arg1 = une matrice de transition. Les lignes de la matrice représente des distributions de tansitions d'états.

# Retour
La fonction retourne la matrice de transition potentiellement normalisée pour que les sommes des valeurs de chaque ligne correspondent à 1.
"""

"""
    check_function_arguments(arg1, arg2)

Vérifie en premier lieu que la matrice donnée soit carrée.
Vérifie en deuxième lieu que tous les états soient représentées dans la matrice donnée.
Renvoie un avertissement si une de ces conditions (ou les deux) ne sont pas respectées.

# Arguments 
arg1 = une matrice de transition.
arg2 = la longueur d'un vecteur contenant les états possibles.

# Retour
La fonction ne retourne rien.
"""

"""
    _sim_stochastic!(arg1, arg2, arg3)

La fonction effectue une modification stochastique d'une matrice des états pour chaque génération.
Les modifications sont effectuées selon une distribution multinomiale prenant en compte les états à la génération
courante et les probabilités de transitions dans la matrice de transition.

# Arguments 
arg1 = une matrice représentant les états de la parcelle pour chaque génération.
arg2 = une matrice de transition.
arg3 = l'indice de la génération courante sur laquelle la fonction s'applique.

# Retour
La fonction ne retourne rien, elle modifie une directement une matrice.
"""

"""
    _sim_determ!(arg1, arg2, arg3)

La fonction effectue une modification déterministe d'une matrice des états pour chaque génération.
La fonction multiplie les états des parcelles à la génération courante par les probabilités de transition
contenues dans la matrice de transition, puis ajoute ces états de la prochaine génération à une matrice
d'états pour chaque génération.

# Arguments 
arg1 = une matrice représentant les états de la parcelle pour chaque génération.
arg2 = une matrice de transition.
arg3 = l'indice de la génération courante sur laquelle la fonction s'applique.

# Retour
La fonction ne retourne rien, elle modifie une directement une matrice.
"""

"""
    simulation(arg1, arg2; keyword1, keyword2)

La fonction effectue les fonctions de programmation défensives établies plus tôt.
Elle identifie ensuite le type de données sur lequel elle effectuera ses opérations et mes celles-ci 
dans une matrice représentant les états de la parcelle pour chaque génération. La fonction détermine finalement
si la simulation à réaliser doit être stochastique ou détermininiste et l'effectue pour chaque générations.

# Arguments 
arg1 = une matrice de transition.
arg2 = la longueur d'un vecteur contenant les états possibles.
keyword1 = un mot-clé représentant le nombre de génération sur lequel on effectue la simulation.
keyword2 = un mot-clé pour identifier la manière dont la simulation est réalisée; de façon stochastique ou déterministe.

# Retour
La fonction retourne une matrice représentant le nombre de parcelles dans chaque état pour chaque génération.
"""

## Code 
# Corrige la marice de transition, afin qu'elle suive le modèle de Markov 
# Non paramétrique puisque l'historique des états n'intervient pas ( les transitions précédentes n'affecte pas les prochaines)
function check_transition_matrix!(T)
    for ligne in axes(T, 1)
        if sum(T[ligne, :]) != 1
            @warn "La somme de la ligne $(ligne) n'est pas égale à 1 et a été modifiée"
            T[ligne, :] ./= sum(T[ligne, :])
        end
    end
    return T
end

function check_function_arguments(transitions, states)
    if size(transitions, 1) != size(transitions, 2)
        throw("La matrice de transition n'est pas carrée")
    end

    if size(transitions, 1) != length(states)
        throw("Le nombre d'états ne correspond psa à la matrice de transition")
    end
    return nothing
end

function _sim_stochastic!(timeseries, transitions, generation)
    for state in axes(timeseries, 1)
        pop_change = rand(Multinomial(timeseries[state, generation], transitions[state, :]))
        timeseries[:, generation+1] .+= pop_change
    end
end

function _sim_determ!(timeseries, transitions, generation)
    pop_change = (timeseries[:, generation]' * transitions)'
    timeseries[:, generation+1] .= pop_change
end

function simulation(transitions, states; generations=500, stochastic=false)

    check_transition_matrix!(transitions)
    check_function_arguments(transitions, states)

    _data_type = stochastic ? Int64 : Float32
    timeseries = zeros(_data_type, length(states), generations + 1)
    timeseries[:, 1] = states

    _sim_function! = stochastic ? _sim_stochastic! : _sim_determ!

    for generation in Base.OneTo(generations)
        _sim_function!(timeseries, transitions, generation)
    end

    return timeseries
end

# States
# Barren, Grass, Shrubs
s = [100, 0, 0]
states = length(s)
patches = sum(s)

# Transitions
T = zeros(Float64, states, states)
T[1, :] = [110, 8, 0]
T[2, :] = [2, 120, 3]
T[3, :] = [1, 0, 94]
T

states_names = ["Barren", "Grasses", "Shrubs"]
states_colors = [:grey40, :orange, :teal]

# Simulations

f = Figure()
ax = Axis(f[1, 1], xlabel="Nb. générations", ylabel="Nb. parcelles")

# Stochastic simulation
for _ in 1:100
    sto_sim = simulation(T, s; stochastic=true, generations=200)
    for i in eachindex(s)
        lines!(ax, sto_sim[i, :], color=states_colors[i], alpha=0.1)
    end
end

# Deterministic simulation
det_sim = simulation(T, s; stochastic=false, generations=200)
for i in eachindex(s)
    lines!(ax, det_sim[i, :], color=states_colors[i], alpha=1, label=states_names[i], linewidth=4)
end

axislegend(ax)
tightlimits!(ax)
current_figure()




# # Présentation des résultats

# La figure suivante représente des valeurs aléatoires:

hist(randn(100))

# # Discussion

# On peut aussi citer des références dans le document `references.bib`,
# @ermentrout1993cellular -- la bibliographie sera ajoutée automatiquement à la
# fin du document.
