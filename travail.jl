### alllooo

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


# stochasticite

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
    for ligne in axes(T, 1) # Pour tout les lignes de la matrice T
        if sum(T[ligne, :]) != 1 # Si la somme d'une ligne n'est pas égale à 1
            @warn "La somme de la ligne $(ligne) n'est pas égale à 1 et a été modifiée"
            T[ligne, :] ./= sum(T[ligne, :]) # On divise chaques valeurs dans ligne par la somme de la ligne
        end
    end
    return T # Retourne la matrice corrigée au besoin
end

print(T)

function check_function_arguments(transitions, states)
    if size(transitions, 1) != size(transitions, 2)
        throw("La matrice de transition n'est pas carrée")
    end

    if size(transitions, 1) != length(states) # Si le nombre de lignes de la matrice de transition n'est pas égal au nombres d'états possibles 
        throw("Le nombre d'états ne correspond pas à la matrice de transition") #  renvoie un message d'avertissement
    end 
    return nothing 
end

# SIMULATION STOCHASTIC DU MODEL DE SUCCESION VÉGÉTALE
# Timeseries est une matrice, les lignes sont les états et les colones sont les générations. Au début, seul la colone de la premiere 
#génération est définie, les autres générations sont vides ( les états pour les générations suivantes ne sont pas calculés ).

function _sim_stochastic!(timeseries, transitions, generation) # ! signifie que la fonction modifie l'objet (la matrice), et ne fais pas seulement une copie 
    for state in axes(timeseries, 1) # Pour chaques lignes dans la matrice timeseries

        pop_change = rand(Multinomial(timeseries[state, generation], transitions[state, :])) 

        # Applique les probabilités multinomiales de transition sur le nombre de parcelle de chaque état à la génération t ( courante ), afin de déterminer 
        #le nombre de parcelles pour chaques états à la génération t+1 ( suivante )
        # rand permet de sélectionner au hasard quelles parcelles vont subir quelles changement d'état (en respectant les  proabilité weighted de transition )

        timeseries[:, generation+1] .+= pop_change 

     # remplis la matrice timeseries avec les nouveaux états pour chaques générations en fonction du nouveau état déterminé par "pop_change" appliqué sur la
     # génération précédente

    end
end

# SIMULATION DÉTERMINISTE DU MODEL DE SUCCESSION VÉGÉTALE 
function _sim_determ!(timeseries, transitions, generation)

    pop_change = (timeseries[:, generation]' * transitions)'

    # Opération sur matrice, calcul le nombre de parcelles pour chaques nouveau état à la génération suivante (t+1) à  partir des probabilité de la matrice 
    # de transition appliqueée sur les parcelles de la génération actuelle (t)

    timeseries[:, generation+1] .= pop_change

    # remplis la matrice timeseries avec les nouveaux états pour chaques générations en fonction du nouveau état déterminé par "pop_change" appliqué sur la 
    #génération précédente

end

# SIMULATION FINALE
# Permet de comparer les effet du nombre de generation sur les deux simulations en même temps ainsi que les différences entre les 2 simulations
# de succession végétale ( stochastique et déterministe )
# Les arguments de la fonction contiennent des mots clé ( generations = , stochastic = ), permettant de visualiser facilement l'effet de la variation 
# des paramètres

function simulation(transitions, states; generations=200, stochastic=false) # peut indiquer directement le nb de generation souhaité et le type de simulation

    # Fonctions  " programmation défensive " , s'assurer que respecte model de Markov et que matrice est complète 
    check_transition_matrix!(transitions) 
    check_function_arguments(transitions, states)

     # Si les données sont stochastique, les données sont des nombres entiers arrondit à la baisse, si non ( déterministe ) ce sont des nombres à virgule
    _data_type = stochastic ? Int64 : Float32 
   
    # Créer la matrice vide qui va stocker l'information sur le nombre de parcelle de chaques, pour chaques générations ( lignes = états, colones = generations )
    timeseries = zeros(_data_type, length(states), generations + 1)

    timeseries[:, 1] = states # première dimension (lignes) correspondent aux états

    # Si l'aurgment dans les mots clé est "stochastique = true ", faire appelle à la fonction  _sim_stochastic!, sinon faire appelle a la fonction  _sim_determ!  
    _sim_function! = stochastic ? _sim_stochastic! : _sim_determ!  

    # Pour tout les génération suivant la génération initiale, appliquer la simulation  (stochaistique ou deterministe, celon l'argument)
    # Et retourner la matrice qui indique le nombre de parcelle pour chaques états, pour toutes les générations 
    for generation in Base.OneTo(generations)
        _sim_function!(timeseries, transitions, generation)
    end

    return timeseries
end

# PARAMETRES 

# Vecteur avec les états initiaux  : Vide, gazon, Rose, Lila 
initial_states = [200, 0, 0, 0]  # Commence avec 100 parcelles vides ( petit effectif )
# L'effet de stochasticité est grand, puisque la population est petite
states = length(initial_states) # nombres d'états possibles 

# Création de la matrice de transition
T = zeros(Float64, states, states)

# Ajouter une probabilité associé aux états posisble à la génération suivante,  pour chaques état initial

# Une parcelle vide a 13,75 x plus de chance de rester vide que devenir gazon, et aucune chance de devenir directement un buisson (impossible de grandir autant d'un coup )
T[1, :] = [110, 8, 0] 
# Une parcelle de gazon à 2,4 % de chance de devenir buisson, et 1,6 % de mourrir ( devenir vide )
T[2, :] = [2, 120, 3]
# Une parcelle de buisson a environ 1% de chance de mourrir ( devenir vide ), et ne peut pas devenir gazon 
T[3, :] = [1, 0, 94]
T
 
#PARAMETRE DU GRAPHIQUE

#Légende et couleurs associé
states_names = ["vide", "Gazon", "Arbuste"]
states_colors = [:grey40, :orange, :teal]

# Simulations sur le graphique
f = Figure()
ax = Axis(f[1, 1], xlabel="Nb. générations", ylabel="Nb. parcelles")


# Superpose les 2 types de simulation sur le graphique, pour l'analyse
# Stochastic simulation
for _ in 1:100 # répéter la simulation stochaistique 100 fois
    sto_sim = simulation(T, initial_states; stochastic=true, generations=200)
    for i in eachindex(initial_states)
        lines!(ax, sto_sim[i, :], color=states_colors[i], alpha=0.1)
    end
end

# Deterministic simulation
det_sim = simulation(T, initial_states; stochastic=false, generations=200)
for i in eachindex(initial_states)
    lines!(ax, det_sim[i, :], color=states_colors[i], alpha=1, label=states_names[i], linewidth=4)
end

axislegend(ax)
tightlimits!(ax)
current_figure()



# # Présentation des résultats

# La figure suivante représente des valeurs aléatoires:

#hist(randn(100))

# # Discussion

# On peut aussi citer des références dans le document `references.bib`,
# @ermentrout1993cellular -- la bibliographie sera ajoutée automatiquement à la
# fin du document.
