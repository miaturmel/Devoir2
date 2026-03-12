# ---
# title: Titre du travail
# repository: tpoisot/BIO245-modele
# auteurs:
#    - nom: Auteur
#      prenom: Premier
#      matricule: XXXXXXXX
#      github: premierAuteur
#    - nom: Auteur
#      prenom: Deuxième
#      matricule: XXXXXXXX
#      github: DeuxiAut
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


# ## Code 

#FONCTIONS

# Corrige la marice de transition, afin qu'elle suive le modèle de Markov ( programmation défensive )
# Simulation Non paramétrique puisque l'historique des états n'intervient pas (les transitions précédentes n'affecte pas les prochaines)
# La fonction "check_transition_matrice"  permet de normaliser la matrice, afin que la somme des probabilité de chaques transition possibles pour un état 
# initial donné sois de 100% ( chaques lignes = 1)
function check_transition_matrix!(T)
    for ligne in axes(T, 1) # Pour tout les lignes de la matrice T
        if sum(T[ligne, :]) != 1 # Si la somme d'une ligne n'est pas égale à 1
            @warn "La somme de la ligne $(ligne) n'est pas égale à 1 et a été modifiée"
            T[ligne, :] ./= sum(T[ligne, :]) # On divise chaques valeurs dans ligne par la somme de la ligne
        end
    end
    return T # Retourne la matrice corrigée au besoin
end

#  La fonction "Check_functions_arguments" vérifie  que l'ensemble des états initiaux et transitions possibles sont inclues ( Programmation défensive )
# Ne corrige pas automatiquement, mais envoie un message d'avertissement qui signal une exception
function check_function_arguments(transitions, states) 
    if size(transitions, 1) != size(transitions, 2) # Si le nombre de ligne n'est pas égal au nombre de colone, renvoie un message d'avertissement 
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
