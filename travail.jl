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

# Le modèle présenté ci-dessous est un modèle "états-transitions". Ce genre de modèle décrit généralement 
# les états que des individus peuvent représenter, les transitions qui permettent à ces individus de changer 
# d'états et la probabilité que ces transitions se manifestent. Plusieurs facteurs peuvent être ajoutés à la
# modélisation de ces intéractions, tel que des intéractions entre les états, ce qui modifie la probabilité
# de transition d'un individu selon les états des voisins ou des transitions régulières sur un certain
# interval de temps [1].

# Le modèle présenté ci-dessous, tant qu'à lui, représente le modèle de Markov; un modèle qui ne prend pas
# en compte les intéractions entre les états, mais admet des transitions régulières sur un interval de temps 
# préalablement défini [1]. Le résultat de la simulation exécutée dans ce modèle devrait ainsi reproduire
# une chaine de Markov qui explique la transitions entre deux états, qui est capable de prédire un état 
# stationnaire et qui n'a pas de processus de mémoire (le modèle ne retient pas d'informations d'une transition
# à une autre). C'est ainsi pour conserver l'intégrité des chaines de Markov que le modèle est non-paramétrique,
# c'est-à-dire que la matrice d'états des parcelles et la matrice de transition modélisées séparémment. 

# De plus, le modèle peut être représenté de deux manières : stochastique ou déterministe.

# Pour le modèle stochastique, une augmentation de la taille de la population modélisée tend à diminuer 
# l'effet de la stochasticité. Au contraire, une diminution de la taille de la population amplifie l'effet 
# de la stochasticité. Ses points forts sont qu'il permet de visualiser l'effet de la stochasticité et qu'il
# prend en compte des nombres entiers, ce qui permet au modèle d'être plus réaliste, selon la situation. Selon 
# ce type de modèle, nous faisons plusieurs simulations qui n'ont pas toujours les mêmes résultats.

# Pour le modèle déterministe suit l'équation suivante pour déterminer l'état des parcelles à la prochaine
# génération : L'état de la parcelle à la génération suivante = l'état de la parcelle à la génération 
# courante * la probabilité de transition associée à ce changement d'état. Dans ce modèle, la taille de 
# la population est considérée comme infinie, puisqu'elle n'a pas d'effet sur le résultat de la simulation.
# Ce modèle représente donc une dynamique des populations plutôt thérique, qui manque un peu de réaliste puisqu'il
# ne simule pas la variance des résultats.

# # Présentation du modèle

# Ce modèle simule, à la base, 3 états finis de parcelles différentes : vide, herbes ou buisson. 
# Ceux-ci peuvent être représentés dans le modèle par un vecteur représentant le nombre de parcelles dans 
# chaque état à un temps donné. Ces états sont définis dans le contexte de l'aménagement d'un corridor
# sous une ligne électrique à haute tension qui subit les effets de la transition végétale.

# Les transitions entre ces états, tant qu'à elles, sont représentées par une matrice de transition qui est
# composées des différents poids (probabilités) de chaque transition. Chaque ligne de la matrice de transition
# doit avoir une somme de 1 afin de garder un nombre de parcelle égal à chaque génération.

# Avec le modèle présenté ci-dessous, nous nous demandons : quelle est la matrice de transition qui permet,
# en ajoutant un état, de respecter des critères d'abondance des états dans 80% des simulations. Ces critères 
# d'abondance sont les suivant : à l'équilibre dans la simulation, il faut que 20% des parcelles soient
# végétalisées et que parmi ces 20%, 30% soient des herbes, et 70% soient des buissons. Il faut ensuite
# que la variété de buisson la moins abondante ne représente pas moins de 30% du total des parcelles occupées
# par des buissons.

# De plus, nous devons débuter la simulation avec un corridor de 200 parcelles vides, parmis lesquelles nous 
# pouvons en végétaliser un maximum de 50 parcelles avec un mélange de buisson.

# Notre hypothèse se présente sous la forme de la matrice du nombre de parcelle se trouvant dans chaque état, 
# que nous supposons que le modèle respectera à l'équilibre pour respecter les critères énumérés ci-haut.

# # Implémentation

# ## Packages nécessaires

import Random
Random.seed!(123456)
using CairoMakie
using Distributions
using LinearAlgebra 

## Code 

#FONCTIONS

# Corrige la marice de transition, afin qu'elle suive le modèle de Markov ( programmation défensive )
# Simulation Non paramétrique puisque l'historique des états n'intervient pas (les transitions précédentes n'affecte pas les prochaines)
# La fonction "check_transition_matrice"  permet de normaliser la matrice, afin que la somme des probabilité de chaques transition possibles pour un état 
# initial donné sois de 100% ( chaques lignes = 1)

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
initial_states = [200, 0, 0, 0]  # Commence avec 200 parcelles vides ( petit effectif )
# L'effet de stochasticité est grand, puisque la population est petite
states = length(initial_states) # nombres d'états possibles 


# Création de la matrice de transition
T = zeros(Float64, states, states)

# Nous essayons différentes valeurs de contenu de matrice pour nous approcher de la solution
# La fonction suivante permet de vérifier si la proportion des états à l'équilibre selon la matrice
# donne le résultat indiqué dans les consignes

T[1, :] = [300, 10, 0, 0] # une parcelle vide a 30 fois plus de chance de rester vide que de devenir gazon

T[2, :] = [2, 50, 3, 4] # une parcelle de gazon a plus de chance de rester gazon. Elle a plus de chance de devenir Lila que Rose, et une petite chance de devenir vide

T[3, :] = [30, 0, 70, 0] # une parcelle de Rose a 30% de chance de devenir vide, aucune chance d'être remplacée par du gazon ou lila

T[4, :] = [27, 0, 0, 73] # Une parcelle de lila à 27% de chance de devenir vide, aucune chance d'ête remplacée par du gazon ou lila 

# CALCUL DES PROPORTIONS À L'ÉQUILIBRE (vecteur propre associé à la plus grande valeur propre)
# À l'équilibre, les probabilité resteront les même lorsqu'ils sont multipliés par la matrice de transition 
#  État à l'équilibre : Vecteur_propre*T= vecteurs_propre = T*valeurs_propre (1)
# Cette fonction permet de vérifier la distribution des différents états à l'équilibre résultant des valeur posé dans la matrice de transition posés précedament 
# pour que à l'équilibre 20% des parcelles soient végétalisées, et que parmi ces 20%, 30% soient des herbes, et 70% soient des buissons et ue la variété de
# buisson la moins abondante ne représente pas moins de 30% du total des parcelles occupées par des buissons le vecteur propre souhaité est :

proportions_souhaitees = [0.8,0.06,0.05,0.09] # ( vide, gazon, Rose, Lila)

"""
    Verification_resultat_equilibre(arg1, arg2)

La fonction vérifie que la matrice de transition permet d'atteindre les porportions de chaque état 
possible lorsque la simulation atteint l'équilibre selon une certaine marge d'erreur.

# Arguments 
arg1 = une matrice de transition.
arg2 = un vecteur représentant les proportions de chaque états souhaitées à l'équilibre.

# Retour
La fonction retourne les proportions calculées à l'équilibre et nous informe si la matrice de transition
respecte ou non les proportions souhaitées.
"""

function Verification_resultat_equilibre(T,proportions_souhaitees)
  # Calcul les valeurs et vecteurs propre de la matrice de transition
  valeurs_propres, vecteurs_propres = eigen(T)
  # Trouver l'indice du vecteur propre associé à la valeur propre qui se rapproche le plus de 1 (correspond au vecteur à l'équilibre)
  indices_vecteur_propre_equilibre = findmin(abs.(valeurs_propres.-1)) 
  # normalise le vecteur propre pour que la somme de la ligne = 1 et faire une distribution de probabilité à l'équilibre 
  proportions_calculees= vecteurs_propres[indices_vecteur_propre_equilibre]./sum(vecteurs_propres[indice_vecteurs_propre_equilibre])


  #Comparer avec la distribution souhaité
 marge_erreur = 0.20 # marge d'erreur de 20% accepté

 if maximum(abs.(proportions_calculees.-proportions_souhaitees)) < marge_erreur
    println("La matrice de transition est adéquate")
   else 
    throw("La matrice ne permet pas d'atteindre les proportions souhaités à l'équilibre")
   end 
 return proportions_calcules
end 

"""
    resultat(arg1, arg2)

La fonction évalue le pourcentage de réussite de la simulation stochastique. Une réussite est 
lorsque les critères de proportions de chaque état sont respectés.

# Arguments 
arg1 = une matrice représentant les états de la parcelle pour chaque génération.
arg2 = un vecteur représentant les proportions de chaque états souhaitées à l'équilibre.

# Retour
La fonction retourne le pourcentage de réussite de la simulation stochastique.
"""

#PARAMETRE DU GRAPHIQUE

#Légende et couleurs associé
states_names = ["vide", "Gazon", "Rose","Lila"]
states_colors = [:grey40, :green, :pink, :purple]

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
