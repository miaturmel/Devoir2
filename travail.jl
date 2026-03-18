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
# Le choix manuel des valeurs de la matrice permet de s'assurer que les transitions d'État font du sens biologiquement

T[1, :] = [100, 1, 0, 0] # Une parcelle vide à beaucoup plus de chance de rester vide que de devenir gazon

T[2, :] = [80, 25, 1, 13] # Transition possible d'une parcelle gazon

T[3, :] = [80, 0, 20, 0]  # Transition possible d'une parcelle Rose

T[4, :] = [77, 0, 0, 25]  # Transition possible d'une parcelle Lila

T_normal = check_transition_matrix!(T)
println(T_normal)

proportions_souhaitees = [0.8,0.06,0.05,0.09] # ( vide, gazon, Rose, Lila)

# CALCUL DES PROPORTIONS À L'ÉQUILIBRE (vecteur propre associé à la plus grande valeur propre)
# À l'équilibre, les probabilité resteront les même lorsqu'ils sont multipliés par la matrice de transition 
# Cette fonction permet de vérifier la distribution des différents états à l'équilibre résultant des valeur posé dans la matrice de transition posés précedament 
# pour que à l'équilibre 20% des parcelles soient végétalisées, et que parmi ces 20%, 30% soient des herbes, et 70% soient des buissons et ue la variété de
# buisson la moins abondante ne représente pas moins de 30% du total des parcelles occupées par des buissons le vecteur propre souhaité est :

"""
    Verification_resultat_equilibre(arg1, arg2)

La fonction vérifie que la matrice de transition permet d'atteindre les porportions de chaque état 
possible lorsque la simulation atteint l'équilibre selon une certaine marge d'erreur.

# Arguments 
arg1 = une matrice de transition normalisée .
arg2 = un vecteur représentant les proportions de chaque états souhaitées à l'équilibre.

# Retour
La fonction retourne les proportions calculées à l'équilibre et nous informe si la matrice de transition
respecte ou non les proportions souhaitées.
"""

function Verification_resultat_equilibre(T_normal,proportions_souhaitees)
    
  # Calcul les valeurs et vecteurs propre de la matrice de transition
  valeurs_propres, vecteurs_propres = eigen(T_normal)
  # Trouver l'indice du vecteur propre associé à la valeur propre qui se rapproche le plus de 1 (correspond au vecteur à l'équilibre)
  _,indices_vecteur_propre_equilibre = findmin(abs.(valeurs_propres.-1)) # retourne juste l'index, pas la valeur qui est donné par la fonction findmin
  # normalise le vecteur propre pour que la somme de la ligne = 1 et faire une distribution de probabilité à l'équilibre 
  proportions_calculees= (vecteurs_propres[:,indices_vecteur_propre_equilibre])./sum(vecteurs_propres[:,indices_vecteur_propre_equilibre]) 

  #Comparer avec la distribution souhaité
 marge_erreur = 0.05 # marge d'erreur de 5% accepté
  
 # Envoie un message d'erreur si les probabilités calculés pour chaques état a l'équilibre selon la matrice de transition ne correspond pas
 # aux proportions souhaités, avec une marge d'erreur de 5% accepté
 if maximum(abs.(proportions_calculees.-proportions_souhaitees)) < marge_erreur
    println("La matrice de transition est adéquate")
   else 
    println("La matrice ne permet pas d'atteindre les proportions souhaités à l'équilibre")
   end 
 return proportions_calculees
end 

# Créer un objet avec le résultat de la fonction, ce qui permet de faire le message d'erreur si besoin
proportions_calculees = Verification_resultat_equilibre(T_normal,proportions_souhaitees)
println(proportions_calculees) # Donne les proportions acceptable atteint avec la matrice de transition trouvée par essais-erreur

# Une fois la matrice de transition déterminée, vérifier si les critères sont respectés dans au moins 80% des simulations
# function Verification_critères_fixés (proportions_calculees, T, Timesseries, )

# function resultats (timeseries, proportions_souhaitees)

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

function resultat(timeseries, proportion_souhaitees)

    #objet qui stock le nombre de fois que les critères fixés sont respectés 
    resultat_ok = 0

    for _ in 1:100 #répéter la simulation stochaistique 100 fois
        sto_sim = simulation( T, initial_states; stochastic = true, generations = 200)

        derniere_gen = sto_sim[:, end] # derniere generation de chaques simulation, donc état a l'équilibre
        proportion_last_gen = derniere_gen./sum(derniere_gen) # normaliser pour avoir les pourcentage de chaque état

        # Comparer avec les proportions souhaités à l'équilibre
       marge_erreur = 0.01 # marge d'erreur de 1% accepté
       
       if maximum(abs.(proportion_last_gen .- proportions_souhaitees)) < marge_erreur 
        # si la différence entre l'état a l'équilibre d'une simulation et l'état à l'équilibre souhaité est inférieur à 1%
        resultat_ok =+ 1 # considéré dans les simulations réussie
       end 
    end 

  # Calcul du pourcentage de simulation " réussie", qui atteint les critères fixés

  pourcentage_reussite = resultat_ok / 100 # combien de fois de fois la simulation est réussie sur les 100 simulation stochastiques

  return pourcentage_reussite

end 

# Affiche le résultat de la fonction, et indique si la stochasticité permet de maintenir l'atteinte des critère plus ou moins de 80% des simulations
pourcentage_reussite = resultat(timeseries, proportions_souhaitees) # objet avec le pourcentage de réussite 
println(" Les critère fixés sont atteints dans ",(pourcentage_reussite), "% des simulations")

#PARAMETRE DU GRAPHIQUE

#Légende et couleurs associé
states_names = ["vide", "Gazon", "Rose","Lila"]
states_colors = [:grey40, :green, :pink, :purple]

# Simulations sur le graphique
f = Figure()
ax = Axis(f[1, 1], xlabel="Nb. générations", ylabel="Nb. parcelles")

# Zoom 1 sur 10 % des 200 parcelles ( gazon, lila, Rose)
 limits!(ax, 0, 100, 0, 20)

# Zoom 2 sur les parcelles vide  
# limits!(ax, 0, 100, 180, 200)

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

# _MATRICE DE TRANSITION DES ESPÈCES QUI PERMET D'OBTENIR LES CRITÈRES FIXÉS_
# Aucune combinaison de valeurs essayé dans la matrice de transition n'as permit d'atteindre les proportions de chaques états voulues. En effet, la fonction 
# " Verification_resultat_equilibre" retourne des états à l'équilibre dans une proportion de 
# [0.24999999999999997, 0.25000000000000006, 0.24999999999999997, 0.24999999999999997] 
# représentant respectivement les parcelles vides, gazon, rose et lila peut importe les valeurs mise dans la matrice. 
# Un probléme empêche donc la fonction de guider le choix de valeurs dans la matrice de transition. Si  " Verification_resultat_equilibre" serait fonctionnelle, 
# il serait possible de trouver des valeurs à l'équilibre se rapprochant des critères fixés. 


# Toutefois, Une matrice de transition qui tente à de reproduire les proportions désirés à été déterminée en regardant les résultats graphiques de différentes
# combinaison de valeur.
# À partir de celle-ci, il est possible de visualiser les résultats de simulations déterministes et stochastiques :

# _RÉSULTAT DU MODEL DÉTERMINISTE_

##### A FINIR

# global
# zoom 1
# zoom 2


# _RÉSULTAT DU MODEL STOCHASTIQUE_


#### A FIINIR
# global
# zoom 1
# zoom 2


# _GARENTIE DU RESPECT DES CRITÈRES FIXÉS_ 
# Afin d'évaluer l'ampleur des variations de proportion des états à l'équilibre dans les simulations stochastiques, la fonction "resultat" calcul de pourcentage de 
# simulation où les critères fixés sont respectés, sur 100 simulations stochastique. 
# Puisque nous n'avons pas réussi à déterminer une matrice de transition qui permet d'atteindre les critères par la simulation déterministe, il est impossible d'utiliser
# la fonction " resultat" pour évaluer les résultats ( les critères sont respectés dans 0 % des cas). 
# Si la matrice de transition aurait pu déterminés, cette fonction aurait permis de vérifier si l'effet stochastique était assez important pour 
# Toutefois, en analysant les graphiques avec un zoom, l'ampleur de la variation dans les simulations stochastiques laissent croire que ..........

#### A FINIR 

# _EFFET DU NOMBRE DE PARCELLE À L'ÉTAT FINAL SUR LA STOCHASTICITÉ_
# La matrice de transition utilisée favorise grandement les parcelles vide, alors que les arbuste ( rose et lila ) et le gazon représentais individuellement moins de 2%.
# En analysant les zooms 1 et 2, il est possible de ......


#### A FINIR 

# La figure suivante représente des valeurs aléatoires:
#hist(randn(100))

# # Discussion

# _ÉTAT À L'ÉQUILIBRE_
# Nous considérons que ce modèle suit le model de Markov (le nombre de parcelles dans chaque état dépend seulement de la génération actuelle, pas de celles d’avant).
# Pour le model déterminisite, lorsque l'ont multiplie les états de la génération actuelle par la matrice de transition, le résultat est les états à la génération suivante.
# Ainsi, il est possible d'en déduire qu'à l'équilibre : Vecteur_propre*T= vecteurs_propre. 
# En d'autres mots, la distribution à l'équilibre  correspond au vecteur propre associé à la valeur propre 1 de la matrice de transition.
#
# _ÉTAT INITIAL_
# Nous détbutons avec un terrain vierge de 200 parcelles vides

# _MATRICE DE_TRANSITION_
# Il aurait été possible d'isoler la matrice de transition qui donne les proportions souhaités à l'équilibre mathématiquement. 
# Toutefois, cette  matrice de transition de ce modèle de succession végétale doit également respecter des **contraintes biologiques**. 
#
# Lorsque nous avont choisit les valeurs de la matrice de transition, nous avons considérés que : 
# 1. Une parcelle doit d'abord devenir gazon avant de pouvoir devenir arbuste
# 2. Une parcelle d'une espèce d'arbuste ( Lilas ou Rose ), doit devenir vide pour pouvoir ensuite changer d'espèce. Par exemple, pour parcelle occupée par un rosier, 
#   le rosier doit mourrir pour qu'un lilas puisse y pousser dans les prochaines generations. On peut ainsi représenter la compétition interspécifique
# 3. Le gazon a plus de chance de mourrir qu'être remplacé par un arbuste. En effet, dans ce terrain rasé par les activité anthropiques, le sol est sèche et pauvre.
# 4. Le type de sol, la luminosité  et les conditions environnementale favorise d'avantage le lilas que le rosier. Les graines de Lilas germent 13 fois mieux que celle de rosier
#   dans le gazon. De plus, la mortalité à chaque generation est de 80% pour le rosier, comparativement a 75% pour le lilas. 
# 5. Lorsqu'un arbuste meurt, la parcelle devient vide, en raison de la perturbation des conditions du sol et des microorganismes de décomposition.
#
#
# _EFFET DE STOCHASTICITÉ_ 
#
#  #### a FINIR : idée : nb de parcelle, nb de générations,  distribution des états ( bcp plus de vide), retour sur les observations graphiques...
#
#_LIMITES DU MODEL_ 
# D'une part, déterminer une matrice de transition qui permet d'atteindre suffisament les critères fixés, tout en respectant les contraintes biologiques oblige à faire certaines
# concessions sur le réalisme du modèle. De plus, plusieurs éléments qui régule la succession végétale sont négligés, tel que les conditions environnementales, et l'écosystème 
# du paysage ( prédateurs, polinisateurs, activités anthropiques, autres espèces végétales, effet des saisons...). Ces facteurs pourrait modifier la distribution des états et 
# perturber l'équilibre. La supposition que la succession végétale suis le model de Markov peut également limiter le réalisme du model. En réalité, la présence d'une espèce 
# d'arbuste dans une generation antérieur pourrait augmenter la chance que cette espèce d'arbuste resurgise en raison de sa banque de graine. 
# 
#  ### a FINIR : si tu as des idées a ajouter
#
 

# On peut aussi citer des références dans le document `references.bib`,
# @ermentrout1993cellular -- la bibliographie sera ajoutée automatiquement à la
# fin du document.
