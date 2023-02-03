from matplotlib import style
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import random

#Génération d'une matrice aléatoire de taille n avec des valeurs entre a et b
# a doit être strictement inférieur à b
#La valeur percent doit être entre 0 et 100
#Le pourcentage représente le pourcentage de valeurs non infini
def random_matrix(n,a,b,percent):
    K = np.zeros((n,n))
    nums = percent * [1]*100 + (100 - percent) * [0]*100
    random.shuffle(nums)
    for i in range(n):
        # nbRéels = b-a*10
        for j in range(n):
            if random.choice(nums) == 1:
                K[i,j] = round(random.uniform(a,b),2)
            else :
                K[i,j] = float('inf')
    return K

#Remplace les numéros des sommets par leur lettre
def int_list_to_chr(l):
    k=[]
    for i in l:
        if i == 'None':
            k.append('None')
        else:
            k.append(chr(int(i)+65))
    return k

#Affiche le graphe grâce à networkx
def print_Matrice(G):
    G = Matrice_Vers_Graphes(G)
    #A partir d'un graphe dirigé 
    edges = [(u, v) for (u, v,d) in G.edges(data=True)]
    #Choisi la place des noeuds
    pos = nx.spring_layout(G)
    #Configuration graphique des noeuds
    nx.draw_networkx_nodes(G, pos, node_size=500)
    #Configuration graphique des arrêtes
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2)
    #Configuration graphique des légendes
    nx.draw_networkx_labels(G, pos, font_size=14, font_family="sans-serif")
    #Définition du poids des arrêtes
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    ax = plt.gca()
    ax.margins(0.1)
    #Enlever les axes
    plt.axis("off")
    #Ajuste la taille de la figure automatiquement pour qu'elle rentre dans la fenêtre
    plt.tight_layout()
    #On affiche le graphe
    plt.show()

#Converti une matrice en liste de flèches pour pouvoir l'affiché
def Matrice_Vers_Graphes(M):
    #Création d'un graphe dirigé
    G = nx.DiGraph()
    size = np.shape(M)[0]
    #On parcours la matrice
    for i in range(size):
        for j in range(size):
            #Si il y a une arrête entre i et j on l'ajoute avec son poids dans le graphe
            if M[i,j] != float('inf'):
                G.add_edge(i,j,weight=M[i,j])
    return G
#Algorithme de Dijkstra, retourne tout les chemins les plus courts qui partent de s
def Dijkstra(M,s):
    #Initialisation
    M = np.array(M)
    size = np.shape(M)[0]
    #Création de la table distance et du tableau des prédécesseurs
    table_dist = [float('inf')]*size
    table_pred = ['None']*size    
    table_dist[s] = 0
    choice = []
    #Création de la liste des choix
    choice.append(s)
    choix = s
    # Tant qu'on a pas choisi tous les sommets
    while len(choice) < size:
         # On prends le premier élément jamais choisi
        new_choix=0
        while new_choix in choice:
            new_choix+=1
        for i in range(size):
            #Si le chemin est plus court on garde le prédecesseur et la distance
            if i not in choice and table_dist[i] > (table_dist[choix] + M[choix,i]):
                table_dist[i] = table_dist[choix] + M[choix,i]
                table_pred[i] = choix
            #Si on a trouvé un chemin plus court et qui n'a pas été choisi on le garde pour le prochain choix
            if i not in choice and table_dist[i] < table_dist[new_choix]:
                new_choix = i
        choix = new_choix       
        choice.append(choix)
        
    #Marquer tout les sommets non joignables
    for i in range(size):
        if table_dist[i] == float('inf'):
            table_dist[i] = "sommet non joignable à s"+str(s)+" par un chemin dans le graphe G"
    # Retourner un dictionnaire avec les distances et les prédécesseurs
    table_pred = int_list_to_chr(table_pred)
    sommets = int_list_to_chr(range(size))
    # return choice
    return zip(sommets,table_dist,table_pred)

#Retourne la liste de flèche pour Bellman Ford dans l'ordre alphabétique
def liste_fleches(M):
    size = np.shape(M)[0]
    k = []
    for i in range(size):
        for j in range(size):
            if M[i,j] != float('inf'):
                tuple = (i,j,M[i,j])
                k.append(tuple)
    return k
# Algorithme de Bellman Ford
def Bellman_Ford(M,s,liste_fleches):
    size = np.shape(M)[0]
    K = liste_fleches
    #Initialisation
    table_dist = [float('inf')]*size
    table_pred = ['None']*size
    if M[s,s] == float('inf'):
        table_dist[s] = 0
    else:
        table_dist[s] = M[s,s]
    previous = []*size
    tours = 0
    isChanged = True
    while isChanged and tours <= size:
        isChanged = False
        previous = table_dist.copy()
        for i,j,poids in K:
            if table_dist[i] != float('inf') and table_dist[j] > table_dist[i] + poids:
                table_dist[j] = table_dist[i] + poids
                table_pred[j] = i
                isChanged = True
        tours += 1
    #Mise en forme des résultats
    for i in range(size):
        if table_dist[i] == float('inf'):
            table_dist[i] = "sommet non joignable à s"+str(s)+" par un chemin dans le graphe G"
        if table_dist[i] != previous[i]:
            table_dist[i] = "pas de plus court chemin : présence d’un cycle de poids négatif"
    # Retourner un dictionnaire avec les distances et les prédécesseurs
    return zip(int_list_to_chr(range(size)),table_dist,int_list_to_chr(table_pred))
        
# Parcours Largeur d'un graphe
def Pl(M,s):
    size = len(M)
    _file=[s]
    visite = [s]
    while len(_file)>0:
        for i in range(size):
            if M[_file[0],i] == 1 and i not in visite:
                visite.append(i)
                _file.append(i)
        _file.pop(0)
    return visite

#Parcours Profondeur d'un graphe
def Pp(M,s):
    Couleur={}
    Parcours = [s]
    SommetBlanc = []
    for i in range(len(M)):
        Couleur[i]='Blanc'
    Couleur[s]='Vert'
    Pile = [s]
    while Pile != []:
        i = Pile[-1]
        for j in range(len(M)):
            if M[i,j] == 1 and Couleur[j] == 'Blanc':
                SommetBlanc.append(j)
        if SommetBlanc != []:
            Parcours.append(SommetBlanc[0])
            Pile.append(SommetBlanc[0])
            Couleur[SommetBlanc[0]]=chr(i+65)
            SommetBlanc.clear()
        else:
            Pile.pop()
        
    return Parcours
#Liste flèche parcours largeur
def liste_fleches_pl(M,s):
    k = []
    size = len(M)
    parcours = Pl(M,s)
    for i in parcours:
        for j in range(size):
            if M[i,j] != INF:
                tuple = (i,j,M[i,j])
                k.append(tuple) 
    if (size != len(parcours)):
        for i in range(size):
            if i not in parcours:
                for j in range(size):
                    if M[i,j] != INF:
                        tuple = (i,j,M[i,j])
                        k.append(tuple)
    return k

#Liste flèche parcours profondeur
def liste_fleches_pp(M,s):
    k = []
    size = len(M)
    parcours = Pp(M,s)
    for i in parcours:
        for j in range(size):
            if M[i,j] != INF:
                tuple = (i,j,M[i,j])
                k.append(tuple) 
    if (size != len(parcours)):
        for i in range(size):
            if i not in parcours:
                for j in range(size):
                    if M[i,j] != INF:
                        tuple = (i,j,M[i,j])
                        k.append(tuple)
    return k
def perf_compare(size):
    M = random_matrix(size,0,5,50)
    fleches= liste_fleches(M)
    fleches_pp = liste_fleches_pp(M,0)
    fleches_pl = liste_fleches_pl(M,0)
    print("Parcours Profondeur")
    start = time.perf_counter()
    Bellman_Ford(M,0,fleches_pp)
    stop = time.perf_counter()
    print("Temps d'exécution : ",stop-start)
    print("Parcours Largeur")
    start = time.perf_counter()
    Bellman_Ford(M,0,fleches_pl)
    stop = time.perf_counter()
    print("Temps d'exécution : ",stop-start)
    print("Ordre alphabétique")
    start = time.perf_counter()
    Bellman_Ford(M,0,fleches)
    stop = time.perf_counter()
    print("Temps d'exécution : ",stop-start)

def perfDijsktra(maxSize):
    l=[]
    for i in range(2,maxSize):
        M = random_matrix(i,1,5,50)
        start = time.perf_counter()
        Dijkstra(M,0)
        stop = time.perf_counter()
        l.append(stop-start)
    return l
def perfBellman(maxSize):
    l=[]
    for i in range(2,maxSize):
        M = random_matrix(i,1,5,50)
        start = time.perf_counter()
        Bellman_Ford(M,0,liste_fleches(M))
        stop = time.perf_counter()
        l.append(stop-start)
    return l
def comparaisonAlgorithmique(maxSize):
    plt.plot(perfDijsktra(maxSize),label="Dijkstra",color="red")
    plt.plot(perfBellman(maxSize), label='Bellman-Ford',color="blue")
    plt.legend()
    plt.figure("comparaison temps d'exécution")
    plt.show()
# k = Matrice_Vers_Graphes(random_matrix(4,1,4,50))
INF=float('inf')
MATRIX = np.matrix([[INF,8,6,2],[INF,INF,INF,INF],[INF,3,INF,INF],[INF,5,1,INF]])
MATRIX_NEGATIVE = np.matrix([[INF,8,6,2],[INF,INF,INF,INF],[-3,3,INF,INF],[INF,5,1,INF]])
MATRIX_CYCLE_NEGATIF = np.matrix([[INF,8,6,2],[INF,INF,INF,INF],[-4,3,INF,INF],[INF,5,1,INF]])


#Exemples d'utilisations
#Bellman ford matrice exemple et ces 3 ordres de flèches
# for items in Bellman_Ford(MATRIX,0,liste_fleches(MATRIX)):
#     print(items)
# for items in Bellman_Ford(MATRIX,0,liste_fleches_pp(MATRIX)):
#     print(items)
# for items in Bellman_Ford(MATRIX,0,liste_fleches_pl(MATRIX)):
#     print(items)

#Dijkstra
# for items in Dijkstra(MATRIX,0):
#    print(items)

#Afficher un graphe à partir d'une matrice
#Attention cette fonction arrête l'éxécution du programme l'utilisé à la fin du code
# print_Matrice(MATRIX)

#Comparer l'importance de l'ordre des flèches sur une matrice de taille 150
#perf_compare(150)

#Comparer les deux algorithmes sur une matrice de taille 150
#comparaisonAlgorithmique(150)

for items in Bellman_Ford(MATRIX_CYCLE_NEGATIF,0,liste_fleches(MATRIX_CYCLE_NEGATIF)):
    print(items)

print_Matrice(MATRIX_NEGATIVE)