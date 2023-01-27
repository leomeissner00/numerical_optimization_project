@doc doc"""
#### Objet

Résolution des problèmes de minimisation avec une contrainte d'égalité scalaire par l'algorithme du lagrangien augmenté.

#### Syntaxe
```julia
xmin,fxmin,flag,iter,muks,lambdaks = Lagrangien_Augmente(algo,f,gradf,hessf,c,gradc,hessc,x0,options)
```

#### Entrées
  - algo : (String) l'algorithme sans contraintes à utiliser:
    - "newton"  : pour l'algorithme de Newton
    - "cauchy"  : pour le pas de Cauchy
    - "gct"     : pour le gradient conjugué tronqué
  - f : (Function) la fonction à minimiser
  - gradf       : (Function) le gradient de la fonction
  - hessf       : (Function) la hessienne de la fonction
  - c     : (Function) la contrainte [x est dans le domaine des contraintes ssi ``c(x)=0``]
  - gradc : (Function) le gradient de la contrainte
  - hessc : (Function) la hessienne de la contrainte
  - x0 : (Array{Float,1}) la première composante du point de départ du Lagrangien
  - options : (Array{Float,1})
    1. epsilon     : utilisé dans les critères d'arrêt
    2. tol         : la tolérance utilisée dans les critères d'arrêt
    3. itermax     : nombre maximal d'itération dans la boucle principale
    4. lambda0     : la deuxième composante du point de départ du Lagrangien
    5. mu0, tho    : valeurs initiales des variables de l'algorithme

#### Sorties
- xmin : (Array{Float,1}) une approximation de la solution du problème avec contraintes
- fxmin : (Float) ``f(x_{min})``
- flag : (Integer) indicateur du déroulement de l'algorithme
   - 0    : convergence
   - 1    : nombre maximal d'itération atteint
   - (-1) : une erreur s'est produite
- niters : (Integer) nombre d'itérations réalisées
- muks : (Array{Float64,1}) tableau des valeurs prises par mu_k au cours de l'exécution
- lambdaks : (Array{Float64,1}) tableau des valeurs prises par lambda_k au cours de l'exécution

#### Exemple d'appel
```julia
using LinearAlgebra
algo = "gct" # ou newton|gct
f(x)=100*(x[2]-x[1]^2)^2+(1-x[1])^2
gradf(x)=[-400*x[1]*(x[2]-x[1]^2)-2*(1-x[1]) ; 200*(x[2]-x[1]^2)]
hessf(x)=[-400*(x[2]-3*x[1]^2)+2  -400*x[1];-400*x[1]  200]
c(x) =  (x[1]^2) + (x[2]^2) -1.5
gradc(x) = [2*x[1] ;2*x[2]]
hessc(x) = [2 0;0 2]
x0 = [1; 0]
options = []
xmin,fxmin,flag,iter,muks,lambdaks = Lagrangien_Augmente(algo,f,gradf,hessf,c,gradc,hessc,x0,options)
```

#### Tolérances des algorithmes appelés

Pour les tolérances définies dans les algorithmes appelés (Newton et régions de confiance), prendre les tolérances par défaut définies dans ces algorithmes.

"""
function Lagrangien_Augmente(algo,fonc::Function,contrainte::Function,gradfonc::Function,
        hessfonc::Function,grad_contrainte::Function,hess_contrainte::Function,x0,options)

  if options == []
		epsilon = 1e-2
		tol = 1e-5
		itermax = 1000
		lambda0 = 2
		mu0 = 100
		tho = 2
	else
		epsilon = options[1]
		tol = options[2]
		itermax = options[3]
		lambda0 = options[4]
		mu0 = options[5]
		tho = options[6]
	end

  #Definition des constantes
  beta = 0.9
  eta = 0.1258925
  alpha = 0.1

  #Definition des variables
  xmin = x0
  epsilonk = epsilon
  muk = mu0
  lambdak = lambda0
  etak = eta/(muk^alpha)
  muks = [muk]
  lambdaks = [lambdak]

  #Definition des fonctions
  function L(x, lambda)
    return fonc(x) + lambda * contrainte(x)
  end

  function DeltaL(x, lambda)
    return gradfonc(x) + lambda * grad_contrainte(x)
  end

  function fLa(x)
    return fonc(x) + lambdak' * contrainte(x) + muk/2 * norm(contrainte(x))^2
  end

  function gradfLa(x)
    return gradfonc(x) + lambdak' * grad_contrainte(x) + muk * contrainte(x) * grad_contrainte(x)
  end

  function hessfLa(x)
    return hessfonc(x) + lambdak' * hess_contrainte(x) + muk * contrainte(x) * hess_contrainte(x)' + muk * grad_contrainte(x) * grad_contrainte(x)'
  end
 
  flag = 2
  iter = 0
  #Test de vérification des parametres d'entree
  if norm(DeltaL(xmin,lambdak)) <= max(tol * norm(DeltaL(x0,lambda0)),tol) && norm(contrainte(xmin)) <= max(tol * norm(contrainte(x0)),tol)
    flag = 0;
  elseif iter >= itermax
    flag = 1;
  end  

  while flag == 2
    #Recuperation des parametres suivant l'algo
    if algo == "newton"
      xmin,_,_,_ = Algorithme_De_Newton(fLa,gradfLa,hessfLa,xmin,[])
    elseif algo == "gct"
      xmin,_,_,_ = Regions_De_Confiance("gct",fLa,gradfLa,hessfLa,xmin,[])
    elseif algo == "cauchy"
      xmin,_,_,_ = Regions_De_Confiance("cauchy",fLa,gradfLa,hessfLa,xmin,[])
    else
      println("algo non defini")
      flag = -1
    end
    
    if norm(contrainte(xmin)) <= etak 
      # Mettre a jour (entre autres) les multiplicateurs
      lambdak = lambdak + muk * contrainte(xmin)
      epsilonk = epsilonk/muk    
      etak = eta/(muk^beta)

    else
      # Mettre a jour (entre autres) les parametres de penalite
      muk = tho*muk
      epsilonk = epsilon/muk
      etak = eta/(muk^alpha)
    end

    #Mise a jour des variables de retours
    muks = hcat(muks, [muk])
    lambdaks = hcat(lambdaks, [lambdak])
    iter = iter + 1
    # Conditions d'arret
    if norm(DeltaL(xmin,lambdak)) <= max(tol * norm(DeltaL(x0,lambda0)),tol) && norm(contrainte(xmin)) <= max(tol * norm(contrainte(x0)),tol)
      flag = 0;
    elseif iter + 1 > itermax
      flag = 1;
    end
  end
  fxmin = fonc(xmin) 
        return xmin,fxmin,flag,iter,muks,lambdaks
end

        
